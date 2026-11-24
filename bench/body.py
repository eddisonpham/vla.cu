"""Split the int4 GEMM body at 50 tokens into its two candidate costs.

Concept: the launch is named (EXP-076): an empty kernel is 9 us and cuBLAS bf16 does the
same work in 18.5 us, so 38 to 47 us is inside the kernel. Two candidates remain, the
number of dependent K steps and the bytes each step reads. Changing the K step size at a
fixed shape separates them, because a wider step means fewer dependent iterations over
the same bytes, while varying K at a fixed step size moves the bytes and the steps
together. The int8 GEMM of the same shape, which needs no group scale expansion, prices
the dequantize path. The expert width 720 pads to 768 for group 64, which is the same pad
the int4 paths already pay.

Task: python -m vla_cu body
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemm import int4_gemm_into
from kernels.triton.int8_gemm import int8_gemm_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    time_pair,
    warm_gpu,
)
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8

TOKENS = 50
N_OUT = 1920
STEP_SHAPE = 768
STEP_TILES = ((64, 128, 64, 2), (64, 128, 128, 2), (64, 64, 256, 1))
K_VALUES = (768, 1536, 3072)
K_TILE = (64, 128, 128, 2)
WARMUP = 20
ITERATIONS = 100


def main() -> int:
    """Time the same shape with different K step sizes, K lengths and formats."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | {TOKENS} tokens, {N_OUT} outputs")
    print(f"warmup {WARMUP}, iterations {ITERATIONS}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((TOKENS, STEP_SHAPE), device="cuda", generator=generator)

    print()
    print("one shape, three K step sizes: fewer dependent steps over the same bytes")
    print(f"{'tile':>22}  {'steps':>5}  {'MB':>6}  {'ms':>8}  {'us/step':>8}  {'GB/s':>7}")
    w = torch.randn((N_OUT, STEP_SHAPE), device="cuda", generator=generator) * (1.0 / STEP_SHAPE**0.5)
    packed, scales = pack_int4(w, group_size=GROUP_SIZE)
    out = torch.empty((TOKENS, N_OUT), device="cuda")
    n_bytes = N_OUT * STEP_SHAPE // 2 + 4 * N_OUT * (STEP_SHAPE // GROUP_SIZE) + 4 * TOKENS * STEP_SHAPE + 4 * TOKENS * N_OUT
    for tile in STEP_TILES:
        seconds = time_call(lambda c=tile: int4_gemm_into(packed, scales, x, out, tile=c), WARMUP, ITERATIONS)
        steps = STEP_SHAPE // tile[2]
        print(
            f"{str(tile):>22}  {steps:>5}  {n_bytes / 1e6:>6.2f}  {seconds * 1e3:>8.3f}  "
            f"{seconds * 1e6 / steps:>8.3f}  {bandwidth_gb_s(n_bytes, seconds):>7.1f}"
        )
    del w, packed, scales, out

    print()
    print("one K step size, three K lengths: bytes and steps move together")
    print(f"{'K':>6}  {'steps':>5}  {'MB':>6}  {'ms':>8}  {'us/step':>8}  {'GB/s':>7}  {'% DRAM':>7}")
    for n_in in K_VALUES:
        weights = torch.randn((N_OUT, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
        codes, weight_scales = pack_int4(weights, group_size=GROUP_SIZE)
        activations = torch.randn((TOKENS, n_in), device="cuda", generator=generator)
        result = torch.empty((TOKENS, N_OUT), device="cuda")
        moved = N_OUT * n_in // 2 + 4 * N_OUT * (n_in // GROUP_SIZE) + 4 * TOKENS * n_in + 4 * TOKENS * N_OUT
        seconds = time_call(
            lambda c=codes, s=weight_scales, a=activations, r=result: int4_gemm_into(c, s, a, r, tile=K_TILE),
            WARMUP,
            ITERATIONS,
        )
        gbs = bandwidth_gb_s(moved, seconds)
        steps = n_in // K_TILE[2]
        print(
            f"{n_in:>6}  {steps:>5}  {moved / 1e6:>6.2f}  {seconds * 1e3:>8.3f}  {seconds * 1e6 / steps:>8.3f}  "
            f"{gbs:>7.1f}  {fraction_of_peak(gbs):>6.1%}"
        )
        del weights, codes, weight_scales, activations, result

    print()
    print(f"formats at {TOKENS} tokens and {N_OUT} x {STEP_SHAPE}: dequantize against dense")
    print(f"{'path':>22}  {'MB':>6}  {'ms':>8}  {'GB/s':>7}")
    weights = torch.randn((N_OUT, STEP_SHAPE), device="cuda", generator=generator) * (1.0 / STEP_SHAPE**0.5)
    codes4, scales4 = pack_int4(weights, group_size=GROUP_SIZE)
    codes8, scales8 = quantize_int8(weights)
    activations = torch.randn((TOKENS, STEP_SHAPE), device="cuda", generator=generator)
    result = torch.empty((TOKENS, N_OUT), device="cuda")
    half_weights = weights.to(torch.bfloat16)
    half_activations = activations.to(torch.bfloat16)
    dense = torch.empty((TOKENS, N_OUT), device="cuda", dtype=torch.bfloat16)
    cases = (
        ("int4 group 64", codes4.numel() + scales4.numel() * 4 + 4 * TOKENS * STEP_SHAPE + 4 * TOKENS * N_OUT,
         lambda: int4_gemm_into(codes4, scales4, activations, result, tile=K_TILE)),
        ("int8 rows", codes8.numel() + scales8.numel() * 4 + 4 * TOKENS * STEP_SHAPE + 4 * TOKENS * N_OUT,
         lambda: int8_gemm_into(codes8, scales8, activations, result)),
        ("bf16 cuBLAS", half_weights.numel() * 2 + half_activations.numel() * 2 + 2 * TOKENS * N_OUT,
         lambda: torch.matmul(half_activations, half_weights.T, out=dense)),
    )
    for label, moved, call in cases:
        seconds = time_call(call, WARMUP, ITERATIONS)
        print(f"{label:>22}  {moved / 1e6:>6.2f}  {seconds * 1e3:>8.3f}  {bandwidth_gb_s(moved, seconds):>7.1f}")

    print()
    print("program count at 50 tokens: one 64 row block against four 16 row blocks")
    tall = (64, 128, 128, 2)
    narrow = (16, 128, 128, 2)
    tall_grid = -(-TOKENS // tall[0]) * -(-N_OUT // tall[1])
    narrow_grid = -(-TOKENS // narrow[0]) * -(-N_OUT // narrow[1])
    timing = time_pair(
        lambda: int4_gemm_into(codes4, scales4, activations, result, tile=tall),
        lambda: int4_gemm_into(codes4, scales4, activations, result, tile=narrow),
        50,
        5,
        10,
    )
    print(f"{str(tall):>22}  {tall_grid:>5} programs  {timing.a * 1e3:>8.3f} ms")
    print(f"{str(narrow):>22}  {narrow_grid:>5} programs  {timing.b * 1e3:>8.3f} ms")
    print(
        f"64 row tiles are {timing.ratio:.2f}x the time of 16 row tiles, range "
        f"{timing.ratio_low:.2f} to {timing.ratio_high:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

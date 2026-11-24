"""Is the action expert a batch 1 problem? Measure it at its real M.

Concept: SmolVLA predicts a chunk of 50 actions, and the action expert is a transformer
over those 50 tokens, so its linear layers see M = 50, not 1. That changes the kernel
question. A GEMV reads every weight once per call, so 50 calls read the weights 50
times, while one GEMM reads them once and reuses each byte 50 times. The two paths move
different numbers of bytes, so both a time column and a derived byte budget are shown.

The expert width is 0.75 times the VLM width (verified in the LeRobot config) applied
to the published 960 and 2560 of the SmolVLM2-500M backbone, so 720 and 1920. Note that 720 is not a multiple of
the 64-weight group: 720 = 11 * 64 + 16, so this benchmark has to fall back to group 16
for that shape, which puts 4 bytes of scale on every 16 weights instead of every 64.
That is a real constraint on the format, not a benchmark detail (DEC-020).

Task: python -m vla_cu m50
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemm import SHARED_LIMIT_BYTES, int4_gemm_into, shared_memory_bytes, tile_for
from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    time_pair,
    warm_gpu,
)
from vla_cu.quant import dequantize_int4, pack_int4

TOKENS = 50
EXPERT_HIDDEN = 720
EXPERT_INTERMEDIATE = 1920
GROUP_FOR_HIDDEN = 16
GROUP_FOR_INTERMEDIATE = 64
SHAPES = ((EXPERT_INTERMEDIATE, EXPERT_HIDDEN, GROUP_FOR_HIDDEN), (EXPERT_HIDDEN, EXPERT_INTERMEDIATE, GROUP_FOR_INTERMEDIATE))
WARMUP = 5
ITERATIONS = 20
REPEATS = 5
SPLITS = (1, 2, 4)
CANDIDATES = (
    (16, 64, 64, 2),
    (16, 64, 128, 3),
    (32, 64, 128, 3),
    (64, 64, 64, 2),
    (64, 64, 64, 3),
    (64, 64, 128, 2),
    (64, 128, 128, 2),
    (128, 64, 64, 2),
    (128, 64, 128, 2),
)


def split_buffers(tokens: int, n_out: int, splits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the partials and the zeroed arrival counters a one launch split needs."""
    block_m, block_n, _, _ = tile_for(tokens, n_out)
    tiles = -(-tokens // block_m) * -(-n_out // block_n)
    return torch.empty((splits, tokens, n_out), device="cuda"), torch.zeros(tiles, device="cuda", dtype=torch.int32)


def main() -> int:
    """Time the per-row GEMV loop against one int4 GEMM and one bf16 GEMM."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s")
    print(f"action expert: M = {TOKENS} tokens, widths {EXPERT_HIDDEN} and {EXPERT_INTERMEDIATE}")
    print(f"warmup {WARMUP} | iterations {ITERATIONS}")
    print()
    print(f"{'N x K':>12}  {'group':>5}  {'path':>16}  {'bytes':>9}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}  {'floor ms':>8}")

    for n_out, n_in, group in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn((n_out, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
        activations = torch.randn((TOKENS, n_in), device="cuda", generator=generator)
        packed, scales = pack_int4(w, group_size=group)
        out = torch.empty((TOKENS, n_out), device="cuda")

        half = dequantize_int4(packed, scales, group_size=group).to(torch.bfloat16)
        half_activations = activations.to(torch.bfloat16)
        half_out = torch.empty((TOKENS, n_out), device="cuda", dtype=torch.bfloat16)

        def gemv_loop() -> None:
            for token in range(TOKENS):
                int4_gemv_into(packed, scales, activations[token], out[token], group_size=group)

        def half_gemm() -> None:
            torch.matmul(half_activations, half.transpose(0, 1), out=half_out)

        partials = {s: split_buffers(TOKENS, n_out, s) for s in SPLITS if s > 1}

        def int4_gemm() -> None:
            int4_gemm_into(packed, scales, activations, out, group_size=group)

        def split_gemm(splits: int) -> None:
            workspace, counters = partials[splits]
            int4_gemm_into(
                packed,
                scales,
                activations,
                out,
                group_size=group,
                splits=splits,
                partial=workspace,
                counters=counters,
            )

        gemv_loop()
        half_gemm()
        int4_gemm()
        for splits in SPLITS[1:]:
            split_gemm(splits)
        torch.cuda.synchronize()

        gemv_bytes = TOKENS * (n_out * n_in // 2 + 4 * n_out * n_in // group + 4 * n_in) + 4 * TOKENS * n_out
        gemm_bytes = n_out * n_in * 2 + 4 * n_out * n_in // group + 2 * TOKENS * n_in + 2 * TOKENS * n_out
        int4_bytes = n_out * n_in // 2 + 4 * n_out * n_in // group + 4 * TOKENS * n_in + 4 * TOKENS * n_out
        gemv_seconds = time_call(gemv_loop, WARMUP, ITERATIONS)
        gemm_seconds = time_call(half_gemm, WARMUP, ITERATIONS)
        int4_seconds = time_call(int4_gemm, WARMUP, ITERATIONS)
        split_seconds = {1: int4_seconds}
        for splits in SPLITS[1:]:
            split_seconds[splits] = time_call(lambda s=splits: split_gemm(s), WARMUP, ITERATIONS)

        for name, n_bytes, seconds in (
            ("gemv loop x50", gemv_bytes, gemv_seconds),
            ("int4 gemm x1", int4_bytes, int4_seconds),
            ("bf16 gemm x1", gemm_bytes, gemm_seconds),
        ):
            gbs = bandwidth_gb_s(n_bytes, seconds)
            print(
                f"{f'{n_out} x {n_in}':>12}  {group:>5}  {name:>16}  {n_bytes / 1e6:>7.1f}MB  {seconds * 1e3:>8.3f}  "
                f"{gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  {n_bytes / PEAK_BANDWIDTH_GB_S / 1e6:>8.3f}"
            )
        print(
            f"{'':>12}  {'':>5}  int4 gemm / gemv {gemv_seconds / int4_seconds:.1f}x on time, "
            f"bytes {gemv_bytes / int4_bytes:.1f}x; bf16 gemm / gemv {gemv_seconds / gemm_seconds:.1f}x, "
            f"bytes {gemv_bytes / gemm_bytes:.1f}x"
        )
        chosen = tile_for(TOKENS, n_out)
        print(f"tile sweep, int4 gemm only, menu picks {chosen}")
        print(f"{'block_m':>7}  {'block_n':>7}  {'block_k':>7}  {'stages':>6}  {'MB shared':>9}  {'ms':>8}  {'menu':>5}")
        for candidate in CANDIDATES:
            staged = shared_memory_bytes(candidate)
            if staged > SHARED_LIMIT_BYTES:
                continue
            seconds = time_call(
                lambda c=candidate: int4_gemm_into(packed, scales, activations, out, group_size=group, tile=c),
                WARMUP,
                ITERATIONS,
            )
            print(
                f"{candidate[0]:>7}  {candidate[1]:>7}  {candidate[2]:>7}  {candidate[3]:>6}  "
                f"{staged / 1e6:>9.2f}  {seconds * 1e3:>8.3f}  {'yes' if candidate == chosen else '':>5}"
            )
        split_gemm(SPLITS[-1])
        first = out.clone()
        split_gemm(SPLITS[-1])
        print(f"split {SPLITS[-1]} repeats bit for bit: {torch.equal(first, out)}")
        print("stream k, paired against the single launch, partials summed inside the launch")
        print(f"{'splits':>6}  {'ms':>8}  {'speedup':>8}  {'range':>13}  {'MB partial':>10}")
        for splits in SPLITS:
            partial_mb = 4 * splits * TOKENS * n_out / 1e6 if splits > 1 else 0.0
            if splits == 1:
                print(f"{splits:>6}  {split_seconds[1] * 1e3:>8.3f}  {'1.00x':>8}  {'':>13}  {partial_mb:>10.2f}")
                continue
            pair = time_pair(int4_gemm, lambda s=splits: split_gemm(s), ITERATIONS, REPEATS, WARMUP)
            print(
                f"{splits:>6}  {pair.b * 1e3:>8.3f}  {pair.ratio:>7.2f}x  "
                f"{pair.ratio_low:.2f} to {pair.ratio_high:.2f}  {partial_mb:>10.2f}"
            )
        print()
        del w, packed, scales, activations, half, half_activations, out, half_out, partials

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

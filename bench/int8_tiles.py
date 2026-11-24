"""Sweep the int8 GEMM tile menu at the shapes the chunk actually runs.

Concept: the int4 GEMM picks its tile from a measured menu, but its int8 twin has been
running a menu nobody timed. A tile decides two things at chunk sizes: how many output
tiles divide the SMs, and how many bytes each pipeline stage holds against the shared
memory limit. This sweeps candidates at each width the expert uses and prints the winner.

Task: python -m vla_cu int8tiles
"""

from __future__ import annotations

import torch

from bench.real_layers import cosine
from kernels.triton.int8_gemm import SHARED_LIMIT_BYTES, int8_gemm_into, shared_memory_bytes, tile_for
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    warm_gpu,
)
from vla_cu.quant import quantize_int8

TOKENS = (2, 8, 50)
SHAPES = ((1920, 720), (720, 720), (720, 1920))
CANDIDATES = (
    (16, 64, 128, 3),
    (16, 128, 128, 2),
    (32, 64, 64, 3),
    (32, 64, 128, 3),
    (32, 128, 128, 2),
    (64, 64, 64, 2),
    (64, 64, 128, 2),
    (64, 64, 128, 3),
    (64, 128, 128, 2),
    (128, 64, 64, 2),
    (128, 64, 128, 2),
    (128, 128, 64, 2),
)
WARMUP = 10
ITERATIONS = 50


def main() -> int:
    """Time every candidate tile at each shape and token count, and mark the menu pick."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s")
    print(f"warmup {WARMUP}, iterations {ITERATIONS}, shared limit {SHARED_LIMIT_BYTES} bytes")
    print()
    generator = torch.Generator(device="cuda").manual_seed(0)
    best: dict[tuple[int, int, int], tuple[float, tuple[int, int, int, int]]] = {}

    for n_out, n_in in SHAPES:
        w = torch.randn((n_out, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
        codes, scales = quantize_int8(w)
        for tokens in TOKENS:
            x = torch.randn((tokens, n_in), device="cuda", generator=generator)
            out = torch.empty((tokens, n_out), device="cuda")
            int8_gemm_into(codes, scales, x, out)
            expected = (x @ codes.to(torch.float32).T) * scales
            error = cosine(out.reshape(-1), expected.reshape(-1))
            n_bytes = n_out * n_in + 4 * n_out + 4 * tokens * n_in + 4 * tokens * n_out
            chosen = tile_for(tokens)
            shape = (n_out, n_in, tokens)
            print(f"{n_out} x {n_in}, {tokens} tokens, cosine {error:.7f}, menu picks {chosen}")
            print(f"{'tile':>22}  {'MB shared':>9}  {'grid':>5}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}")
            for candidate in CANDIDATES:
                staged = shared_memory_bytes(candidate)
                if staged > SHARED_LIMIT_BYTES:
                    continue
                grid = -(-tokens // candidate[0]) * -(-n_out // candidate[1])
                seconds = time_call(
                    lambda c=candidate: int8_gemm_into(codes, scales, x, out, tile=c), WARMUP, ITERATIONS
                )
                gbs = bandwidth_gb_s(n_bytes, seconds)
                mark = "menu" if candidate == chosen else ""
                print(
                    f"{str(candidate):>22}  {staged / 1e6:>9.2f}  {grid:>5}  {seconds * 1e3:>8.3f}  "
                    f"{gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  {mark}"
                )
                if shape not in best or seconds < best[shape][0]:
                    best[shape] = (seconds, candidate)
            print()
            del x, out
        del w, codes, scales

    print("best measured tile per shape and token count")
    print(f"{'N x K':>12}  {'tokens':>6}  {'tile':>22}  {'ms':>8}")
    for (n_out, n_in, tokens), (seconds, tile) in best.items():
        print(f"{f'{n_out} x {n_in}':>12}  {tokens:>6}  {str(tile):>22}  {seconds * 1e3:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

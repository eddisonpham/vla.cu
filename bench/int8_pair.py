"""Pair the int8 menu's pick against the three tiles that came closest at one shape.

Concept: EXP-075 timed the tiles one after another on a GPU that other processes were
using, so a 4 percent gap cannot decide the menu. `time_pair` alternates the menu pick and
a challenger inside one process, and the spread across repeats is what makes a small
difference real or not.

Task: python -m vla_cu int8pair
"""

from __future__ import annotations

import torch

from bench.real_layers import cosine
from kernels.triton.int8_gemm import int8_gemm_into, tile_for
from vla_cu.bench_utils import enforce_memory_budget, time_pair, warm_gpu
from vla_cu.quant import quantize_int8

TOKENS = 50
N_OUT = 720
N_IN = 1920
CHALLENGERS = ((32, 128, 128, 2), (64, 64, 128, 2), (64, 64, 128, 3))
REPEATS = 5
ITERATIONS = 50
WARMUP = 10


def main() -> int:
    """Pair the menu pick against each challenger and say whether the gap survives."""
    enforce_memory_budget()
    warm_gpu()
    menu = tile_for(TOKENS)
    print(f"GPU {torch.cuda.get_device_name(0)} | {N_OUT} x {N_IN}, {TOKENS} tokens")
    print(f"menu pick {menu}, paired against {len(CHALLENGERS)} challengers")
    print(f"warmup {WARMUP}, repeats {REPEATS}, {ITERATIONS} iterations per repeat")
    print()
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((N_OUT, N_IN), device="cuda", generator=generator) * (1.0 / N_IN**0.5)
    codes, scales = quantize_int8(w)
    x = torch.randn((TOKENS, N_IN), device="cuda", generator=generator)
    out = torch.empty((TOKENS, N_OUT), device="cuda")
    expected = (x @ codes.to(torch.float32).T) * scales
    int8_gemm_into(codes, scales, x, out)
    print(f"every tile computes the same answer, cosine {cosine(out.reshape(-1), expected.reshape(-1)):.7f}")
    print()
    print(f"{'challenger':>22}  {'menu ms':>8}  {'tile ms':>8}  {'menu/tile':>10}  {'range':>13}  {'verdict':>16}")

    def chosen() -> None:
        int8_gemm_into(codes, scales, x, out, tile=menu)

    for tile in CHALLENGERS:
        def challenger(c=tile) -> None:
            int8_gemm_into(codes, scales, x, out, tile=c)

        timing = time_pair(chosen, challenger, ITERATIONS, REPEATS, WARMUP)
        if timing.ratio_low > 1.02:
            verdict = "to take"
        elif timing.ratio_high < 0.98:
            verdict = "to drop"
        else:
            verdict = "inside noise"
        print(
            f"{str(tile):>22}  {timing.a * 1e3:>8.3f}  {timing.b * 1e3:>8.3f}  {timing.ratio:>9.2f}x  "
            f"{timing.ratio_low:.2f} to {timing.ratio_high:.2f}  {verdict:>16}"
        )

    print()
    print("to take means the challenger won every repeat by more than two percent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The cooperative layer tail against the five launches it replaces, at two widths.

Concept: the fused path must reproduce the eager path before it is timed, and the two must
be timed in alternating pairs because this laptop part drifts between runs. The eager path
is exactly the five launches the layer used, so sweeping both widths shows where the one
launch stops paying off as the layer grows.

Task: python -m vla_cu coop
"""

from __future__ import annotations

import torch

from kernels.triton.rmsnorm import add_into, add_norm_into
from vla_cu.bench_utils import enforce_memory_budget, time_pair, warm_gpu
from vla_cu.blocks import MlpTokenBlock
from vla_cu.cuda_coop import coop_layer_into
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8

WIDTHS = ((720, 1920), (960, 2560))
TOKENS = (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 32, 50, 64)
REPEATS = 5
ITERATIONS = 20
WARMUP = 5
TOLERANCE = 2e-2


def sweep(hidden: int, intermediate: int) -> int:
    """Time the eager and fused tails at one width and return the first losing token count."""
    print(f"hidden {hidden}, intermediate {intermediate}")
    print(f"{'tokens':>6}  {'eager ms':>9}  {'fused ms':>9}  {'speedup':>8}  {'range':>13}  {'max diff':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(hidden, device="cuda", generator=generator) + 1.0
    up = torch.randn((intermediate, hidden), device="cuda", generator=generator) * (1.0 / hidden**0.5)
    down = torch.randn((hidden, intermediate), device="cuda", generator=generator) * (1.0 / intermediate**0.5)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP_SIZE)
    blocks = torch.cuda.get_device_properties(0).multi_processor_count
    crossover = 0

    for tokens in TOKENS:
        x = torch.randn((tokens, hidden), device="cuda", generator=generator)
        other = torch.randn((tokens, hidden), device="cuda", generator=generator)
        added = torch.empty_like(x)
        normed = torch.empty_like(x)
        eager_out = torch.empty_like(x)
        tail = torch.empty_like(x)
        fused_added = torch.empty_like(x)
        fused_normed = torch.empty_like(x)
        fused_out = torch.empty_like(x)
        act = torch.empty((tokens, intermediate), device="cuda")
        block = MlpTokenBlock(up, down, down_bits=4, group_size=GROUP_SIZE, tokens=tokens)

        def eager() -> None:
            add_norm_into(x, other, weight, added, normed)
            block.into(normed, tail)
            add_into(added, tail, eager_out)

        def fused() -> None:
            coop_layer_into(
                x, other, weight, up_codes, up_scales, down_packed, down_scales,
                fused_added, fused_normed, act, fused_out, blocks,
            )

        eager()
        fused()
        torch.cuda.synchronize()
        difference = (eager_out - fused_out).abs().max().item()
        agrees = (
            torch.allclose(eager_out, fused_out, atol=TOLERANCE, rtol=TOLERANCE)
            and torch.allclose(normed, fused_normed, atol=1e-4, rtol=1e-4)
            and torch.allclose(added, fused_added, atol=1e-5, rtol=1e-5)
        )
        if not agrees:
            print(f"{tokens:>6}  fused path disagrees with the eager path, max diff {difference:.2e}")
            return 1

        timing = time_pair(eager, fused, ITERATIONS, REPEATS, WARMUP)
        print(
            f"{tokens:>6}  {timing.a * 1e3:>9.3f}  {timing.b * 1e3:>9.3f}  {timing.ratio:>7.2f}x  "
            f"{timing.ratio_low:.2f} to {timing.ratio_high:.2f}  {difference:>9.1e}"
        )
        if crossover == 0 and timing.ratio < 1.0:
            crossover = tokens
        del x, other, added, normed, eager_out, tail, fused_added, fused_normed, fused_out, act, block

    print(f"crossover: fused loses first at {crossover or 'no losing token count'}")
    print()
    return crossover


def main() -> int:
    """Check the fused tail against the five launch path, then time both widths paired."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)}")
    print("eager: add_norm, up GEMM, SiLU, down GEMM, residual add = 5 launches")
    print("fused: one cooperative launch with two grid barriers")
    print(f"warmup {WARMUP}, repeats {REPEATS}, {ITERATIONS} iterations per repeat")
    print()
    crossovers = []
    for hidden, intermediate in WIDTHS:
        crossovers.append((hidden, intermediate, sweep(hidden, intermediate)))
        if crossovers[-1][2] == 1:
            return 1
    print(f"{'hidden':>6}  {'intermediate':>12}  {'crossover':>9}")
    for hidden, intermediate, crossover in crossovers:
        print(f"{hidden:>6}  {intermediate:>12}  {crossover:>9}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

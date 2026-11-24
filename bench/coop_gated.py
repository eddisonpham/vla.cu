"""The gated SwiGLU tail in one cooperative launch against the six it replaces.

Concept: a SmolVLA MLP carries gate, up and down, so the eager chain is add and norm, two
int8 projections, the SiLU times up multiply, the int4 down projection and the residual
add. The fused kernel stages the norm once and computes both projections in one pass over
it, which is the only structural difference from the two-matrix tail. Eager and fused run
in alternating pairs, because this laptop part drifts between runs.

Task: python -m vla_cu coopgated
"""

from __future__ import annotations

import torch

from kernels.triton.rmsnorm import add_into, add_norm_into
from vla_cu.bench_utils import enforce_memory_budget, time_pair, warm_gpu
from vla_cu.blocks import GatedMlpBlock
from vla_cu.cuda_coop import coop_gated_into
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8

WIDTHS = ((720, 1920), (960, 2560))
TOKENS = (1, 2, 4, 8, 10, 12, 16, 32, 50, 64)
REPEATS = 5
ITERATIONS = 20
WARMUP = 5
TOLERANCE = 2e-2


def sweep(hidden: int, intermediate: int) -> int:
    """Time the eager and fused gated tails at one width and return the first losing token count."""
    print(f"hidden {hidden}, intermediate {intermediate}")
    print(f"{'tokens':>6}  {'eager ms':>9}  {'fused ms':>9}  {'speedup':>8}  {'range':>13}  {'max diff':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(hidden, device="cuda", generator=generator) + 1.0
    gate = torch.randn((intermediate, hidden), device="cuda", generator=generator) * (1.0 / hidden**0.5)
    up = torch.randn((intermediate, hidden), device="cuda", generator=generator) * (1.0 / hidden**0.5)
    down = torch.randn((hidden, intermediate), device="cuda", generator=generator) * (1.0 / intermediate**0.5)
    gate_codes, gate_scales = quantize_int8(gate)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP_SIZE)
    blocks = torch.cuda.get_device_properties(0).multi_processor_count
    crossover = 0

    for tokens in TOKENS:
        x = torch.randn((tokens, hidden), device="cuda", generator=generator)
        other = torch.randn((tokens, hidden), device="cuda", generator=generator)
        added = torch.empty_like(x)
        normed = torch.empty_like(x)
        tail = torch.empty_like(x)
        eager_out = torch.empty_like(x)
        fused_added = torch.empty_like(x)
        fused_normed = torch.empty_like(x)
        fused_out = torch.empty_like(x)
        act = torch.empty((tokens, intermediate), device="cuda")
        block = GatedMlpBlock(gate, up, down, down_bits=4, group_size=GROUP_SIZE, tokens=tokens)

        def eager() -> None:
            add_norm_into(x, other, weight, added, normed)
            if tokens == 1:
                block.into(normed[0], tail[0])
            else:
                block.into(normed, tail)
            add_into(added, tail, eager_out)

        def fused() -> None:
            coop_gated_into(
                x, other, weight, gate_codes, gate_scales, up_codes, up_scales,
                down_packed, down_scales, fused_added, fused_normed, act, fused_out, blocks,
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
        del x, other, added, normed, tail, eager_out, fused_added, fused_normed, fused_out, act, block

    print(f"crossover: fused loses first at {crossover or 'no losing token count'}")
    print()
    return crossover


def main() -> int:
    """Check the fused gated tail against the six launch chain, then time both widths paired."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)}")
    print("eager: add_norm, gate GEMM, up GEMM, SiLU times up, down GEMM, residual add = 6 launches")
    print("fused: one cooperative launch with two grid barriers, both projections in one pass")
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

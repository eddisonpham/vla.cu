"""The cooperative layer tail against the five launches it replaces, in one process.

Concept: the fused path must reproduce the eager path before it is timed, and the two
must be timed in alternating pairs because this laptop part drifts between runs. The
eager path is exactly the five launches the layer used: add-norm, int8 up GEMM, SiLU,
int4 down GEMM and the residual add.

Task: python -m vla_cu coop
"""

from __future__ import annotations

import torch

from kernels.triton.rmsnorm import add_into, add_norm_into
from vla_cu.bench_utils import enforce_memory_budget, time_pair, warm_gpu
from vla_cu.blocks import MlpTokenBlock
from vla_cu.cuda_coop import HIDDEN, INTERMEDIATE, coop_layer_into
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8

TOKENS = (1, 2, 3, 8, 50)
REPEATS = 5
ITERATIONS = 20
WARMUP = 5
TOLERANCE = 2e-2


def main() -> int:
    """Check the fused tail against the five launch path, then time them paired."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | hidden {HIDDEN}, intermediate {INTERMEDIATE}")
    print("eager: add_norm, up GEMM, SiLU, down GEMM, residual add = 5 launches")
    print("fused: one cooperative launch with two grid barriers")
    print(f"warmup {WARMUP}, repeats {REPEATS}, {ITERATIONS} iterations per repeat")
    print()
    print(f"{'tokens':>6}  {'eager ms':>9}  {'fused ms':>9}  {'speedup':>8}  {'range':>13}  {'max diff':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(HIDDEN, device="cuda", generator=generator) + 1.0
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * (1.0 / HIDDEN**0.5)
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * (1.0 / INTERMEDIATE**0.5)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP_SIZE)
    blocks = torch.cuda.get_device_properties(0).multi_processor_count

    for tokens in TOKENS:
        x = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
        other = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
        added = torch.empty_like(x)
        normed = torch.empty_like(x)
        eager_out = torch.empty_like(x)
        tail = torch.empty_like(x)
        fused_added = torch.empty_like(x)
        fused_normed = torch.empty_like(x)
        fused_out = torch.empty_like(x)
        act = torch.empty((tokens, INTERMEDIATE), device="cuda")
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
        del x, other, added, normed, eager_out, tail, fused_added, fused_normed, fused_out, act, block

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""BLOCK_K and split-K sweeps for the fused int8 GEMV.

Two questions, one table each. The first: which block size streams a DRAM-bound
weight matrix fastest. The second: which shapes have too few output rows to fill 26
SMs, and how much splitting the reduction recovers.

Every split result is checked against split 1 before it is timed, and the split time
includes the output zero fill that atomics require, since a caller would pay it too.

Task: python -m vla_cu tune
"""

from __future__ import annotations

import torch

from kernels.triton.int8_gemv import int8_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
)

SWEEP_SHAPE = (8192, 16384)
BLOCK_SIZES = (128, 256, 512, 1024, 2048, 4096)
SPLIT_SHAPES = ((16, 8_388_608), (64, 2_097_152), (256, 524_288), (1024, 131_072), (4096, 32_768))
SPLITS = (1, 2, 4, 8)
BLOCK_FOR_SPLIT = 1024
WARMUP = 5
ITERATIONS = 20


def fused_bytes(n_out: int, n_in: int) -> int:
    """Bytes one fused call must move: weights, scales, activations, output."""
    return n_out * n_in + 4 * n_out + 4 * n_in + 4 * n_out


def make_inputs(n_out: int, n_in: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weights, scales, activations and an output buffer for one shape."""
    w = torch.randint(-127, 128, (n_out, n_in), device="cuda", dtype=torch.int8)
    scales = torch.rand(n_out, device="cuda") * 0.02 + 0.001
    x = torch.randn(n_in, device="cuda")
    out = torch.empty(n_out, device="cuda")
    return w, scales, x, out


def time_config(
    w: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    block_k: int,
    split: int,
) -> float:
    """Return mean seconds per call for one block size and split, warmed up."""

    def call() -> None:
        int8_gemv_into(w, scales, x, out, block_k=block_k, split=split)

    return time_call(call, WARMUP, ITERATIONS)


def main() -> int:
    """Print the block size sweep, then the split-K table with its error column."""
    enforce_memory_budget()
    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | int8 weights, batch 1")
    print()

    n_out, n_in = SWEEP_SHAPE
    w, scales, x, out = make_inputs(n_out, n_in)
    n_bytes = fused_bytes(n_out, n_in)
    print(f"block size sweep at {n_out} x {n_in}, {n_bytes / 1e6:.1f} MB moved, split 1")
    print(f"{'BLOCK_K':>8}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}")
    for block_k in BLOCK_SIZES:
        seconds = time_config(w, scales, x, out, block_k, 1)
        gbs = bandwidth_gb_s(n_bytes, seconds)
        print(f"{block_k:>8}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}")
    del w, scales, x, out

    print()
    print(f"split-K table, BLOCK_K {BLOCK_FOR_SPLIT}, each shape moves the same 134 MB of weights")
    print(f"{'N x K':>15}  {'split':>5}  {'programs':>8}  {'ms':>8}  {'GB/s':>8}  {'speedup':>7}  {'max rel err':>11}")
    for split_out, split_in in SPLIT_SHAPES:
        w, scales, x, out = make_inputs(split_out, split_in)
        shape_bytes = fused_bytes(split_out, split_in)
        int8_gemv_into(w, scales, x, out, block_k=BLOCK_FOR_SPLIT, split=1)
        torch.cuda.synchronize()
        baseline = out.clone()
        base_seconds = 0.0
        for split in SPLITS:
            species = time_config(w, scales, x, out, BLOCK_FOR_SPLIT, split)
            int8_gemv_into(w, scales, x, out, block_k=BLOCK_FOR_SPLIT, split=split)
            torch.cuda.synchronize()
            error = (torch.abs(out - baseline) / torch.clamp(baseline.abs(), min=1e-6)).max().item()
            if split == 1:
                base_seconds = species
            gbs = bandwidth_gb_s(shape_bytes, species)
            label = f"{split_out} x {split_in}"
            print(
                f"{label:>15}  {split:>5}  {split_out * split:>8}  {species * 1e3:>8.3f}  "
                f"{gbs:>8.1f}  {base_seconds / species:>6.2f}x  {error:>11.2e}"
            )
        del w, scales, x, out, baseline
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

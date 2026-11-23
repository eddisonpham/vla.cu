"""Fused int8 GEMV against dequantize-then-matmul on the same weights.

The naive path materializes fp32 weights, so it writes and re-reads 4 bytes per
weight that the fused path never touches. Both paths write into preallocated
buffers, so the comparison is traffic, not allocation.

Task: python -m vla_cu gemv
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

SIZES = ((1024, 1024), (4096, 2048), (4096, 8192), (8192, 16384))
WARMUP = 10
ITERATIONS = 50


def measure(n_out: int, n_in: int) -> tuple[float, float, int, int]:
    """Return (naive seconds, fused seconds, naive bytes, fused bytes) for one size."""
    weights = torch.randint(-127, 128, (n_out, n_in), device="cuda", dtype=torch.int8)
    scales = torch.rand(n_out, device="cuda") * 0.02 + 0.001
    x = torch.randn(n_in, device="cuda")
    out = torch.empty(n_out, device="cuda")
    weights_fp32 = torch.empty((n_out, n_in), device="cuda")

    def naive() -> None:
        weights_fp32.copy_(weights)
        weights_fp32.mul_(scales[:, None])
        torch.mv(weights_fp32, x, out=out)

    def fused() -> None:
        int8_gemv_into(weights, scales, x, out)

    weight_bytes = n_out * n_in
    naive_bytes = weight_bytes + 8 * weight_bytes + 4 * n_in + 4 * n_out
    fused_bytes = weight_bytes + 4 * n_out + 4 * n_in + 4 * n_out

    naive_s = time_call(naive, WARMUP, ITERATIONS)
    fused_s = time_call(fused, WARMUP, ITERATIONS)
    return naive_s, fused_s, naive_bytes, fused_bytes


def main() -> int:
    """Print time, speedup, bandwidth, and whether the weights fit in L2."""
    enforce_memory_budget()
    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | int8 weights, fp32 activations, batch 1")
    print()
    print(
        f"{'N x K':>13}  {'W MB':>6}  {'naive ms':>8}  {'fused ms':>8}  "
        f"{'speedup':>7}  {'fused GB/s':>10}  {'% DRAM':>7}  {'in L2':>6}"
    )

    for n_out, n_in in SIZES:
        naive_s, fused_s, naive_bytes, fused_bytes = measure(n_out, n_in)
        weights_mb = n_out * n_in / 1e6
        fused_gbs = bandwidth_gb_s(fused_bytes, fused_s)
        in_l2 = "yes" if n_out * n_in <= l2_bytes else "no"
        print(
            f"{f'{n_out} x {n_in}':>13}  {weights_mb:>6.1f}  {naive_s * 1e3:>8.3f}  {fused_s * 1e3:>8.3f}  "
            f"{naive_s / fused_s:>6.2f}x  {fused_gbs:>10.1f}  {fraction_of_peak(fused_gbs):>6.1%}  {in_l2:>6}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

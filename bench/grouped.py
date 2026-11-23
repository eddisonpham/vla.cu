"""Three projections of one layer as three launches against one grouped launch.

The SmolVLA layer shape is used twice: a small case where the whole layer fits in L2,
so the only cost left is dispatch, and a large case that streams from DRAM, where the
question becomes whether the concatenated layout still coalesces.

Both paths are checked against each other before either is timed, and the byte count is
identical, so the table is a launch comparison and not a traffic comparison. The
grouped path additionally reads the activation vector once per matrix group instead of
once per matrix.

Task: python -m vla_cu grouped
"""

from __future__ import annotations

import torch

from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    warm_gpu,
)
from vla_cu.quant import GROUP_SIZE, pack_int4

MATRICES = 3
SHAPES = ((960, 960), (8192, 4096))
WARMUP = 10
ITERATIONS = 50


def main() -> int:
    """Compare three separate launches against one grouped launch at two sizes."""
    enforce_memory_budget()
    warm_gpu()
    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"{MATRICES} matrices per layer, one activation vector shared by all of them")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | group {GROUP_SIZE}")
    print()

    for n_out, n_in in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn((MATRICES * n_out, n_in), device="cuda", generator=generator) * 0.05
        x = torch.randn(n_in, device="cuda", generator=generator)
        packed, scales = pack_int4(w)

        out_separate = torch.empty(MATRICES * n_out, device="cuda")
        out_grouped = torch.empty(MATRICES * n_out, device="cuda")
        segment = torch.arange(MATRICES, device="cuda", dtype=torch.int32).repeat_interleave(n_out)
        x_base = torch.zeros(MATRICES, device="cuda", dtype=torch.int32)

        def separate() -> None:
            for matrix in range(MATRICES):
                start = matrix * n_out
                int4_gemv_into(
                    packed[start : start + n_out],
                    scales[start : start + n_out],
                    x,
                    out_separate[start : start + n_out],
                )

        def grouped() -> None:
            int4_grouped_gemv_into(packed, scales, x, out_grouped, segment, x_base)

        separate()
        grouped()
        torch.cuda.synchronize()
        agree = bool(torch.allclose(out_separate, out_grouped, rtol=1e-4, atol=1e-3))
        max_error = (out_separate - out_grouped).abs().max().item()

        n_bytes = MATRICES * (n_out * n_in // 2 + 4 * n_out * n_in // GROUP_SIZE) + 4 * n_in + 4 * MATRICES * n_out
        separate_seconds = time_call(separate, WARMUP, ITERATIONS)
        grouped_seconds = time_call(grouped, WARMUP, ITERATIONS)

        print(f"{MATRICES} x {n_out} x {n_in}, {n_bytes / 1e6:.2f} MB moved")
        print(f"{'path':>9}  {'launches':>8}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}")
        for name, launches, seconds in (
            ("separate", MATRICES, separate_seconds),
            ("grouped", 1, grouped_seconds),
        ):
            gbs = bandwidth_gb_s(n_bytes, seconds)
            print(
                f"{name:>9}  {launches:>8}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}"
            )
        print(f"grouped / separate {separate_seconds / grouped_seconds:.2f}x | agree {agree} | max abs diff {max_error:.2e}")
        print()
        del w, packed, scales, x, out_separate, out_grouped

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

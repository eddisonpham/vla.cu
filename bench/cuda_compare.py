"""Triton against CUDA C++ on the same int4 GEMV, same shapes, same inputs.

Both kernels implement the same format and the same memory traffic, so the only
differences are how the work is mapped onto threads and how the compiler schedules
it. Agreement is checked before either is timed: an error table next to a speed
table is the only honest way to present a port.

Task: python -m vla_cu cuda
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
)
from vla_cu.cuda_int4 import int4_gemv_cuda, int4_gemv_cuda_into
from vla_cu.quant import GROUP_SIZE, pack_int4

SHAPES = ((1024, 1024), (4096, 2048), (8192, 16384))
WARMUP = 5
ITERATIONS = 20


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the L2 norm of the error over the L2 norm of the reference."""
    return (torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)).item()


def moved_bytes(n_out: int, n_in: int) -> int:
    """Bytes one call moves: packed weights, group scales, activations, output."""
    return n_out * n_in // 2 + 4 * n_out * n_in // GROUP_SIZE + 4 * n_in + 4 * n_out


def main() -> int:
    """Run each shape through both kernels, verify agreement, then time them."""
    enforce_memory_budget()
    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | group {GROUP_SIZE} | batch 1")
    print()
    print(f"{'N x K':>14}  {'MB':>7}  {'impl':>7}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}  {'rel L2 err':>11}  {'agree':>7}")

    for n_out, n_in in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn((n_out, n_in), device="cuda", generator=generator) * 0.05
        x = torch.randn(n_in, device="cuda", generator=generator)
        packed, scales = pack_int4(w)
        n_bytes = moved_bytes(n_out, n_in)

        out_cuda = int4_gemv_cuda(packed, scales, x)
        out_triton = torch.empty(n_out, device="cuda")
        int4_gemv_into(packed, scales, x, out_triton)
        torch.cuda.synchronize()

        exact = torch.mv(w, x)
        agree = bool(torch.allclose(out_cuda, out_triton, rtol=1e-4, atol=1e-3))
        times = {
            "triton": time_call(lambda: int4_gemv_into(packed, scales, x, out_triton), WARMUP, ITERATIONS),
            "cuda": time_call(lambda: int4_gemv_cuda_into(packed, scales, x, out_cuda), WARMUP, ITERATIONS),
        }

        for name, result in (("triton", out_triton), ("cuda", out_cuda)):
            seconds = times[name]
            gbs = bandwidth_gb_s(n_bytes, seconds)
            print(
                f"{f'{n_out} x {n_in}':>14}  {n_bytes / 1e6:>7.1f}  {name:>7}  {seconds * 1e3:>8.3f}  "
                f"{gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  {relative_error(result, exact):>11.2e}  "
                f"{'yes' if agree else 'NO':>7}"
            )
        print(f"{'':>14}  cuda / triton {times['triton'] / times['cuda']:.2f}x on time")
        print()

        del w, packed, scales, out_cuda, out_triton, exact

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Triton against three CUDA C++ variants of the same int4 GEMV, same inputs.

Every implementation here reads the same bytes and does the same arithmetic, so the
only differences are how the work is mapped onto threads and which cache hints the
loads carry. Agreement is checked before anything is timed.

The three CUDA variants differ by one change each: "baseline" is the guarded loop from
EXP-016, "unroll" splits the loop so the compiler can unroll four steps, and "stream"
adds streaming weight loads on top.

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
    warm_gpu,
)
from vla_cu.cuda_int4 import int4_gemv_cuda_into
from vla_cu.quant import GROUP_SIZE, pack_int4

SHAPES = ((1024, 1024), (4096, 2048), (8192, 16384))
CUDA_VARIANTS = ("baseline", "unroll", "stream")
WARMUP = 5
ITERATIONS = 20


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the L2 norm of the error over the L2 norm of the reference."""
    return (torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)).item()


def moved_bytes(n_out: int, n_in: int) -> int:
    """Bytes one call moves: packed weights, group scales, activations, output."""
    return n_out * n_in // 2 + 4 * n_out * n_in // GROUP_SIZE + 4 * n_in + 4 * n_out


def main() -> int:
    """Run each shape through every implementation, verify agreement, then time them."""
    enforce_memory_budget()
    warm_gpu()
    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | group {GROUP_SIZE} | batch 1")
    print()
    print(f"{'N x K':>14}  {'impl':>9}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}  {'vs triton':>10}  {'rel L2 err':>11}  {'agree':>7}")

    for n_out, n_in in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn((n_out, n_in), device="cuda", generator=generator) * 0.05
        x = torch.randn(n_in, device="cuda", generator=generator)
        packed, scales = pack_int4(w)
        n_bytes = moved_bytes(n_out, n_in)

        reference_out = torch.empty(n_out, device="cuda")
        int4_gemv_into(packed, scales, x, reference_out)
        cuda_out = torch.empty(n_out, device="cuda")
        exact = torch.mv(w, x)
        torch.cuda.synchronize()

        times = {"triton": time_call(lambda: int4_gemv_into(packed, scales, x, reference_out), WARMUP, ITERATIONS)}
        results = {"triton": reference_out}
        for variant in CUDA_VARIANTS:
            def call(variant=variant) -> None:
                int4_gemv_cuda_into(packed, scales, x, cuda_out, variant=variant)

            times[f"cuda:{variant}"] = time_call(call, WARMUP, ITERATIONS)
            results[f"cuda:{variant}"] = cuda_out.clone()

        triton_seconds = times["triton"]
        for name in ("triton",) + tuple(f"cuda:{variant}" for variant in CUDA_VARIANTS):
            result = results[name]
            seconds = times[name]
            gbs = bandwidth_gb_s(n_bytes, seconds)
            agree = bool(torch.allclose(result, results["triton"], rtol=1e-4, atol=1e-3))
            print(
                f"{f'{n_out} x {n_in}':>14}  {name:>9}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  "
                f"{fraction_of_peak(gbs):>6.1%}  {triton_seconds / seconds:>9.2f}x  "
                f"{relative_error(result, exact):>11.2e}  {'yes' if agree else 'NO':>7}"
            )
        print()
        del w, packed, scales, x, exact, reference_out, cuda_out, results

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

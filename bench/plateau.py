"""Triton against the three CUDA variants of the int4 GEMV on the scaling shape grid.

Concept: one kernel at many sizes separates the implementation from the part. Each
implementation is timed in every repeat, and the reported number is the median of the
repeats, because this laptop part moves its clocks and a single 50 ms window is not a
stable measurement.

Task: python -m vla_cu plateau
"""

from __future__ import annotations

import statistics

import torch

from bench.scale import SHAPES, VERIFY_MAX_ELEMENTS
from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    warm_gpu,
)
from vla_cu.cuda_int4 import VARIANTS, int4_gemv_cuda_into
from vla_cu.quant import GROUP_SIZE, dequantize_int4

WARMUP = 20
ITERATIONS = 30
REPEATS = 5


def moved_bytes(n_out: int, n_in: int) -> int:
    """Return the bytes one call reads and writes, group scales included."""
    return n_out * n_in // 2 + 4 * n_out * (n_in // GROUP_SIZE) + 4 * n_in + 4 * n_out


def main() -> int:
    """Time every implementation at every shape and report median peak percentage."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | batch 1")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | repeats {REPEATS} | group {GROUP_SIZE}")
    print()
    print(
        f"{'N x K':>15}  {'impl':>13}  {'MB':>7}  {'ms':>8}  {'range ms':>9}  {'GB/s':>8}  "
        f"{'% DRAM':>7}  {'vs triton':>10}  {'max abs diff':>12}"
    )
    best: dict[str, tuple[float, str]] = {}
    labels = ("triton",) + tuple(f"cuda:{variant}" for variant in VARIANTS)

    for n_out, n_in in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(n_in, device="cuda", generator=generator)
        packed = torch.randint(0, 256, (n_out, n_in // 2), dtype=torch.uint8, device="cuda", generator=generator)
        scales = torch.rand((n_out, n_in // GROUP_SIZE), device="cuda", generator=generator) * 0.01 + 0.001
        reference = torch.empty(n_out, device="cuda")
        out = torch.empty(n_out, device="cuda")
        int4_gemv_into(packed, scales, x, reference)
        exact = torch.mv(dequantize_int4(packed, scales), x) if n_out * n_in <= VERIFY_MAX_ELEMENTS else None
        torch.cuda.synchronize()

        samples: dict[str, list[float]] = {name: [] for name in labels}
        diffs: dict[str, float] = {name: 0.0 for name in labels}
        for repeat in range(REPEATS):
            warmup = WARMUP if repeat == 0 else 0
            for name in labels:
                if name == "triton":
                    samples[name].append(
                        time_call(lambda: int4_gemv_into(packed, scales, x, reference), warmup, ITERATIONS)
                    )
                else:
                    variant = name.split(":", 1)[1]
                    samples[name].append(
                        time_call(
                            lambda v=variant: int4_gemv_cuda_into(packed, scales, x, out, variant=v), warmup, ITERATIONS
                        )
                    )
                    diffs[name] = max(diffs[name], torch.max(torch.abs(out - reference)).item())
                    out.zero_()

        n_bytes = moved_bytes(n_out, n_in)
        label = f"{n_out} x {n_in}"
        seconds_by_name = {name: statistics.median(samples[name]) for name in labels}
        triton_seconds = seconds_by_name["triton"]
        for name in labels:
            seconds = seconds_by_name[name]
            spread = (max(samples[name]) - min(samples[name])) * 1e3
            gbs = bandwidth_gb_s(n_bytes, seconds)
            share = fraction_of_peak(gbs)
            ratio = "" if name == "triton" else f"{triton_seconds / seconds:.2f}x"
            print(
                f"{label:>15}  {name:>13}  {n_bytes / 1e6:>7.1f}  {seconds * 1e3:>8.3f}  {spread:>9.3f}  "
                f"{gbs:>8.1f}  {share:>6.1%}  {ratio:>10}  {diffs[name]:>12.2e}"
            )
            if name not in best or share > best[name][0]:
                best[name] = (share, label)
        if exact is not None:
            error = (torch.linalg.vector_norm(reference - exact) / torch.linalg.vector_norm(exact)).item()
            print(f"{'':>15}  triton relative L2 against the dequantized reference {error:.2e}")
        print()
        del x, packed, scales, reference, out, exact

    print("plateau, best shape per implementation")
    for name in labels:
        share, label = best[name]
        print(f"  {name:>13}  {share:>6.1%} at {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

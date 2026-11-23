"""Shared measurement utilities: the 7 GB cap, CUDA event timing, bandwidth math.

Concept: CUDA work is asynchronous, so wall clock time measures launches and queue
depth, not execution. CUDA events are recorded on the GPU timeline after warmup,
so they time the actual work.
"""

from __future__ import annotations

import statistics
from typing import Callable, NamedTuple

import torch

MEMORY_BUDGET_GB = 7.0
PEAK_BANDWIDTH_GB_S = 384.0


def enforce_memory_budget(gb: float = MEMORY_BUDGET_GB, device: int = 0) -> None:
    """Cap this process at `gb` gigabytes of CUDA memory, so overshoot fails loudly.

    Implemented as a fraction of total device memory because that is what torch
    exposes. A cap above the physical memory is clamped to 1.0.
    """
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    fraction = min(gb * 1024**3 / total_bytes, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)


def warm_gpu(iterations: int = 200) -> None:
    """Keep the GPU at full memory bandwidth briefly before any path is timed.

    A laptop part ramps its clocks and settles its power state in tens of milliseconds,
    which is longer than a single kernel. Without this, whichever path a benchmark
    times first pays the ramp and the comparison reflects the order of the table, not
    the code (measured: the first path read 0.150 ms and the second 0.071 ms for the
    same work, and the two swapped places between runs).
    """
    buffer = torch.empty(16 * 1024 * 1024, device="cuda")
    for _ in range(iterations):
        buffer.sum()
    torch.cuda.synchronize()
    del buffer


def time_call(fn: Callable[[], None], warmup: int = 25, iterations: int = 100) -> float:
    """Return mean seconds per call of `fn`, timed with CUDA events after warmup.

    Warmup runs populate the Triton compile cache and raise clocks; without it the
    first call's JIT cost is measured instead of the kernel.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        fn()
    stop.record()
    stop.synchronize()

    return start.elapsed_time(stop) / 1000.0 / iterations


class PairTiming(NamedTuple):
    """Median seconds of each call, plus the ratio of the two across paired repeats."""

    a: float
    b: float
    ratio: float
    ratio_low: float
    ratio_high: float


def time_pair(
    fn_a: Callable[[], None],
    fn_b: Callable[[], None],
    iterations: int = 30,
    repeats: int = 5,
    warmup: int = 5,
) -> PairTiming:
    """Time two calls in alternating pairs and report the median ratio with its spread.

    One clock state is shared by both halves of a pair, so a drift that moves both
    cancels in the ratio; the spread across repeats is the honest error bar. Pairing is
    what replaces pinned clocks, which this GPU denies to a non-administrator.
    """
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()

    a_times: list[float] = []
    b_times: list[float] = []
    ratios: list[float] = []
    for _ in range(repeats):
        a_seconds = time_call(fn_a, 0, iterations)
        b_seconds = time_call(fn_b, 0, iterations)
        a_times.append(a_seconds)
        b_times.append(b_seconds)
        ratios.append(a_seconds / b_seconds)

    return PairTiming(
        a=statistics.median(a_times),
        b=statistics.median(b_times),
        ratio=statistics.median(ratios),
        ratio_low=min(ratios),
        ratio_high=max(ratios),
    )


def bandwidth_gb_s(n_bytes: int, seconds: float) -> float:
    """Return achieved bandwidth in GB/s for a call that moved `n_bytes`."""
    return n_bytes / seconds / 1e9


def fraction_of_peak(gb_s: float, peak: float = PEAK_BANDWIDTH_GB_S) -> float:
    """Return achieved bandwidth as a fraction of the spec peak."""
    return gb_s / peak

"""Shared measurement utilities: the 7 GB cap, CUDA event timing, bandwidth math.

Concept: CUDA work is asynchronous, so wall clock time measures launches and queue
depth, not execution. CUDA events are recorded on the GPU timeline after warmup,
so they time the actual work.
"""

from __future__ import annotations

from typing import Callable

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


def bandwidth_gb_s(n_bytes: int, seconds: float) -> float:
    """Return achieved bandwidth in GB/s for a call that moved `n_bytes`."""
    return n_bytes / seconds / 1e9


def fraction_of_peak(gb_s: float, peak: float = PEAK_BANDWIDTH_GB_S) -> float:
    """Return achieved bandwidth as a fraction of the spec peak."""
    return gb_s / peak

"""Attainable read bandwidth on this GPU, measured with no unpacking work at all.

Concept: every "% of peak" in this repo divides by a spec sheet. This task builds the
denominator that matters, a read-only kernel streaming contiguous bytes, so a real
kernel's streaming rate can be compared against what the part actually delivers. Each
configuration is repeated, and the repeats are interleaved across block sizes, so a
part that throttles while the task runs slows every configuration instead of only the
last one. The best sample is the estimate of attainable bandwidth; the spread is the
estimate of how much of a table is noise.

Task: python -m vla_cu stream
"""

from __future__ import annotations

import statistics

import torch
import triton
import triton.language as tl

from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
)

BLOCK_SIZES = (1024, 4096, 16384)
SIZES_IN_FLOATS = (16_777_216, 33_554_432)
REPEATS = 5
WARMUP = 5
ITERATIONS = 20


@triton.jit
def read_kernel(src_ptr, out_ptr, n, BLOCK: tl.constexpr) -> None:
    """Sum one BLOCK of floats per program so the read cannot be optimised away.

    Mapping: program p reads src[p*BLOCK : (p+1)*BLOCK] and writes one partial sum.
    Loads are fully coalesced with no reuse, so this is pure DRAM streaming and the
    expected bottleneck is memory, at whatever rate this power-limited part sustains.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src_ptr + offsets, mask=offsets < n, other=0.0)
    tl.store(out_ptr + tl.program_id(0), tl.sum(values, axis=0))


def read_once(src: torch.Tensor, out: torch.Tensor, block: int) -> float:
    """Return mean seconds for one read pass over `src` at this block size."""
    n = src.numel()
    grid = (triton.cdiv(n, block),)

    def call() -> None:
        read_kernel[grid](src, out, n, BLOCK=block)

    return time_call(call, WARMUP, ITERATIONS)


def main() -> int:
    """Print one row per size and block size, with the spread across repeats."""
    enforce_memory_budget()
    largest = max(SIZES_IN_FLOATS)
    out = torch.empty(triton.cdiv(largest, min(BLOCK_SIZES)) + 1, device="cuda")
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | read-only")
    print(f"repeats {REPEATS} | warmup {WARMUP} | iterations {ITERATIONS}")
    print()
    print(f"{'size MB':>9}  {'BLOCK':>6}  {'min GB/s':>9}  {'median':>8}  {'max':>8}  {'% of 384':>9}  {'spread':>7}")

    for n in SIZES_IN_FLOATS:
        src = torch.randn(n, device="cuda")
        n_bytes = n * 4
        samples: dict[int, list[float]] = {block: [] for block in BLOCK_SIZES}
        for _ in range(REPEATS):
            for block in BLOCK_SIZES:
                samples[block].append(bandwidth_gb_s(n_bytes, read_once(src, out, block)))
        for block in BLOCK_SIZES:
            best = max(samples[block])
            print(
                f"{n_bytes / 1e6:>9.1f}  {block:>6}  {min(samples[block]):>9.1f}  {statistics.median(samples[block]):>8.1f}  "
                f"{best:>8.1f}  {fraction_of_peak(best):>8.1%}  {best / min(samples[block]) - 1:>6.1%}"
            )
        del src
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

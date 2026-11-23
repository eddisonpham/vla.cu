"""Benchmark the Triton vector add against torch on identical buffers.

Three sizes: one that fits in cache, one at a few tens of MB, and one large enough
that the caches cannot help. Both paths write into a preallocated output, so the
only difference measured is the kernel.

Task: python -m vla_cu bench
"""

from __future__ import annotations

import torch

from kernels.triton.vector_add import BLOCK_SIZE, vector_add_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
)

SIZES = (1 << 12, 1 << 20, 1 << 24)
WARMUP = 25
ITERATIONS = 100
BYTES_PER_ELEMENT = 12
GRAPH_REPEAT = 10
GRAPH_WARMUP = 5
GRAPH_ITERATIONS = 20


def measure(n_elements: int) -> tuple[float, float, float]:
    """Return (torch GB/s, triton GB/s, graphed triton GB/s) on the same buffers.

    The graph holds GRAPH_REPEAT launches, so one replay overhead is split across
    them and the per-call number stays comparable with a plain Python launch.
    """
    x = torch.randn(n_elements, device="cuda")
    y = torch.randn(n_elements, device="cuda")
    out = torch.empty_like(x)
    n_bytes = n_elements * BYTES_PER_ELEMENT

    def torch_call() -> None:
        torch.add(x, y, out=out)

    def triton_call() -> None:
        vector_add_into(x, y, out)

    torch_s = time_call(torch_call, WARMUP, ITERATIONS)
    triton_s = time_call(triton_call, WARMUP, ITERATIONS)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_REPEAT):
            triton_call()
    graph_s = time_call(graph.replay, GRAPH_WARMUP, GRAPH_ITERATIONS) / GRAPH_REPEAT

    return (
        bandwidth_gb_s(n_bytes, torch_s),
        bandwidth_gb_s(n_bytes, triton_s),
        bandwidth_gb_s(n_bytes, graph_s),
    )


def main() -> int:
    """Print a bandwidth table, then the largest size as a fraction of peak."""
    enforce_memory_budget()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | BLOCK {BLOCK_SIZE}")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | bytes per element {BYTES_PER_ELEMENT}")
    print()
    print(
        f"{'n':>10}  {'moved MB':>9}  {'torch GB/s':>10}  {'triton GB/s':>11}  "
        f"{'graph GB/s':>10}  {'graph % peak':>12}"
    )

    for n_elements in SIZES:
        torch_gbs, triton_gbs, graph_gbs = measure(n_elements)
        moved_mb = n_elements * BYTES_PER_ELEMENT / 1e6
        ratio = fraction_of_peak(graph_gbs)
        print(
            f"{n_elements:>10}  {moved_mb:>9.2f}  {torch_gbs:>10.1f}  {triton_gbs:>11.1f}  "
            f"{graph_gbs:>10.1f}  {ratio:>11.1%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

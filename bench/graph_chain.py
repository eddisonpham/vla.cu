"""Two int8 GEMV kernels chained, launched from Python against one graph replay.

A policy forward pass is a chain of small kernels, so submission cost is paid once
per layer. The graph holds both layers and replays them as a single submission, and
the script checks the graph's output against the plain path before timing anything.

Task: python -m vla_cu graph
"""

from __future__ import annotations

import torch

from kernels.triton.int8_gemv import int8_gemv_into
from vla_cu.bench_utils import enforce_memory_budget, time_call

SIZES = ((1024, 1024, 512), (2048, 2048, 2048))
WARMUP = 20
ITERATIONS = 200
GRAPH_WARMUP = 5


def make_chain(n_in: int, n_hidden: int, n_out: int) -> tuple[torch.Tensor, ...]:
    """Return w1, s1, w2, s2, x, hidden, out with fixed shapes and int8 weights."""
    w1 = torch.randint(-127, 128, (n_hidden, n_in), device="cuda", dtype=torch.int8)
    s1 = torch.rand(n_hidden, device="cuda") * 0.02 + 0.001
    w2 = torch.randint(-127, 128, (n_out, n_hidden), device="cuda", dtype=torch.int8)
    s2 = torch.rand(n_out, device="cuda") * 0.02 + 0.001
    x = torch.randn(n_in, device="cuda")
    hidden = torch.empty(n_hidden, device="cuda")
    out = torch.empty(n_out, device="cuda")
    return w1, s1, w2, s2, x, hidden, out


def main() -> int:
    """Print plain against graphed time for each chain, plus an exactness check."""
    enforce_memory_budget()
    print(f"GPU {torch.cuda.get_device_name(0)} | two int8 GEMV kernels per chain")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | graph replays the whole chain")
    print()
    print(f"{'in -> hidden -> out':>20}  {'plain ms':>9}  {'graph ms':>9}  {'speedup':>8}  {'saved ms':>9}  {'exact':>6}")

    for n_in, n_hidden, n_out in SIZES:
        w1, s1, w2, s2, x, hidden, out = make_chain(n_in, n_hidden, n_out)

        def chain() -> None:
            int8_gemv_into(w1, s1, x, hidden)
            int8_gemv_into(w2, s2, hidden, out)

        chain()
        torch.cuda.synchronize()
        expected = out.clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain()
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        exact = torch.equal(out, expected)

        plain_s = time_call(chain, WARMUP, ITERATIONS)
        graph_s = time_call(graph.replay, GRAPH_WARMUP, ITERATIONS)

        label = f"{n_in} -> {n_hidden} -> {n_out}"
        print(
            f"{label:>20}  {plain_s * 1e3:>9.3f}  {graph_s * 1e3:>9.3f}  "
            f"{plain_s / graph_s:>7.2f}x  {(plain_s - graph_s) * 1e3:>9.3f}  {'yes' if exact else 'NO':>6}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

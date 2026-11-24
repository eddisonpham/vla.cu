"""Separate what a launch costs from what the kernel body costs, at M = 50.

Concept: every int4 path at the expert shapes reads about 1.3 MB in about 0.05 ms, 26 to
37 GB/s, whatever the program count. That is too slow for DRAM and too flat for a balance
problem, so the next suspect is the launch itself. A CUDA graph removes the host dispatch
and leaves only the GPU side of each launch, so the difference between a loop and a one
node graph is the per launch cost this stack pays, and a graph of an empty kernel is that
cost with no work behind it. The marginal cost of a node is what a persistent kernel would
stop paying once per stage.

Task: python -m vla_cu floor
"""

from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl

from kernels.triton.int4_gemm import int4_gemm_into
from vla_cu.bench_utils import enforce_memory_budget, time_call, warm_gpu
from vla_cu.quant import dequantize_int4, pack_int4

TOKENS = 50
SHAPES = ((1920, 720, 16), (720, 1920, 64))
WARMUP = 20
ITERATIONS = 200
NODES = 11


@triton.jit
def empty_kernel(ptr) -> None:
    """Store one value per program: the smallest kernel that still appears on the timeline."""
    tl.store(ptr + tl.program_id(axis=0), 0)


def capture(fn: Callable[[], None], repeats: int = 1) -> torch.cuda.CUDAGraph:
    """Return a graph that replays `fn` `repeats` times, captured after the caller warms it."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    return graph


def main() -> int:
    """Time each case as a launch loop and as a graph, and print the gap per case."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)}")
    print(f"loop: {ITERATIONS} back to back launches with CUDA events, graph: one replay of the same work")
    print()
    print(f"{'case':>26}  {'loop ms':>8}  {'graph ms':>9}  {'gap us':>7}  {'body us':>8}")

    flag = torch.zeros(32, device="cuda", dtype=torch.int32)

    def empty_one() -> None:
        empty_kernel[(1,)](flag)

    def empty_many() -> None:
        empty_kernel[(26,)](flag)

    cases: list[tuple[str, Callable[[], None]]] = [
        ("empty kernel, 1 program", empty_one),
        ("empty kernel, 26 programs", empty_many),
    ]

    generator = torch.Generator(device="cuda").manual_seed(0)
    for n_out, n_in, group in SHAPES:
        w = torch.randn((n_out, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
        packed, scales = pack_int4(w, group_size=group)
        x = torch.randn((TOKENS, n_in), device="cuda", generator=generator)
        out = torch.empty((TOKENS, n_out), device="cuda")
        dense = dequantize_int4(packed, scales, group_size=group).to(torch.bfloat16)
        half_x = x.to(torch.bfloat16)
        half_out = torch.empty((TOKENS, n_out), device="cuda", dtype=torch.bfloat16)

        def gemm() -> None:
            int4_gemm_into(packed, scales, x, out, group_size=group)

        def reference() -> None:
            torch.matmul(half_x, dense.transpose(0, 1), out=half_out)

        cases.append((f"int4 gemm {n_out} x {n_in}", gemm))
        cases.append((f"bf16 gemm {n_out} x {n_in}", reference))

    for name, plain in cases:
        plain()
        torch.cuda.synchronize()
        loop_seconds = time_call(plain, WARMUP, ITERATIONS)
        graph_seconds = float("nan")
        try:
            graph = capture(plain)
            graph_seconds = time_call(graph.replay, WARMUP, ITERATIONS)
        except RuntimeError as error:
            print(f"  capture failed for {name}: {error}")
        gap = (loop_seconds - graph_seconds) * 1e6
        print(
            f"{name:>26}  {loop_seconds * 1e3:>8.3f}  {graph_seconds * 1e3:>9.3f}  {gap:>7.2f}  {graph_seconds * 1e6:>8.2f}"
        )

    n_out, n_in, group = SHAPES[0]
    w = torch.randn((n_out, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
    packed, scales = pack_int4(w, group_size=group)
    x = torch.randn((TOKENS, n_in), device="cuda", generator=generator)
    out = torch.empty((TOKENS, n_out), device="cuda")

    def gemm() -> None:
        int4_gemm_into(packed, scales, x, out, group_size=group)

    gemm()
    torch.cuda.synchronize()
    print()
    print(f"marginal cost of a launch inside a graph, {n_out} x {n_in}, {TOKENS} tokens")
    print(f"{'nodes':>6}  {'graph ms':>8}  {'per node us':>11}")
    graph_one = capture(gemm)
    graph_many = capture(gemm, NODES)
    one = time_call(graph_one.replay, WARMUP, ITERATIONS)
    many = time_call(graph_many.replay, WARMUP, ITERATIONS)
    print(f"{1:>6}  {one * 1e3:>8.3f}  {one * 1e6:>11.2f}")
    print(f"{NODES:>6}  {many * 1e3:>8.3f}  {many * 1e6 / NODES:>11.2f}")
    print(f"{'marginal':>6}  {'':>8}  {(many - one) * 1e6 / (NODES - 1):>11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

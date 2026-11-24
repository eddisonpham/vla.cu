"""The public API end to end on synthetic weights: one matrix, one block, one stack.

Concept: this is the file to read after the README. It uses only names exported from
`vla_cu`, at the SmolVLA widths, and prints accuracy beside time so a reader can see
what the speed costs. Weights are random, so these times are this GPU's layer times and
not a model's.

Task: python -m vla_cu demo
"""

from __future__ import annotations

import torch

from vla_cu import Int4Linear, MlpBlock
from vla_cu.bench_utils import enforce_memory_budget, time_call, warm_gpu

HIDDEN = 960
INTERMEDIATE = 2560
STACK = 4
WARMUP = 10
ITERATIONS = 50


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the cosine similarity of two vectors."""
    return torch.nn.functional.cosine_similarity(approx, exact, dim=0).item()


def random_matrix(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """Return fp32 weights [rows, cols] at the scale of a transformer projection."""
    return torch.randn((rows, cols), device="cuda", generator=generator) * (1.0 / cols**0.5)


def main() -> int:
    """Run the three usage levels and print time with accuracy for each."""
    enforce_memory_budget()
    warm_gpu()
    generator = torch.Generator(device="cuda").manual_seed(0)
    activation = torch.nn.functional.silu
    x = torch.randn(HIDDEN, device="cuda", generator=generator)

    print(f"GPU {torch.cuda.get_device_name(0)} | warmup {WARMUP} | iterations {ITERATIONS}")
    print()

    wide = random_matrix(8192, 4096, generator)
    linear = Int4Linear(wide)
    dense = torch.randn(4096, device="cuda", generator=generator)
    exact = torch.mv(wide, dense)
    approx = linear(dense)
    print(f"one matrix, 8192 -> 4096, batch 1, group {linear.group_size}, {wide.numel() * 4 / 1e6:.0f} MB fp32")
    print(f"{'path':>14}  {'ms':>8}  {'cos':>8}")
    fp32_seconds = time_call(lambda: torch.mv(wide, dense), WARMUP, ITERATIONS)
    int4_seconds = time_call(lambda: linear(dense), WARMUP, ITERATIONS)
    print(f"{'fp32 mv':>14}  {fp32_seconds * 1e3:>8.3f}  {1.0:>8.5f}")
    print(f"{'int4 linear':>14}  {int4_seconds * 1e3:>8.3f}  {cosine(approx, exact):>8.5f}")
    print()

    down = random_matrix(HIDDEN, INTERMEDIATE, generator)
    up = random_matrix(INTERMEDIATE, HIDDEN, generator)
    block = MlpBlock(up, down)
    out = torch.empty(HIDDEN, device="cuda")
    reference_hidden = torch.empty(INTERMEDIATE, device="cuda")
    reference_out = torch.empty(HIDDEN, device="cuda")

    def block_reference() -> None:
        torch.mv(up, x, out=reference_hidden)
        activation(reference_hidden, inplace=True)
        torch.mv(down, reference_hidden, out=reference_out)

    block_reference()
    torch.cuda.synchronize()
    block_exact = reference_out.clone()
    block.into(x, out)
    torch.cuda.synchronize()
    eager_out = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        block.into(x, out)
    graph.replay()
    torch.cuda.synchronize()
    graph_out = out.clone()

    print(f"one MLP block, {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}, batch 1")
    print(f"{'path':>14}  {'ms':>8}  {'cos':>8}")
    fp32_block = time_call(block_reference, WARMUP, ITERATIONS)
    print(f"{'fp32':>14}  {fp32_block * 1e3:>8.3f}  {1.0:>8.5f}")
    print(f"{'int4 eager':>14}  {time_call(lambda: block.into(x, out), WARMUP, ITERATIONS) * 1e3:>8.3f}  {cosine(eager_out, block_exact):>8.5f}")
    graph_block = time_call(graph.replay, WARMUP, ITERATIONS)
    print(f"{'int4 graph':>14}  {graph_block * 1e3:>8.3f}  {cosine(graph_out, block_exact):>8.5f}")
    print(f"graph reproduces the eager block bit for bit: {torch.equal(graph_out, eager_out)}")
    print()

    weights = [(up, down)]
    blocks = [block]
    for _ in range(STACK - 1):
        layer_up = random_matrix(INTERMEDIATE, HIDDEN, generator)
        layer_down = random_matrix(HIDDEN, INTERMEDIATE, generator)
        weights.append((layer_up, layer_down))
        blocks.append(MlpBlock(layer_up, layer_down))

    current = torch.empty(HIDDEN, device="cuda")
    following = torch.empty(HIDDEN, device="cuda")
    fp32_a = torch.empty(HIDDEN, device="cuda")
    fp32_b = torch.empty(HIDDEN, device="cuda")

    def chain() -> torch.Tensor:
        source = x
        for index, layer in enumerate(blocks):
            target = following if index % 2 == 0 else current
            layer.into(source, target)
            source = target
        return source

    def fp32_chain() -> None:
        source = x
        for layer_up, layer_down in weights:
            torch.mv(layer_up, source, out=fp32_a)
            activation(fp32_a, inplace=True)
            torch.mv(layer_down, fp32_a, out=fp32_b)
            source = fp32_b

    fp32_chain()
    torch.cuda.synchronize()
    stack_exact = fp32_b.clone()
    eager_stack = chain().clone()
    torch.cuda.synchronize()
    stack_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(stack_graph):
        chain()
    stack_graph.replay()
    torch.cuda.synchronize()
    final = following if (STACK - 1) % 2 == 0 else current
    graph_stack = final.clone()

    print(f"stack of {STACK} blocks, batch 1")
    print(f"{'path':>14}  {'ms':>8}  {'cos':>8}")
    fp32_stack = time_call(fp32_chain, WARMUP, ITERATIONS)
    print(f"{'fp32':>14}  {fp32_stack * 1e3:>8.3f}  {1.0:>8.5f}")
    graph_stack_seconds = time_call(stack_graph.replay, WARMUP, ITERATIONS)
    print(f"{'int4 graph':>14}  {graph_stack_seconds * 1e3:>8.3f}  {cosine(graph_stack, stack_exact):>8.5f}")
    print(f"graph reproduces the eager stack bit for bit: {torch.equal(graph_stack, eager_stack)}")
    print(f"speedup of the graphed stack over fp32: {fp32_stack / graph_stack_seconds:.2f}x")
    print()
    print("note: one token uses the GEMV; pass [tokens, in_features] to the same objects for a chunk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

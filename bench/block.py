"""A stack of full action blocks: attention plus MLP, at four precision mixes.

Concept: one block says little about a model, because quantization error compounds
through a stack. This task runs the same fp32 weights through four blocks in three
mixes and reports cosine after every depth, so the projection that binds accuracy is
named rather than guessed. The mixes are int4 attention with int4 down, int8 attention
with int4 down, and int8 everywhere.

Timing is one CUDA graph over the four blocks, because a block is four launches and at
these sizes dispatch costs more than the kernels.

Task: python -m vla_cu block
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    warm_gpu,
)
from vla_cu.blocks import ActionBlock, AttentionBlock, MlpBlock

HIDDEN = 960
INTERMEDIATE = 2560
STACK_BLOCKS = 4
WARMUP = 10
ITERATIONS = 50
MIXES = (
    (4, 4, "int4 attn, int4 down"),
    (4, 8, "int4 attn, int8 down"),
    (8, 4, "int8 attn, int4 down"),
    (8, 8, "int8 attn, int8 down"),
)


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the cosine similarity of two vectors."""
    return F.cosine_similarity(approx, exact, dim=0).item()


def make_weights(count: int, generator: torch.Generator) -> list[dict[str, torch.Tensor]]:
    """Return `count` distinct fp32 attention and MLP weight sets at the SmolVLA widths."""
    scale = 1.0 / HIDDEN**0.5
    weights = []
    for _ in range(count):
        weights.append(
            {
                "wq": torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale,
                "wk": torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale,
                "wv": torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale,
                "wo": torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale,
                "up": torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * scale,
                "down": torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * scale,
            }
        )
    return weights


def main() -> int:
    """Run the fp32 chain and three quantized graphs, and print cosine per depth."""
    enforce_memory_budget()
    warm_gpu()
    generator = torch.Generator(device="cuda").manual_seed(0)
    weights = make_weights(STACK_BLOCKS, generator)
    x = torch.randn(HIDDEN, device="cuda", generator=generator)
    qkv_buffer = torch.empty(3 * HIDDEN, device="cuda")
    attention_buffer = torch.empty(HIDDEN, device="cuda")
    hidden_buffer = torch.empty(INTERMEDIATE, device="cuda")
    output_buffer = torch.empty(HIDDEN, device="cuda")

    def fp32_chain(count: int) -> torch.Tensor:
        current = x
        for index in range(count):
            w = weights[index]
            torch.mv(w["wq"], current, out=qkv_buffer[:HIDDEN])
            torch.mv(w["wk"], current, out=qkv_buffer[HIDDEN : 2 * HIDDEN])
            torch.mv(w["wv"], current, out=qkv_buffer[2 * HIDDEN :])
            torch.mv(w["wo"], qkv_buffer[2 * HIDDEN :], out=attention_buffer)
            torch.mv(w["up"], attention_buffer, out=hidden_buffer)
            F.silu(hidden_buffer, inplace=True)
            torch.mv(w["down"], hidden_buffer, out=output_buffer)
            current = output_buffer
        return current

    reference = []
    for depth in range(1, STACK_BLOCKS + 1):
        reference.append(fp32_chain(depth).clone())
    torch.cuda.synchronize()

    fp32_bytes = STACK_BLOCKS * (4 * HIDDEN * HIDDEN + 2 * INTERMEDIATE * HIDDEN) * 4
    fp32_seconds = time_call(lambda: fp32_chain(STACK_BLOCKS), WARMUP, ITERATIONS)

    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s")
    print(f"action block {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}, attention {HIDDEN}, stack of {STACK_BLOCKS}, batch 1")
    print(f"norms and residuals are not modelled, so these are the projections alone")
    print(f"fp32 bytes {fp32_bytes / 1e6:.1f} MB | warmup {WARMUP} | iterations {ITERATIONS}")
    print()
    print(f"{'mix':>22}  " + "  ".join(f"{'cos@' + str(depth):>8}" for depth in range(1, STACK_BLOCKS + 1)) + f"  {'MB':>7}  {'ms':>8}  {'ms/block':>8}  {'GB/s':>8}  {'% PEAK':>7}")

    for attention_bits, down_bits, label in MIXES:
        blocks = [
            ActionBlock(
                AttentionBlock(w["wq"], w["wk"], w["wv"], w["wo"], bits=attention_bits),
                MlpBlock(w["up"], w["down"], down_bits=down_bits),
            )
            for w in weights
        ]
        outputs = [torch.empty(HIDDEN, device="cuda") for _ in range(STACK_BLOCKS)]

        def chain(count: int, blocks=blocks, outputs=outputs) -> torch.Tensor:
            current = x
            for index in range(count):
                blocks[index].into(current, outputs[index])
                current = outputs[index]
            return current

        cosines = []
        for depth in range(1, STACK_BLOCKS + 1):
            out = chain(depth)
            torch.cuda.synchronize()
            cosines.append(cosine(out, reference[depth - 1]))

        eager = chain(STACK_BLOCKS).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain(STACK_BLOCKS)
        graph.replay()
        single = torch.cuda.CUDAGraph()
        with torch.cuda.graph(single):
            blocks[0].into(x, outputs[0])
        single.replay()
        torch.cuda.synchronize()
        bit_exact = torch.equal(outputs[-1], eager)
        seconds = time_call(graph.replay, WARMUP, ITERATIONS)
        single_seconds = time_call(single.replay, WARMUP, ITERATIONS)
        n_bytes = sum(block.bytes for block in blocks)
        gbs = bandwidth_gb_s(n_bytes, seconds)
        print(
            f"{label:>22}  " + "  ".join(f"{value:>8.5f}" for value in cosines) +
            f"  {n_bytes / 1e6:>7.1f}  {seconds * 1e3:>8.3f}  {seconds / STACK_BLOCKS * 1e3:>8.3f}  "
            f"{gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}"
        )
        print(f"{'':>22}  graph bit exact {bit_exact} | one block graphed {single_seconds * 1e3:.3f} ms")
        del blocks, outputs, graph, single

    print()
    print(f"fp32 stack {fp32_seconds * 1e3:.3f} ms, {fp32_seconds / STACK_BLOCKS * 1e3:.3f} ms per block, {bandwidth_gb_s(fp32_bytes, fp32_seconds):.1f} GB/s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

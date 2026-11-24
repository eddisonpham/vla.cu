"""The whole action expert at 50 tokens, int8 against int4 down, against the fp32 chain.

Concept: the expert runs over an action chunk, so every projection is a GEMM over 50 rows,
and every second layer cross attends to a 960 wide prefix while q reads the 720 wide expert
stream. This builds every expert layer through `layer_from_checkpoint` at 50 tokens, checks
each depth against fp32 with the same fixed prefix, and times captured graphs.

Task: python -m vla_cu expertstack
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from bench.real_layers import CHECKPOINT, cosine, relative_error
from kernels.triton.rmsnorm import EPS
from vla_cu.bench_utils import enforce_memory_budget, time_call, warm_gpu
from vla_cu.blocks import gqa_mix
from vla_cu.build import layer_from_checkpoint
from vla_cu.checkpoint import open_checkpoint

LAYERS = 16
TOKENS = 50
WARMUP = 10
ITERATIONS = 50
WEIGHTS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def read_reference(checkpoint, index: int) -> tuple[torch.Tensor, ...]:
    """Return the nine fp32 tensors of one expert layer, in the order the reference unpacks them."""
    prefix = f"model.vlm_with_expert.lm_expert.layers.{index}"
    return tuple(checkpoint.read(f"{prefix}.{name}").to(device="cuda", dtype=torch.float32) for name in WEIGHTS)


def normalize(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the row wise RMSNorm of `x` [tokens, hidden] scaled by `weight` [hidden]."""
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS) * weight


def reference_layer(
    x: torch.Tensor, weights: tuple[torch.Tensor, ...], other: torch.Tensor | None
) -> torch.Tensor:
    """Return one expert layer in fp32 for `x` [tokens, hidden], cross attending to `other` when given."""
    input_norm, post_norm, wq, wk, wv, wo, gate, up, down = weights
    normed = normalize(x, input_norm)
    source = normed if other is None else other
    residual = x + (gqa_mix(normed @ wq.T, source @ wk.T, source @ wv.T) @ wo.T)
    normed = normalize(residual, post_norm)
    return residual + (F.silu(normed @ gate.T) * (normed @ up.T)) @ down.T


def run_stack(layers, prefixes, x, first, second, result) -> None:
    """Run every layer from `x`, alternating two chunk buffers, and copy the last output into `result`."""
    current, target = first, second
    current.copy_(x)
    for layer, other in zip(layers, prefixes):
        layer.into(current, target, other)
        current, target = target, current
    result.copy_(current)


def main() -> int:
    """Build every expert layer at each down precision, check against fp32, then time the graph replays."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    print(f"GPU {torch.cuda.get_device_name(0)} | {checkpoint.path.name} | {LAYERS} expert layers | {TOKENS} tokens")
    print(f"warmup {WARMUP}, iterations {ITERATIONS}, down at int8 against int4, gate and up at int8")

    int8_layers = [
        layer_from_checkpoint(checkpoint, index, tokens=TOKENS, down_bits=8, part="expert") for index in range(LAYERS)
    ]
    int4_layers = [
        layer_from_checkpoint(checkpoint, index, tokens=TOKENS, down_bits=4, part="expert") for index in range(LAYERS)
    ]
    hidden = int8_layers[0].hidden_width
    generator = torch.Generator(device="cuda").manual_seed(0)
    prefixes = []
    for layer in int8_layers:
        width = layer.attention.other_width
        prefixes.append(None if width == hidden else torch.randn((TOKENS, width), device="cuda", generator=generator))
    x = torch.randn((TOKENS, hidden), device="cuda", generator=generator)
    reference = [read_reference(checkpoint, index) for index in range(LAYERS)]

    fp32_first = torch.empty_like(x)
    fp32_second = torch.empty_like(x)
    fp32_result = torch.empty_like(x)

    def fp32_chain() -> None:
        current, target = fp32_first, fp32_second
        current.copy_(x)
        for weights, other in zip(reference, prefixes):
            target.copy_(reference_layer(current, weights, other))
            current, target = target, current
        fp32_result.copy_(current)

    fp32_chain()
    torch.cuda.synchronize()
    exact = fp32_result.clone()

    states = {}
    for bits, layers in ((8, int8_layers), (4, int4_layers)):
        first, second = torch.empty_like(x), torch.empty_like(x)
        source = x
        per_layer = []
        for index, (layer, other) in enumerate(zip(layers, prefixes)):
            target = first if index % 2 == 0 else second
            layer.into(source, target, other)
            source = target
            per_layer.append(target.clone())
        states[bits] = per_layer
    torch.cuda.synchronize()

    current = x.clone()
    print()
    print(f"{'layer':>5}  {'int8 cos':>9}  {'int4 cos':>9}")
    for index in range(LAYERS):
        current = reference_layer(current, reference[index], prefixes[index])
        print(
            f"{index + 1:>5}  {cosine(states[8][index].reshape(-1), current.reshape(-1)):>9.5f}  "
            f"{cosine(states[4][index].reshape(-1), current.reshape(-1)):>9.5f}"
        )

    stacks = {}
    for bits, layers in ((8, int8_layers), (4, int4_layers)):
        first, second, result = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
        run_stack(layers, prefixes, x, first, second, result)
        torch.cuda.synchronize()
        eager = states[bits][-1]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_stack(layers, prefixes, x, first, second, result)
        first.zero_()
        second.zero_()
        result.zero_()
        graph.replay()
        torch.cuda.synchronize()
        stacks[bits] = (layers, graph, eager, torch.equal(result, eager))

    fp32_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fp32_graph):
        fp32_chain()
    fp32_first.zero_()
    fp32_second.zero_()
    fp32_result.zero_()
    fp32_graph.replay()
    torch.cuda.synchronize()
    fp32_exact = torch.equal(fp32_result, exact)
    fp32_seconds = time_call(fp32_graph.replay, WARMUP, ITERATIONS)
    fp32_bytes = sum(weight.numel() * 4 for weights in reference for weight in weights)
    print()
    print(f"{'path':>10}  {'MB':>7}  {'ms':>8}  {'x fp32':>7}  {'cos':>9}  {'rel L2':>9}  {'exact':>5}")
    print(
        f"{'fp32':>10}  {fp32_bytes / 1e6:>7.2f}  {fp32_seconds * 1e3:>8.3f}  {'1.00x':>7}  "
        f"{'1.00000':>9}  {'0.00e+00':>9}  {'yes' if fp32_exact else 'NO':>5}"
    )
    for bits, label in ((8, "int8 down"), (4, "int4 down")):
        layers, graph, eager, bit_exact = stacks[bits]
        seconds = time_call(graph.replay, WARMUP, ITERATIONS)
        n_bytes = sum(layer.bytes for layer in layers)
        print(
            f"{label:>10}  {n_bytes / 1e6:>7.2f}  {seconds * 1e3:>8.3f}  {fp32_seconds / seconds:>6.2f}x  "
            f"{cosine(eager.reshape(-1), exact.reshape(-1)):>9.5f}  "
            f"{relative_error(eager.reshape(-1), exact.reshape(-1)):>9.2e}  {'yes' if bit_exact else 'NO':>5}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

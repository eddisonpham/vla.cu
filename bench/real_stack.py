"""All 16 real SmolVLA backbone layers at int8 and int4, against the fp32 chain.

Concept: EXP-043 read single matrices and EXP-036 measured four blocks on synthetic weights.
This builds every backbone layer of the real checkpoint through `layer_from_checkpoint`, so
the cosine includes both norms, the grouped attention and the gated MLP over the full depth.
Each stack is captured as one graph and timed against the same layers in fp32.

Task: python -m vla_cu realstack
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
    """Return the nine fp32 tensors of one backbone layer, in the order `reference_layer` unpacks them."""
    prefix = f"model.vlm_with_expert.vlm.model.text_model.layers.{index}"
    return tuple(checkpoint.read(f"{prefix}.{name}").to(device="cuda", dtype=torch.float32) for name in WEIGHTS)


def reference_layer(x: torch.Tensor, weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return one layer in fp32 from a [hidden] vector, the chain the quantized blocks approximate."""
    input_norm, post_norm, wq, wk, wv, wo, gate, up, down = weights
    normed = x * torch.rsqrt(x.square().mean() + EPS) * input_norm
    residual = x + (gqa_mix(normed[None] @ wq.T, normed[None] @ wk.T, normed[None] @ wv.T)[0] @ wo.T)
    normed = residual * torch.rsqrt(residual.square().mean() + EPS) * post_norm
    return residual + (F.silu(normed @ gate.T) * (normed @ up.T)) @ down.T


def run_stack(
    layers,
    x: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    result: torch.Tensor,
) -> None:
    """Run every layer from `x`, alternating two row buffers, and copy the last output into `result`."""
    current, target = first, second
    current.copy_(x)
    for layer in layers:
        layer.into(current, target)
        current, target = target, current
    result.copy_(current)


def main() -> int:
    """Build three real layers at each down precision, check both against fp32, then time the graph replays."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    print(f"GPU {torch.cuda.get_device_name(0)} | {checkpoint.path.name} | {LAYERS} backbone layers | one token")
    print(f"warmup {WARMUP}, iterations {ITERATIONS}, down at int8 against int4, gate and up at int8")

    int8_layers = [layer_from_checkpoint(checkpoint, index, tokens=1, down_bits=8) for index in range(LAYERS)]
    int4_layers = [layer_from_checkpoint(checkpoint, index, tokens=1, down_bits=4) for index in range(LAYERS)]
    hidden = int8_layers[0].hidden_width
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((1, hidden), device="cuda", generator=generator)

    reference = [read_reference(checkpoint, index) for index in range(LAYERS)]
    fp32_first = torch.empty_like(x)
    fp32_second = torch.empty_like(x)
    fp32_result = torch.empty_like(x)

    def fp32_chain() -> None:
        current, target = fp32_first, fp32_second
        current.copy_(x)
        for weights in reference:
            target.copy_(reference_layer(current[0], weights)[None])
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
        for index, layer in enumerate(layers):
            target = first if index % 2 == 0 else second
            layer.into(source, target)
            source = target
            per_layer.append(target.clone())
        states[bits] = per_layer
    torch.cuda.synchronize()

    current = x.clone()
    print()
    print(f"{'layer':>5}  {'int8 cos':>9}  {'int4 cos':>9}")
    for index in range(LAYERS):
        current = reference_layer(current[0], reference[index])[None]
        print(f"{index + 1:>5}  {cosine(states[8][index][0], current[0]):>9.5f}  {cosine(states[4][index][0], current[0]):>9.5f}")

    stacks = {}
    for bits, layers in ((8, int8_layers), (4, int4_layers)):
        first, second, result = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
        run_stack(layers, x, first, second, result)
        torch.cuda.synchronize()
        eager = states[bits][-1]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_stack(layers, x, first, second, result)
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
            f"{cosine(eager[0], exact[0]):>9.5f}  {relative_error(eager[0], exact[0]):>9.2e}  {'yes' if bit_exact else 'NO':>5}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

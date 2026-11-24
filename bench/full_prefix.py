"""The full 241 row prefix and the expert that reads it, int8 against int4 down.

Concept: a deployed prefill is 3 frames of 64 image rows through SigLIP and the connector,
48 language rows, and 1 state row. This builds that cache from the real tower, connector
and backbone, then runs the 16 expert layers at 50 action tokens on 241 keys. Frames are
seeded noise because no camera data ships with the repo, and the state is a zero 32 wide
vector because no robot state does either.

Task: python -m vla_cu fullprefix
"""

from __future__ import annotations

import torch

from bench.expert_stack import read_reference, reference_layer, run_stack
from bench.real_layers import CHECKPOINT, cosine, relative_error
from vla_cu.bench_utils import enforce_memory_budget, time_call, warm_gpu
from vla_cu.build import layer_from_checkpoint
from vla_cu.checkpoint import open_checkpoint
from vla_cu.prefix import FULL_TOKENS, cache_from_rows, full_prefix_inputs

LAYERS = 16
TOKENS = 50
WARMUP = 10
ITERATIONS = 50
BITS = (8, 4)


def main() -> int:
    """Build the full prefix cache, then check and time the expert at both down precisions."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    generator = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.rand((3, 3, 512, 512), device="cuda", generator=generator) * 2 - 1
    state = torch.zeros(32, device="cuda")
    print(f"GPU {torch.cuda.get_device_name(0)} | full prefix {FULL_TOKENS} rows = 3 x 64 image + 48 language + 1 state")
    print("frames are seeded uniform noise and the state is zeros, not measured inputs")

    rows, visible = full_prefix_inputs(checkpoint, frames, state)
    cached_keys, cached_values = cache_from_rows(checkpoint, rows, visible)
    n_bytes = sum(t.numel() * 4 for t in cached_keys + cached_values)
    build_seconds = time_call(lambda: cache_from_rows(checkpoint, rows, visible), 1, 3)
    print(f"cache {len(cached_keys)} layers x {tuple(cached_keys[0].shape)} fp32, {n_bytes / 1e6:.2f} MB")
    print(f"prefix build {build_seconds * 1e3:.2f} ms per call, which includes reading the weights from disk")
    print(f"warmup {WARMUP}, iterations {ITERATIONS}, expert attention at int8")

    def build(index: int, bits: int):
        """Build one expert layer, sizing the cross buffers for the 241 prefix rows."""
        other = FULL_TOKENS if index % 2 else TOKENS
        return layer_from_checkpoint(
            checkpoint, index, tokens=TOKENS, down_bits=bits, part="expert", other_tokens=other
        )

    variants = [(bits, [build(index, bits) for index in range(LAYERS)]) for bits in BITS]
    hidden = variants[0][1][0].hidden_width
    prefixes = [
        None if layer.attention.other_width == hidden else (cached_keys[index], cached_values[index])
        for index, layer in enumerate(variants[0][1])
    ]
    x = torch.randn((TOKENS, hidden), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))

    reference = [read_reference(checkpoint, index) for index in range(LAYERS)]
    fp32_first, fp32_second, fp32_result = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)

    def fp32_chain() -> None:
        current, target = fp32_first, fp32_second
        current.copy_(x)
        for weights, prefix in zip(reference, prefixes):
            target.copy_(reference_layer(current, weights, prefix))
            current, target = target, current
        fp32_result.copy_(current)

    fp32_chain()
    torch.cuda.synchronize()
    exact = fp32_result.clone()

    stacks = {}
    states = {}
    for bits, layers in variants:
        first, second = torch.empty_like(x), torch.empty_like(x)
        source = x
        per_layer = []
        for index, (layer, prefix) in enumerate(zip(layers, prefixes)):
            target = first if index % 2 == 0 else second
            if prefix is None:
                layer.into(source, target)
            else:
                layer.into(source, target, prefix[0], prefix[1])
            source = target
            per_layer.append(target.clone())
        states[bits] = per_layer

        first, second, result = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
        run_stack(layers, prefixes, x, first, second, result)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_stack(layers, prefixes, x, first, second, result)
        first.zero_()
        second.zero_()
        result.zero_()
        graph.replay()
        torch.cuda.synchronize()
        stacks[bits] = (graph, torch.equal(result, per_layer[-1]))
    torch.cuda.synchronize()

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
    print(f"{'down':>8}  {'MB':>7}  {'ms':>8}  {'x fp32':>7}  {'cos':>9}  {'rel L2':>9}  {'exact':>5}")
    print(
        f"{'fp32':>8}  {fp32_bytes / 1e6:>7.2f}  {fp32_seconds * 1e3:>8.3f}  {'1.00x':>7}  "
        f"{'1.00000':>9}  {'0.00e+00':>9}  {'yes' if fp32_exact else 'NO':>5}"
    )
    for bits, layers in variants:
        graph, bit_exact = stacks[bits]
        seconds = time_call(graph.replay, WARMUP, ITERATIONS)
        label = f"int{bits} down"
        print(
            f"{label:>8}  {sum(layer.bytes for layer in layers) / 1e6:>7.2f}  {seconds * 1e3:>8.3f}  "
            f"{fp32_seconds / seconds:>6.2f}x  {cosine(states[bits][-1].reshape(-1), exact.reshape(-1)):>9.5f}  "
            f"{relative_error(states[bits][-1].reshape(-1), exact.reshape(-1)):>9.2e}  {'yes' if bit_exact else 'NO':>5}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

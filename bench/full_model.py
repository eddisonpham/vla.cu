"""The whole policy path in one CUDA graph: 3 frames, 16 backbone layers, 16 expert layers.

Concept: the deployed prefix is 241 rows, 192 image rows from the SigLIP tower at int8 plus
48 language rows and 1 state row. The backbone leaves the key and value cache that the
expert cross layers read; the expert then runs the 50 action tokens. One graph covers both
stacks, and its replay is compared with the same work in two graphs and with eager
dispatch. Cosine is against an fp32 chain that starts from the fp32 tower's rows.

Task: python -m vla_cu fullmodel
"""

from __future__ import annotations

import torch

from bench.expert_stack import read_reference, reference_layer
from bench.real_layers import CHECKPOINT, cosine, relative_error
from vla_cu.bench_utils import enforce_memory_budget, time_call, warm_gpu
from vla_cu.blocks import rotary
from vla_cu.build import layer_from_checkpoint
from vla_cu.checkpoint import open_checkpoint
from vla_cu.prefix import TOKENS, cache_from_rows, full_prefix_inputs

LAYERS = 16
TOKENS = 50
WARMUP = 10
ITERATIONS = 50


def main() -> int:
    """Capture both stacks in one graph, check it against eager and fp32, then time every replay path."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    generator = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.rand((3, 3, 512, 512), device="cuda", generator=generator) * 2 - 1
    state = torch.zeros(32, device="cuda")
    prefix_input, visible = full_prefix_inputs(checkpoint, frames, state, tower_bits=8)
    reference_rows, _ = full_prefix_inputs(checkpoint, frames, state, tower_bits=32)
    rows = prefix_input.shape[0]
    print(f"GPU {torch.cuda.get_device_name(0)} | {checkpoint.path.name} | {LAYERS} backbone + {LAYERS} expert layers")
    print(f"backbone {rows} prefix rows bidirectional from the int8 tower, expert {TOKENS} action tokens, attention and down at int8")
    print("the state row sees every row and no other row sees it, which is the block mask the backbone applies")
    print("frames are seeded noise and the state is zeros; the cosine is against an fp32 tower and fp32 layers")
    print(f"warmup {WARMUP}, iterations {ITERATIONS}")

    backbone = [
        layer_from_checkpoint(checkpoint, index, tokens=rows, down_bits=8, causal=False)
        for index in range(LAYERS)
    ]
    expert = [
        layer_from_checkpoint(
            checkpoint,
            index,
            tokens=TOKENS,
            down_bits=8,
            part="expert",
            other_tokens=rows if index % 2 else TOKENS,
        )
        for index in range(LAYERS)
    ]
    kv_width = backbone[0].attention.k_out.shape[1]
    expert_input = torch.randn(
        (TOKENS, expert[0].hidden_width), device="cuda", generator=torch.Generator(device="cuda").manual_seed(0)
    )
    cached_keys = [torch.empty((rows, kv_width), device="cuda") for _ in range(LAYERS)]
    cached_values = [torch.empty((rows, kv_width), device="cuda") for _ in range(LAYERS)]
    prefix_first, prefix_second = torch.empty_like(prefix_input), torch.empty_like(prefix_input)
    expert_first, expert_second = torch.empty_like(expert_input), torch.empty_like(expert_input)
    result = torch.empty_like(expert_input)

    def run_prefix() -> None:
        """Run the backbone and leave the post-rope keys and values of every layer in the cache."""
        current, target = prefix_first, prefix_second
        current.copy_(prefix_input)
        for index, layer in enumerate(backbone):
            layer.into(current, target, visible=visible)
            current, target = target, current
            cached_keys[index].copy_(rotary(layer.attention.k_out))
            cached_values[index].copy_(layer.attention.v_out)

    def run_expert() -> None:
        """Run the expert from its chunk input into `result`, cross attending on the odd layers."""
        source, other = expert_first, expert_second
        source.copy_(expert_input)
        for index, layer in enumerate(expert):
            if index % 2 == 0:
                layer.into(source, other)
            else:
                layer.into(source, other, cached_keys[index], cached_values[index])
            source, other = other, source
        result.copy_(source)

    def model() -> None:
        """Run the whole path into `result`."""
        run_prefix()
        run_expert()

    model()
    torch.cuda.synchronize()
    eager = result.clone()
    model()
    torch.cuda.synchronize()
    repeat_exact = torch.equal(result, eager)
    eager_seconds = time_call(model, WARMUP, ITERATIONS)

    whole = torch.cuda.CUDAGraph()
    with torch.cuda.graph(whole):
        model()
    result.zero_()
    whole.replay()
    torch.cuda.synchronize()
    whole_exact = torch.equal(result, eager)
    whole_seconds = time_call(whole.replay, WARMUP, ITERATIONS)

    backbone_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(backbone_graph):
        run_prefix()
    expert_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(expert_graph):
        run_expert()

    def pair() -> None:
        """Replay the two stacks as two graphs, one after the other."""
        backbone_graph.replay()
        expert_graph.replay()

    result.zero_()
    pair()
    torch.cuda.synchronize()
    pair_exact = torch.equal(result, eager)
    pair_seconds = time_call(pair, WARMUP, ITERATIONS)

    cached_k, cached_v = cache_from_rows(checkpoint, reference_rows, visible)
    reference = [read_reference(checkpoint, index) for index in range(LAYERS)]
    prefixes = [
        None if layer.attention.other_width == expert[0].hidden_width else (cached_k[index], cached_v[index])
        for index, layer in enumerate(expert)
    ]
    exact = expert_input
    for index in range(LAYERS):
        exact = reference_layer(exact, reference[index], prefixes[index])
    torch.cuda.synchronize()

    stored = sum(layer.bytes for layer in backbone) + sum(layer.bytes for layer in expert)
    print()
    print(f"{'path':>16}  {'ms':>8}  {'x eager':>8}")
    for label, seconds in (("eager", eager_seconds), ("two graphs", pair_seconds), ("one graph", whole_seconds)):
        print(f"{label:>16}  {seconds * 1e3:>8.3f}  {eager_seconds / seconds:>7.2f}x")
    print()
    print(
        f"stored {stored / 1e6:.2f} MB | cosine {cosine(result.reshape(-1), exact.reshape(-1)):>9.5f} | "
        f"rel L2 {relative_error(result.reshape(-1), exact.reshape(-1)):>9.2e}"
    )
    print(f"exact: one graph {'yes' if whole_exact else 'NO'} | two graphs {'yes' if pair_exact else 'NO'} | repeat {'yes' if repeat_exact else 'NO'}")
    print(f"tower contribution alone, int8 rows against fp32 rows: {cosine(prefix_input.reshape(-1), reference_rows.reshape(-1)):>9.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

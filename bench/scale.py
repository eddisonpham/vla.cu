"""Iterative scaling: shapes, precision, and stack depth, grown until the GPU says stop.

Concept: a kernel has three ways to stop being fast, and each stage isolates one. Shapes grow
until the launch stops mattering and DRAM becomes the wall. Group size grows the scale table
until bytes per weight dominate accuracy. Depth grows the number of weight streams per token
until a block's cost is the whole model's cost. Nothing here is tuned; each row is the same
kernel with one number changed.

Task: python -m vla_cu scale
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from bench.real_layers import CHECKPOINT, MATRICES, cosine, pad_columns, relative_error, run_int4
from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.rmsnorm import EPS, add_into, add_norm_into, rmsnorm_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    warm_gpu,
)
from vla_cu.blocks import GEMM_MIN_TOKENS, THETA, GqaTokenBlock, GqaValueBlock, MlpBlock, MlpTokenBlock, gqa_mix
from vla_cu.checkpoint import open_checkpoint
from vla_cu.quant import GROUP_SIZE, dequantize_int4, pack_int4

SHAPES = ((256, 1024), (1024, 1024), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 16384), (16384, 32768))
GROUPS = (64, 32, 16, 8)
STACKS = (1, 2, 4, 8, 16)
TOKENS = (2, 3, 4, 8)
CHUNK_TOKENS = 50
CHUNK_STACKS = (1, 4, 16)
CHUNK_WARMUP = 5
CHUNK_ITERATIONS = 10
EXPERT_LAYERS = 16
WARMUP = 20
ITERATIONS = 100
WORKING_CAP_BYTES = 4 * 1024**3
VERIFY_MAX_ELEMENTS = 8192 * 16384


def shape_sweep() -> None:
    """Time int4 at each shape in the grid, verifying where a dequantized copy still fits.

    Weights are random nibbles rather than a model, because only the byte count sets the
    expected time here. A shape whose working set would breach the cap is reported and skipped.
    """
    print("stage 1: shape sweep, int4 group 64, batch 1")
    print(f"{'N x K':>15}  {'MB':>8}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}  {'rel L2':>9}  {'checked':>7}")
    generator = torch.Generator(device="cuda").manual_seed(0)

    for n_out, n_in in SHAPES:
        working_bytes = n_out * n_in // 2 + 4 * n_out * (n_in // GROUP_SIZE) + 4 * n_out * n_in
        label = f"{n_out} x {n_in}"
        if working_bytes > WORKING_CAP_BYTES:
            print(f"{label:>15}  skipped, {working_bytes / 1e9:.2f} GB over the {WORKING_CAP_BYTES / 1e9:.0f} GB cap")
            continue

        x = torch.randn(n_in, device="cuda", generator=generator)
        packed = torch.randint(0, 256, (n_out, n_in // 2), dtype=torch.uint8, device="cuda", generator=generator)
        scales = torch.rand((n_out, n_in // GROUP_SIZE), device="cuda", generator=generator) * 0.01 + 0.001
        out = torch.empty(n_out, device="cuda")

        checked = n_out * n_in <= VERIFY_MAX_ELEMENTS
        error = float("nan")
        if checked:
            int4_gemv_into(packed, scales, x, out)
            exact = torch.mv(dequantize_int4(packed, scales), x)
            torch.cuda.synchronize()
            error = relative_error(out, exact)
            del exact

        n_bytes = packed.numel() + scales.numel() * 4
        seconds = time_call(lambda: int4_gemv_into(packed, scales, x, out), WARMUP, ITERATIONS)
        gbs = bandwidth_gb_s(n_bytes, seconds)
        print(
            f"{label:>15}  {n_bytes / 1e6:>8.1f}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  "
            f"{fraction_of_peak(gbs):>6.1%}  {error:>9.2e}  {'yes' if checked else 'no':>7}"
        )
        del x, packed, scales, out
    print()


def precision_ladder(checkpoint) -> None:
    """Quantize three real matrices at each group size in GROUPS and print bytes with cosine."""
    print("stage 2: precision ladder on real matrices, group size against bytes and cosine")
    print(f"{'layer':>18}  {'group':>6}  {'MB':>7}  {'ms':>8}  {'cos':>9}  {'rel L2':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)

    for label, name in MATRICES[:3]:
        w = checkpoint.read(name).to(device="cuda", dtype=torch.float32)
        x = torch.randn(w.shape[1], device="cuda", generator=generator)
        exact = torch.mv(w, x)
        for group_size in GROUPS:
            padded, activations = pad_columns(w, x, group_size)
            packed, scales = pack_int4(padded, group_size=group_size)
            out = torch.empty(padded.shape[0], device="cuda")
            result = run_int4(packed, scales, activations, out, group_size)
            torch.cuda.synchronize()
            n_bytes = packed.numel() + scales.numel() * 4
            seconds = time_call(
                lambda p=packed, s=scales, a=activations, o=out, g=group_size: run_int4(p, s, a, o, g),
                WARMUP,
                ITERATIONS,
            )
            print(
                f"{label:>18}  {group_size:>6}  {n_bytes / 1e6:>7.2f}  {seconds * 1e3:>8.3f}  "
                f"{cosine(result, exact):>9.5f}  {relative_error(result, exact):>9.2e}"
            )
        print()
        del w, x, exact
    print()


def reference_norm(vector: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return the fp32 RMSNorm of `vector` [K], scaled by the learned `weight` [K]."""
    return vector * torch.rsqrt(vector.square().mean() + EPS) * weight


def stack_scale(checkpoint) -> None:
    """Run the 16 real expert layers three ways: projections, full layer, graphed layer.

    The projection chain multiplies quantized matrices only, so its activation scale drifts
    from the fp32 reference. The layer chain adds both norms, the attention value path and
    the residuals, which is the layer SmolVLA runs. The graphed chain replays those kernels
    as one submission, which removes the per-layer Python dispatch.
    """
    print("stage 3: the real expert stack, int8 attention and up, int4 down")
    print(f"{'layers':>7}  {'MB':>7}  {'proj ms':>8}  {'layer ms':>9}  {'graph ms':>9}  {'cos proj':>9}  {'cos layer':>10}  {'exact':>5}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    names = (
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    )
    source = []
    layers = []
    for index in range(EXPERT_LAYERS):
        prefix = f"model.vlm_with_expert.lm_expert.layers.{index}"
        wv, wo, up, down, input_norm, post_norm = (
            checkpoint.read(f"{prefix}.{name}").to(device="cuda", dtype=torch.float32) for name in names
        )
        attention = GqaValueBlock(wv, wo)
        prefix = None
        if attention.input_width != attention.hidden_width:
            prefix = torch.randn(attention.input_width, device="cuda", generator=generator)
        source.append((wv, wo, up, down, input_norm, post_norm))
        layers.append((attention, MlpBlock(up, down, down_bits=4), (input_norm, post_norm), prefix))

    hidden = layers[0][1].hidden_width
    x = torch.randn(hidden, device="cuda", generator=generator)
    buffers = (torch.empty(hidden, device="cuda"), torch.empty(hidden, device="cuda"))
    normed = torch.empty(hidden, device="cuda")
    delta = torch.empty(hidden, device="cuda")
    scratch = torch.empty(hidden, device="cuda")

    def proj_chain(depth: int) -> torch.Tensor:
        """Run the first `depth` MLP projection pairs, alternating two buffers."""
        buffers[0].copy_(x)
        for index in range(depth):
            layers[index][1].into(buffers[index % 2], buffers[(index + 1) % 2])
        return buffers[depth % 2]

    def layer_chain(depth: int) -> torch.Tensor:
        """Run attention then MLP for the first `depth` layers, norms and residuals included."""
        buffers[0].copy_(x)
        for index in range(depth):
            attention, mlp, (input_norm, post_norm), prefix = layers[index]
            current = buffers[index % 2]
            rmsnorm_into(current, input_norm, normed)
            attention.into(normed if prefix is None else prefix, scratch)
            add_into(current, scratch, delta)
            rmsnorm_into(delta, post_norm, normed)
            mlp.into(normed, scratch)
            add_into(delta, scratch, buffers[(index + 1) % 2])
        return buffers[depth % 2]

    proj_reference = x
    layer_reference = x
    for depth in range(1, EXPERT_LAYERS + 1):
        wv, wo, up, down, input_norm, post_norm = source[depth - 1]
        proj_reference = torch.mv(down, F.silu(torch.mv(up, proj_reference)))
        normalized = reference_norm(layer_reference, input_norm)
        value_input = normalized if layers[depth - 1][3] is None else layers[depth - 1][3]
        attention = torch.mv(wo, torch.mv(wv, value_input).index_select(0, layers[depth - 1][0].expand))
        layer_reference = layer_reference + attention
        normalized = reference_norm(layer_reference, post_norm)
        layer_reference = layer_reference + torch.mv(down, F.silu(torch.mv(up, normalized)))
        if depth not in STACKS:
            continue
        proj = proj_chain(depth).clone()
        eager = layer_chain(depth).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer_chain(depth)
        buffers[0].zero_()
        graph.replay()
        torch.cuda.synchronize()
        exact = torch.equal(buffers[depth % 2], eager)
        proj_seconds = time_call(lambda d=depth: proj_chain(d), WARMUP, ITERATIONS)
        layer_seconds = time_call(lambda d=depth: layer_chain(d), WARMUP, ITERATIONS)
        graph_seconds = time_call(graph.replay, WARMUP, ITERATIONS)
        n_bytes = sum(attention_block.bytes + mlp.bytes for attention_block, mlp, _, _ in layers[:depth])
        print(
            f"{depth:>7}  {n_bytes / 1e6:>7.2f}  {proj_seconds * 1e3:>8.3f}  {layer_seconds * 1e3:>9.3f}  "
            f"{graph_seconds * 1e3:>9.3f}  {cosine(proj, proj_reference):>9.5f}  "
            f"{cosine(eager, layer_reference):>10.5f}  {'yes' if exact else 'NO':>5}"
        )
    print("note: odd layers are cross attention, so their value path reads a fixed random prefix [320]")
    print()


def layer_launches(tokens: int) -> int:
    """Return the launches one layer issues at `tokens`: projections, mix, MLP, norms and adds.

    Above GEMM_MIN_TOKENS both projection pairs are one GEMM per matrix, so only the two
    norms and the two residual adds still scale with the token count.
    """
    attention = 5 if tokens >= GEMM_MIN_TOKENS else 1 + 4 * tokens
    return attention + 3 + 3 * tokens


def token_chain(
    checkpoint, tokens: int, depths: tuple[int, ...] = STACKS, warmup: int = WARMUP, iterations: int = ITERATIONS
) -> None:
    """Run the 16 real expert layers at `tokens` visible tokens.

    Two tokens make the softmax choose between two keys, up to eight make it choose among
    eight, so every token feeds every other one and a layer pays for q, k and v. Above
    GEMM_MIN_TOKENS the four attention    projections and both MLP projections are one GEMM each, so their launch count stops
    growing with the token count; only the norms and residual adds still run per token. The
    reference is the same chain in fp32 with the same weights, the same fixed
    cross-attention prefix and the same rotation. Attention is causal, so token i reads
    keys 0 to i, which is the schedule the expert runs. Each depth is also captured as one
    CUDA graph, checked exact against the eager run before its time is printed.
    """
    print(f"stage 4: {tokens} tokens through the expert stack, q and k included")
    if depths != STACKS:
        print(
            f"subset of depths {depths}, {warmup} warmup and {iterations} iterations "
            f"because a layer is {layer_launches(tokens)} launches"
        )
    print(f"{'layers':>7}  {'MB':>7}  {'eager ms':>9}  {'graph ms':>9}  {'cos':>9}  {'rel L2':>9}  {'exact':>5}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    names = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    )
    blocks = []
    for index in range(EXPERT_LAYERS):
        prefix = f"model.vlm_with_expert.lm_expert.layers.{index}"
        wq, wk, wv, wo, up, down, input_norm, post_norm = (
            checkpoint.read(f"{prefix}.{name}").to(device="cuda", dtype=torch.float32) for name in names
        )
        attention = GqaTokenBlock(wq, wk, wv, wo, tokens=tokens)
        other = None
        if attention.other_width != attention.hidden_width:
            other = torch.randn(tokens, attention.other_width, device="cuda", generator=generator)
        blocks.append((attention, MlpTokenBlock(up, down, down_bits=4, tokens=tokens), (input_norm, post_norm), other))
        del wq, wk, wv, wo, up, down

    hidden = blocks[0][1].hidden_width
    x = torch.randn(tokens, hidden, device="cuda", generator=generator)
    current = torch.empty(tokens, hidden, device="cuda")
    following = torch.empty(tokens, hidden, device="cuda")
    scratch = torch.empty(tokens, hidden, device="cuda")
    normed = torch.empty(tokens, hidden, device="cuda")
    delta = torch.empty(tokens, hidden, device="cuda")

    def chain(depth: int) -> torch.Tensor:
        """Run attention and MLP for the first `depth` layers at every token."""
        current.copy_(x)
        for index in range(depth):
            attention, mlp, (input_norm, post_norm), other = blocks[index]
            source = current if index % 2 == 0 else following
            target = following if index % 2 == 0 else current
            for token in range(tokens):
                rmsnorm_into(source[token], input_norm, normed[token])
            attention.into(normed, other, scratch)
            for token in range(tokens):
                add_norm_into(source[token], scratch[token], post_norm, delta[token], normed[token])
            mlp.into(normed, scratch)
            for token in range(tokens):
                add_into(delta[token], scratch[token], target[token])
        return following if depth % 2 else current

    references: dict[int, torch.Tensor] = {}
    reference = x.clone()
    for depth in range(1, EXPERT_LAYERS + 1):
        prefix = f"model.vlm_with_expert.lm_expert.layers.{depth - 1}"
        wq, wk, wv, wo, up, down, input_norm, post_norm = (
            checkpoint.read(f"{prefix}.{name}").to(device="cuda", dtype=torch.float32) for name in names
        )
        other = blocks[depth - 1][3]
        normalized = torch.stack([reference_norm(reference[token], input_norm) for token in range(tokens)])
        value_input = normalized if other is None else other
        context = gqa_mix(normalized @ wq.T, value_input @ wk.T, value_input @ wv.T)
        reference = reference + context @ wo.T
        normalized = torch.stack([reference_norm(reference[token], post_norm) for token in range(tokens)])
        reference = reference + F.silu(normalized @ up.T) @ down.T
        if depth in depths:
            references[depth] = reference.clone()
        del wq, wk, wv, wo, up, down

    for depth in depths:
        result = chain(depth).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain(depth)
        current.zero_()
        graph.replay()
        torch.cuda.synchronize()
        exact = torch.equal(following if depth % 2 else current, result)
        seconds = time_call(lambda d=depth: chain(d), warmup, iterations)
        graph_seconds = time_call(graph.replay, warmup, iterations)
        n_bytes = sum(attention.bytes + mlp.bytes for attention, mlp, _, _ in blocks[:depth])
        flat = result.reshape(-1)
        expected = references[depth].reshape(-1)
        print(
            f"{depth:>7}  {n_bytes / 1e6:>7.2f}  {seconds * 1e3:>9.3f}  {graph_seconds * 1e3:>9.3f}  "
            f"{cosine(flat, expected):>9.5f}  {relative_error(flat, expected):>9.2e}  {'yes' if exact else 'NO':>5}"
        )
    print(
        f"note: {layer_launches(tokens)} launches per layer, theta {THETA:.0f} with the Llama pairing, "
        "attention is causal"
    )
    print()


def main() -> int:
    """Run the scaling stages in order, smallest first, inside the memory cap."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | working cap {WORKING_CAP_BYTES / 1e9:.0f} GB per shape")
    print()
    shape_sweep()
    checkpoint = open_checkpoint(CHECKPOINT)
    precision_ladder(checkpoint)
    stack_scale(checkpoint)
    for tokens in TOKENS:
        token_chain(checkpoint, tokens)
    token_chain(checkpoint, CHUNK_TOKENS, depths=CHUNK_STACKS, warmup=CHUNK_WARMUP, iterations=CHUNK_ITERATIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

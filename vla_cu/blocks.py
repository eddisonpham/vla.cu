"""Attention and MLP blocks assembled from the packed kernels, at batch 1.

Concept: a block is a fixed set of projections plus one elementwise step, and every
projection is a packed matrix with its own scale layout. Holding the packed weights and
the scratch buffers in one object is what lets a benchmark, a CUDA graph and a future
model call the same code. `LayerBlock` puts the two norms and the residual pair around
those projections, and `GatedMlpBlock.tail_into` runs the whole tail in one launch.

Attention at one visible token reduces to the value path, so the block is v = Wv x and
out = Wo v. The three qkv matrices share x, which is what the grouped launch exploits.
At two tokens q and k decide the output, because the softmax has to choose how much of
each value to keep, so that path prices a real attention block.

Task: python -m vla_cu block, python -m vla_cu scale
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from kernels.triton.gqa_attention import gqa_attention_into
from kernels.triton.int4_gemm import int4_gemm_into
from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from kernels.triton.int8_gemm import int8_gemm_into
from kernels.triton.int8_gemv import int8_gemv_into
from kernels.triton.int8_grouped_gemv import int8_grouped_gemv_into
from kernels.triton.rmsnorm import add_into, rmsnorm_into
from vla_cu.cuda_coop import GROUP as COOP_GROUP, MAX_TOKENS as COOP_MAX_TOKENS, coop_gated_into
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8
from vla_cu.validate import require

BITS = (4, 8)
HEAD_DIM = 64
THETA = 100000.0
FUSED_MAX_TOKENS = 64
GEMM_MIN_TOKENS = 2


def _pack(w: Tensor, bits: int, group_size: int) -> tuple[Tensor, Tensor]:
    """Quantize one fp32 matrix at `bits`, using the format that width owns."""
    if bits == 4:
        return pack_int4(w, group_size=group_size)
    return quantize_int8(w)


def matrix_bytes(rows: int, cols: int, bits: int, group_size: int) -> int:
    """Return the stored bytes of one matrix at `bits`, scales included."""
    weights = rows * cols * bits // 8
    scales = 4 * rows * (cols // group_size) if bits == 4 else 4 * rows
    return weights + scales


def rotary_tables(
    tokens: int, head_dim: int = HEAD_DIM, theta: float = THETA, device: torch.device | str = "cuda"
) -> tuple[Tensor, Tensor]:
    """Return the cos and sin tables [tokens, head_dim // 2] for positions 0 to tokens - 1.

    The fused kernel and `rotary` read the same tables, so a rotation is defined once and
    the two attention paths differ only in where the multiply happens.
    """
    half = head_dim // 2
    frequencies = theta ** (-2.0 * torch.arange(half, device=device, dtype=torch.float32) / head_dim)
    angles = torch.arange(tokens, device=device, dtype=torch.float32)[:, None] * frequencies[None, :]
    return torch.cos(angles).contiguous(), torch.sin(angles).contiguous()


def rotary(x: Tensor, head_dim: int = HEAD_DIM, theta: float = THETA) -> Tensor:
    """Return `x` [tokens, heads * head_dim] with each row rotated to its own position.

    Row i is position i, and within a head element p turns with p + head_dim / 2 by
    i * theta ** (-2p / head_dim), the Llama pairing. theta 100000 and the half split
    pairing come from the SmolVLM2-500M text config, which the expert layers copy. Row 0
    is the identity rotation, which is why a one token block needs none of this. Only q
    and k are rotated, never v.
    """
    tokens = x.shape[0]
    heads = x.shape[1] // head_dim
    half = head_dim // 2
    cosine, sine = rotary_tables(tokens, head_dim, theta, x.device)
    pairs = x.reshape(tokens, heads, 2, half)
    first = pairs[:, :, 0]
    second = pairs[:, :, 1]
    rotated = torch.stack(
        (first * cosine[:, None] - second * sine[:, None], first * sine[:, None] + second * cosine[:, None]),
        dim=2,
    )
    return rotated.reshape(tokens, heads * head_dim)


def gqa_mix(q: Tensor, k: Tensor, v: Tensor, head_dim: int = HEAD_DIM, causal: bool = True) -> Tensor:
    """Return the attention context [tokens, heads * head_dim] for q, k and v.

    q is [tokens, heads * head_dim], k and v are [tokens, kv_heads * head_dim], with
    kv_heads dividing heads. q and k rotate to their row's position first, so a score
    depends on the relative position of the pair. Scores are [heads, tokens, tokens],
    q.k over sqrt(head_dim), with future keys set to -inf when `causal` so token i reads
    keys 0 to i and the chain runs the schedule the expert runs. Softmax runs over the
    key axis and each kv head serves `heads // kv_heads` query heads.
    """
    tokens = q.shape[0]
    heads = q.shape[1] // head_dim
    kv_heads = k.shape[1] // head_dim
    require(heads % kv_heads == 0, f"{heads} query heads do not group {kv_heads} kv heads")
    share = torch.arange(heads, device=q.device) // (heads // kv_heads)
    query = rotary(q, head_dim).reshape(tokens, heads, head_dim)
    key = rotary(k, head_dim).reshape(tokens, kv_heads, head_dim).index_select(1, share)
    value = v.reshape(tokens, kv_heads, head_dim).index_select(1, share)
    scores = torch.einsum("thd,shd->hts", query, key) * (1.0 / math.sqrt(head_dim))
    if causal:
        future = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(future[None], float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", weights, value).reshape(tokens, heads * head_dim)


class MlpBlock:
    """A two-layer MLP: int8 up, SiLU in place, down at int4 or int8.

    `up` is [intermediate, hidden], `down` is [hidden, intermediate], both fp32. The
    intermediate vector is owned by the block, so `into` is not reentrant.
    """

    def __init__(self, up: Tensor, down: Tensor, down_bits: int = 4, group_size: int = GROUP_SIZE) -> None:
        require(up.dim() == 2 and down.dim() == 2, f"expected 2D matrices, got {tuple(up.shape)} and {tuple(down.shape)}")
        require(up.shape[0] == down.shape[1] and up.shape[1] == down.shape[0], f"up {tuple(up.shape)} and down {tuple(down.shape)} do not compose")
        require(down_bits in BITS, f"expected 4 or 8 bits, got {down_bits}")
        intermediate, hidden = up.shape
        self.hidden_width = hidden
        self.intermediate = intermediate
        self.down_bits = down_bits
        self.group_size = group_size
        self.up, self.up_scales = quantize_int8(up)
        self.down, self.down_scales = _pack(down, down_bits, group_size)
        self.hidden = torch.empty(intermediate, device=up.device, dtype=torch.float32)
        self.bytes = matrix_bytes(intermediate, hidden, 8, group_size) + matrix_bytes(hidden, intermediate, down_bits, group_size)

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write down(silu(up @ x)) into `out`, reading `x` [hidden] and reusing `hidden`."""
        int8_gemv_into(self.up, self.up_scales, x, self.hidden)
        torch.nn.functional.silu(self.hidden, inplace=True)
        if self.down_bits == 4:
            int4_gemv_into(self.down, self.down_scales, self.hidden, out, group_size=self.group_size)
        else:
            int8_gemv_into(self.down, self.down_scales, self.hidden, out)


class MlpTokenBlock:
    """A two-layer MLP over a chunk of tokens: int8 up, SiLU in place, down at int4 or int8.

    `up` is [intermediate, hidden] and `down` is [hidden, intermediate], both fp32. Every
    projection is one GEMM over the whole chunk, so the weights are read once instead of
    once per token, and the intermediate buffer is [tokens, intermediate]. The block owns
    that buffer, so `into` is not reentrant.
    """

    def __init__(
        self,
        up: Tensor,
        down: Tensor,
        down_bits: int = 4,
        group_size: int = GROUP_SIZE,
        tokens: int = 2,
    ) -> None:
        require(up.dim() == 2 and down.dim() == 2, f"expected 2D matrices, got {tuple(up.shape)} and {tuple(down.shape)}")
        require(up.shape[0] == down.shape[1] and up.shape[1] == down.shape[0], f"up {tuple(up.shape)} and down {tuple(down.shape)} do not compose")
        require(down_bits in BITS, f"expected 4 or 8 bits, got {down_bits}")
        require(tokens > 0, f"token count {tokens} must be positive")
        intermediate, hidden = up.shape
        self.tokens = tokens
        self.hidden_width = hidden
        self.intermediate = intermediate
        self.down_bits = down_bits
        self.group_size = group_size
        self.up, self.up_scales = quantize_int8(up)
        self.down, self.down_scales = _pack(down, down_bits, group_size)
        self.hidden = torch.empty(tokens, intermediate, device=up.device, dtype=torch.float32)
        self.bytes = matrix_bytes(intermediate, hidden, 8, group_size) + matrix_bytes(hidden, intermediate, down_bits, group_size)

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write down(silu(up @ x)) into `out` for every row of `x` [tokens, hidden]."""
        require(x.shape == (self.tokens, self.hidden_width), f"expected x {(self.tokens, self.hidden_width)}, got {tuple(x.shape)}")
        int8_gemm_into(self.up, self.up_scales, x, self.hidden)
        torch.nn.functional.silu(self.hidden, inplace=True)
        if self.down_bits == 4:
            int4_gemm_into(self.down, self.down_scales, self.hidden, out, group_size=self.group_size)
        else:
            int8_gemm_into(self.down, self.down_scales, self.hidden, out)


class AttentionBlock:
    """q, k, v through one grouped launch, then o, at 4 or 8 bits.

    All four matrices are square [hidden, hidden]. At one visible token the softmax is
    the identity, so this is the value path and nothing else.
    """

    def __init__(self, wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor, bits: int = 8, group_size: int = GROUP_SIZE) -> None:
        require(bits in BITS, f"expected 4 or 8 bits, got {bits}")
        require(wq.dim() == 2 and wq.shape == wk.shape == wv.shape == wo.shape, f"expected four equal square matrices, got {tuple(wq.shape)}, {tuple(wk.shape)}, {tuple(wv.shape)}, {tuple(wo.shape)}")
        require(wq.shape[0] == wq.shape[1], f"expected square attention matrices, got {tuple(wq.shape)}")
        hidden = wq.shape[0]
        self.hidden_width = hidden
        self.bits = bits
        self.group_size = group_size
        self.qkv, self.qkv_scales = _pack(torch.cat((wq, wk, wv), dim=0), bits, group_size)
        self.o, self.o_scales = _pack(wo, bits, group_size)
        self.qkv_out = torch.empty(3 * hidden, device=wq.device, dtype=torch.float32)
        self.segment = torch.arange(3, device=wq.device, dtype=torch.int32).repeat_interleave(hidden)
        self.x_base = torch.zeros(3, device=wq.device, dtype=torch.int32)
        self.bytes = matrix_bytes(3 * hidden, hidden, bits, group_size) + matrix_bytes(hidden, hidden, bits, group_size)

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write Wo(Wv x) into `out`, reading `x` [hidden] and keeping the qkv vector."""
        if self.bits == 4:
            int4_grouped_gemv_into(self.qkv, self.qkv_scales, x, self.qkv_out, self.segment, self.x_base, group_size=self.group_size)
            int4_gemv_into(self.o, self.o_scales, self.qkv_out[2 * self.hidden_width :], out, group_size=self.group_size)
        else:
            int8_grouped_gemv_into(self.qkv, self.qkv_scales, x, self.qkv_out, self.segment, self.x_base)
            int8_gemv_into(self.o, self.o_scales, self.qkv_out[2 * self.hidden_width :], out)


class GqaValueBlock:
    """The value path of one grouped-query attention block at a single visible token.

    `wv` is [kv_heads * 64, in_width] and `wo` is [hidden, heads * 64]. With one token the
    softmax is 1 and the rotation is the identity, so q and k cancel out of the output
    exactly and are never computed; the block writes Wo(repeat(Wv x)), each kv head
    repeated `heads // kv_heads` times to match the o_proj input layout. A self-attention
    block reads the residual stream, a cross-attention block reads the other stream.
    """

    def __init__(self, wv: Tensor, wo: Tensor, head_dim: int = HEAD_DIM) -> None:
        require(wv.dim() == 2 and wo.dim() == 2, f"expected 2D matrices, got {tuple(wv.shape)} and {tuple(wo.shape)}")
        require(wo.shape[1] % wv.shape[0] == 0, f"o input width {wo.shape[1]} is not a multiple of v width {wv.shape[0]}")
        require(wv.shape[0] % head_dim == 0 and wo.shape[1] % head_dim == 0, "head_dim must divide both widths")
        self.hidden_width = wo.shape[0]
        self.input_width = wv.shape[1]
        self.groups = wo.shape[1] // wv.shape[0]
        self.v, self.v_scales = quantize_int8(wv)
        self.o, self.o_scales = quantize_int8(wo)
        heads = wo.shape[1] // head_dim
        kv_head_of = torch.arange(heads, device=wv.device) // self.groups
        self.expand = kv_head_of.repeat_interleave(head_dim) * head_dim + torch.arange(head_dim, device=wv.device).repeat(heads)
        self.v_out = torch.empty(wv.shape[0], device=wv.device, dtype=torch.float32)
        self.expanded = torch.empty(wo.shape[1], device=wv.device, dtype=torch.float32)
        self.bytes = matrix_bytes(wv.shape[0], wv.shape[1], 8, group_size=GROUP_SIZE) + matrix_bytes(
            wo.shape[0], wo.shape[1], 8, group_size=GROUP_SIZE
        )

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write Wo(Wv x) into `out`, reading `x` [in_width] and expanding the kv heads."""
        int8_gemv_into(self.v, self.v_scales, x, self.v_out)
        torch.index_select(self.v_out, 0, self.expand, out=self.expanded)
        int8_gemv_into(self.o, self.o_scales, self.expanded, out)


class GqaTokenBlock:
    """One grouped-query attention block at two or more tokens, q and k included.

    `wq` is [heads * 64, input_width], `wk` and `wv` are [kv_heads * 64, other_width] and
    `wo` is [hidden, heads * 64]. Above GEMM_MIN_TOKENS each projection is one int8 GEMM
    over the chunk, so the weights are read once instead of once per token; at one token
    the four GEMVs remain, because a GEMM would pad that row to 16. The score and value
    mix, rope included, is one fused Triton kernel up to FUSED_MAX_TOKENS and two
    [heads, tokens, tokens] einsums above it, because BLOCK_T stops fitting registers past
    a 64 token row tile (EXP-062, EXP-064). `into` reads `x` [tokens, input_width] and
    writes `out` [tokens, hidden], taking k and v from `other` [tokens, other_width] when
    it is given. `tokens` fixes the buffer shapes, so one block serves one token count and
    a graph captures that count.
    """

    def __init__(
        self,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
        head_dim: int = HEAD_DIM,
        tokens: int = 2,
        causal: bool = True,
    ) -> None:
        require(wq.dim() == 2 and wk.shape == wv.shape and wo.dim() == 2, f"expected 2D matrices, got {tuple(wq.shape)} and {tuple(wo.shape)}")
        require(wo.shape[1] == wq.shape[0], f"o input {wo.shape[1]} does not match q output {wq.shape[0]}")
        require(wq.shape[0] % head_dim == 0 and wk.shape[0] % head_dim == 0, "head_dim must divide both head counts")
        require(tokens > 0, f"token count {tokens} must be positive")
        self.tokens = tokens
        self.causal = causal
        self.hidden_width = wo.shape[0]
        self.input_width = wq.shape[1]
        self.other_width = wk.shape[1]
        self.head_dim = head_dim
        self.q, self.q_scales = quantize_int8(wq)
        self.k, self.k_scales = quantize_int8(wk)
        self.v, self.v_scales = quantize_int8(wv)
        self.o, self.o_scales = quantize_int8(wo)
        device = wq.device
        self.cosine, self.sine = rotary_tables(tokens, head_dim, device=device)
        self.q_out = torch.empty(tokens, wq.shape[0], device=device, dtype=torch.float32)
        self.k_out = torch.empty(tokens, wk.shape[0], device=device, dtype=torch.float32)
        self.v_out = torch.empty(tokens, wv.shape[0], device=device, dtype=torch.float32)
        self.context = torch.empty(tokens, wo.shape[1], device=device, dtype=torch.float32)
        self.bytes = sum(
            matrix_bytes(rows, cols, 8, GROUP_SIZE) for rows, cols in (wq.shape, wk.shape, wv.shape, wo.shape)
        )

    def into(self, x: Tensor, other: Tensor | None, out: Tensor) -> None:
        """Write the attention output into `out` [tokens, hidden], reading `x` and `other`."""
        require(x.shape[0] == self.tokens, f"block was built for {self.tokens} tokens, got {x.shape[0]}")
        source = x if other is None else other
        if self.tokens >= GEMM_MIN_TOKENS:
            int8_gemm_into(self.q, self.q_scales, x, self.q_out)
            int8_gemm_into(self.k, self.k_scales, source, self.k_out)
            int8_gemm_into(self.v, self.v_scales, source, self.v_out)
        else:
            for token in range(self.tokens):
                int8_gemv_into(self.q, self.q_scales, x[token], self.q_out[token])
                int8_gemv_into(self.k, self.k_scales, source[token], self.k_out[token])
                int8_gemv_into(self.v, self.v_scales, source[token], self.v_out[token])
        if self.tokens <= FUSED_MAX_TOKENS:
            gqa_attention_into(self.q_out, self.k_out, self.v_out, self.cosine, self.sine, self.context, self.head_dim, self.causal)
        else:
            self.context.copy_(gqa_mix(self.q_out, self.k_out, self.v_out, self.head_dim, self.causal))
        if self.tokens >= GEMM_MIN_TOKENS:
            int8_gemm_into(self.o, self.o_scales, self.context, out)
        else:
            for token in range(self.tokens):
                int8_gemv_into(self.o, self.o_scales, self.context[token], out[token])


class ActionBlock:
    """Attention then MLP, the projection stack of one action-expert layer."""

    def __init__(self, attention: AttentionBlock, mlp: MlpBlock) -> None:
        require(attention.hidden_width == mlp.hidden_width, f"attention width {attention.hidden_width} does not match MLP width {mlp.hidden_width}")
        self.attention = attention
        self.mlp = mlp
        self.hidden_width = attention.hidden_width
        self.intermediate = mlp.intermediate
        self.bytes = attention.bytes + mlp.bytes
        self.attention_out = torch.empty(attention.hidden_width, device=attention.qkv_out.device, dtype=torch.float32)

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write mlp(attention(x)) into `out`, reading `x` [hidden]."""
        self.attention.into(x, self.attention_out)
        self.mlp.into(self.attention_out, out)


class GatedMlpBlock:
    """A SwiGLU MLP: silu(gate @ x) * (up @ x), then down, at 4 or 8 bits.

    gate and up are [intermediate, hidden] and down is [hidden, intermediate], all fp32,
    which is the three-matrix MLP a SmolVLA layer has. One token reads and writes [hidden]
    through the GEMVs, a chunk reads and writes [tokens, hidden] through the GEMMs, and the
    block owns both intermediate buffers.
    """

    def __init__(
        self,
        gate: Tensor,
        up: Tensor,
        down: Tensor,
        down_bits: int = 4,
        group_size: int = GROUP_SIZE,
        tokens: int = 1,
    ) -> None:
        require(gate.dim() == 2 and up.shape == gate.shape, f"expected equal 2D gate and up, got {tuple(gate.shape)} and {tuple(up.shape)}")
        require(down.dim() == 2, f"expected a 2D down matrix, got {tuple(down.shape)}")
        require(down_bits in BITS, f"expected 4 or 8 bits, got {down_bits}")
        require(tokens > 0, f"token count {tokens} must be positive")
        intermediate, hidden = gate.shape
        require(tuple(down.shape) == (hidden, intermediate), f"down {tuple(down.shape)} does not undo gate {tuple(gate.shape)}")
        self.tokens = tokens
        self.hidden_width = hidden
        self.intermediate = intermediate
        self.down_bits = down_bits
        self.group_size = group_size
        self.gate, self.gate_scales = quantize_int8(gate)
        self.up, self.up_scales = quantize_int8(up)
        self.down, self.down_scales = _pack(down, down_bits, group_size)
        self.activated = torch.empty(tokens, intermediate, device=gate.device, dtype=torch.float32)
        self.up_out = torch.empty(tokens, intermediate, device=gate.device, dtype=torch.float32)
        self.bytes = (
            2 * matrix_bytes(intermediate, hidden, 8, group_size)
            + matrix_bytes(hidden, intermediate, down_bits, group_size)
        )

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write down(silu(gate x) * up x) into `out` for `x` [hidden] or [tokens, hidden]."""
        if self.tokens == 1:
            require(
                x.shape == (self.hidden_width,) and out.shape == (self.hidden_width,),
                f"expected [{self.hidden_width}], got {tuple(x.shape)} and {tuple(out.shape)}",
            )
            int8_gemv_into(self.gate, self.gate_scales, x, self.activated[0])
            int8_gemv_into(self.up, self.up_scales, x, self.up_out[0])
        else:
            require(
                x.shape == (self.tokens, self.hidden_width) and out.shape == (self.tokens, self.hidden_width),
                f"expected [{(self.tokens, self.hidden_width)}], got {tuple(x.shape)} and {tuple(out.shape)}",
            )
            int8_gemm_into(self.gate, self.gate_scales, x, self.activated)
            int8_gemm_into(self.up, self.up_scales, x, self.up_out)
        torch.nn.functional.silu(self.activated, inplace=True)
        self.activated.mul_(self.up_out)
        if self.tokens == 1:
            if self.down_bits == 4:
                int4_gemv_into(self.down, self.down_scales, self.activated[0], out, group_size=self.group_size)
            else:
                int8_gemv_into(self.down, self.down_scales, self.activated[0], out)
        elif self.down_bits == 4:
            int4_gemm_into(self.down, self.down_scales, self.activated, out, group_size=self.group_size)
        else:
            int8_gemm_into(self.down, self.down_scales, self.activated, out)

    def tail_into(
        self,
        x: Tensor,
        other: Tensor,
        weight: Tensor,
        added: Tensor,
        normed: Tensor,
        out: Tensor,
        blocks: int = 0,
    ) -> None:
        """Write the residual, norm and gated MLP tail with one cooperative launch.

        Computes the same result as `add_norm_into`, then `into`, then `add_into`, over
        [tokens, hidden] fp32 buffers, with `weight` the [hidden] RMSNorm scale. Needs an
        int4 down at group 64, hidden divisible by 16, intermediate by 64 and 1 to 64
        tokens; the kernel's act buffer is `self.activated`.
        """
        require(self.down_bits == 4, f"the fused tail needs an int4 down, got {self.down_bits} bits")
        require(self.group_size == COOP_GROUP, f"the fused tail needs group size {COOP_GROUP}, got {self.group_size}")
        require(self.tokens <= COOP_MAX_TOKENS, f"the fused tail holds {COOP_MAX_TOKENS} tokens, got {self.tokens}")
        require(self.hidden_width % 16 == 0, f"the fused tail needs hidden divisible by 16, got {self.hidden_width}")
        require(
            self.intermediate % COOP_GROUP == 0,
            f"the fused tail needs intermediate divisible by {COOP_GROUP}, got {self.intermediate}",
        )
        coop_gated_into(
            x,
            other,
            weight,
            self.gate,
            self.gate_scales,
            self.up,
            self.up_scales,
            self.down,
            self.down_scales,
            added,
            normed,
            self.activated,
            out,
            blocks,
        )


class LayerBlock:
    """One transformer layer: input norm, attention, residual, post norm, MLP, residual.

    `input_norm` and `post_norm` are [hidden] fp32 RMSNorm weights and `tokens` comes from
    the attention block. `into` reads `x` [tokens, hidden], takes the cross stream `other`
    when it is given, and owns the buffers between the stages, so it is not reentrant.
    """

    def __init__(self, attention: GqaTokenBlock, mlp: GatedMlpBlock | MlpTokenBlock | MlpBlock, input_norm: Tensor, post_norm: Tensor) -> None:
        hidden = attention.hidden_width
        require(attention.input_width == hidden, f"a layer block needs self attention, got input width {attention.input_width}")
        require(mlp.hidden_width == hidden, f"MLP width {mlp.hidden_width} does not match attention width {hidden}")
        for name, norm in (("input_norm", input_norm), ("post_norm", post_norm)):
            require(norm.shape == (hidden,), f"expected {name} [{hidden}], got {tuple(norm.shape)}")
            require(norm.dtype == torch.float32, f"expected fp32 {name}, got {norm.dtype}")
            require(norm.device == attention.q.device, f"expected {name} on {attention.q.device}, got {norm.device}")
        self.tokens = attention.tokens
        self.hidden_width = hidden
        self.attention = attention
        self.mlp = mlp
        self.input_norm = input_norm
        self.post_norm = post_norm
        device = attention.q.device
        self.normed = torch.empty(self.tokens, hidden, device=device, dtype=torch.float32)
        self.attention_out = torch.empty(self.tokens, hidden, device=device, dtype=torch.float32)
        self.residual = torch.empty(self.tokens, hidden, device=device, dtype=torch.float32)
        self.mlp_out = torch.empty(self.tokens, hidden, device=device, dtype=torch.float32)
        self.bytes = attention.bytes + mlp.bytes

    def into(self, x: Tensor, out: Tensor, other: Tensor | None = None) -> None:
        """Write one layer into `out`, reading `x` [tokens, hidden] and the cross stream `other`."""
        require(x.shape == (self.tokens, self.hidden_width), f"expected x {(self.tokens, self.hidden_width)}, got {tuple(x.shape)}")
        require(out.shape == x.shape, f"expected out {tuple(x.shape)}, got {tuple(out.shape)}")
        rmsnorm_into(x, self.input_norm, self.normed)
        self.attention.into(self.normed, other, self.attention_out)
        add_into(x, self.attention_out, self.residual)
        rmsnorm_into(self.residual, self.post_norm, self.normed)
        if self.tokens == 1:
            self.mlp.into(self.normed[0], self.mlp_out[0])
        else:
            self.mlp.into(self.normed, self.mlp_out)
        add_into(self.residual, self.mlp_out, out)

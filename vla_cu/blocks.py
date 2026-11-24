"""Attention and MLP blocks assembled from the packed kernels, at batch 1.

Concept: a block is a fixed set of projections plus one elementwise step, and every
projection is a packed matrix with its own scale layout. Holding the packed weights and
the scratch buffers in one object is what lets a benchmark, a CUDA graph and a future
model call the same code. Norms and residuals are not modelled, so a block here means
its projections only.

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
from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from kernels.triton.int8_gemv import int8_gemv_into
from kernels.triton.int8_grouped_gemv import int8_grouped_gemv_into
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8
from vla_cu.validate import require

BITS = (4, 8)
HEAD_DIM = 64
THETA = 100000.0
FUSED_MAX_TOKENS = 4


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
    frequencies = theta ** (-2.0 * torch.arange(half, device=x.device, dtype=torch.float32) / head_dim)
    angles = torch.arange(tokens, device=x.device, dtype=torch.float32)[:, None] * frequencies[None, :]
    cosine = torch.cos(angles)[:, None, :]
    sine = torch.sin(angles)[:, None, :]
    pairs = x.reshape(tokens, heads, 2, half)
    first = pairs[:, :, 0]
    second = pairs[:, :, 1]
    rotated = torch.stack((first * cosine - second * sine, first * sine + second * cosine), dim=2)
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
    `wo` is [hidden, heads * 64]. Every projection is one GEMV per token, so the block
    costs four launches per token; q and k rotate by their row's position, the softmax is
    causal so a token reads only itself and earlier tokens, and the score and value mix
    are either one fused Triton kernel or two [heads, tokens, tokens] einsums. The kernel
    wins up to FUSED_MAX_TOKENS and the einsums win above it, because the kernel reloads
    its key and value tiles per query head (EXP-062). `into`
    reads `x` [tokens, input_width] and writes `out` [tokens, hidden], taking k and v from
    `other` [tokens, other_width] when it is given. `tokens` fixes the buffer shapes, so
    one block serves one token count and a graph captures that count.
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
        for token in range(x.shape[0]):
            int8_gemv_into(self.q, self.q_scales, x[token], self.q_out[token])
            int8_gemv_into(self.k, self.k_scales, source[token], self.k_out[token])
            int8_gemv_into(self.v, self.v_scales, source[token], self.v_out[token])
        if self.tokens <= FUSED_MAX_TOKENS:
            query = rotary(self.q_out, self.head_dim)
            key = rotary(self.k_out, self.head_dim)
            gqa_attention_into(query, key, self.v_out, self.context, self.head_dim, self.causal)
        else:
            self.context.copy_(gqa_mix(self.q_out, self.k_out, self.v_out, self.head_dim, self.causal))
        for token in range(x.shape[0]):
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

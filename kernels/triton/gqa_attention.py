"""Causal grouped-query attention for a short token chunk, one program per query head.

Concept: with a few dozen tokens the whole key row fits in registers, so the score dot,
the causal mask, the softmax and the value mix can be one kernel instead of a handful of
torch nodes. Rotation stays in `vla_cu.blocks.rotary`, so this kernel is only the mix and
can be compared against the einsum reference on the same rotated inputs.

Task: python -m vla_cu scale
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from vla_cu.validate import check_power_of_two, require


@triton.jit
def gqa_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    T,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
) -> None:
    """Write the mixed context for one (query token, query head) pair.

    q: fp32 [T, HEADS * HEAD_DIM], k and v: fp32 [T, KV_HEADS * HEAD_DIM], out: fp32
    [T, HEADS * HEAD_DIM]. Query head h reads kv head h // (HEADS // KV_HEADS).
    Mapping: program (t, h) loads one query row, loads up to BLOCK_T key rows, scores
    them with a dot over HEAD_DIM, masks keys past t when causal, softmaxes in fp32 and
    writes the weighted sum of the value rows. Expected bottleneck: the key and value
    tiles are re-read by every query head, so this is L1 traffic, not DRAM; at 50 tokens
    it is launch bound before it is bandwidth bound.
    """
    token = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    kv_head = head // (HEADS // KV_HEADS)
    dims = tl.arange(0, HEAD_DIM)
    keys = tl.arange(0, BLOCK_T)
    key_mask = keys < T
    if CAUSAL:
        key_mask = key_mask & (keys <= token)

    query = tl.load(q_ptr + token * HEADS * HEAD_DIM + head * HEAD_DIM + dims)
    key_offsets = keys[:, None] * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dims[None, :]
    key_tile = tl.load(k_ptr + key_offsets, mask=key_mask[:, None], other=0.0)
    scores = tl.sum(key_tile * query[None, :], axis=1) * SCALE
    scores = tl.where(key_mask, scores, -1.0e30)
    weights = tl.exp(scores - tl.max(scores, axis=0))
    weights = weights / tl.sum(weights, axis=0)
    value_tile = tl.load(v_ptr + key_offsets, mask=key_mask[:, None], other=0.0)
    context = tl.sum(value_tile * weights[:, None], axis=0)
    tl.store(out_ptr + token * HEADS * HEAD_DIM + head * HEAD_DIM + dims, context)


def gqa_attention_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    head_dim: int = 64,
    causal: bool = True,
) -> None:
    """Write the attention context of q, k and v into `out` with one launch.

    q is [tokens, heads * head_dim], k and v are [tokens, kv_heads * head_dim], out is
    [tokens, heads * head_dim], all fp32 contiguous on one device. head_dim must be a
    power of two and kv_heads must divide heads. q and k are expected already rotated,
    so this computes the mix only.
    """
    tokens = q.shape[0]
    require(q.dim() == 2 and k.dim() == 2 and v.dim() == 2 and out.dim() == 2, "expected four 2D tensors")
    require(all(t.dtype == torch.float32 for t in (q, k, v, out)), "expected fp32 tensors")
    check_power_of_two("head_dim", head_dim)
    heads = q.shape[1] // head_dim
    kv_heads = k.shape[1] // head_dim
    require(heads * head_dim == q.shape[1] and kv_heads * head_dim == k.shape[1], "head_dim must divide both widths")
    require(heads % kv_heads == 0, f"{heads} query heads do not group {kv_heads} kv heads")
    require(out.shape == q.shape and v.shape == k.shape, f"expected out {tuple(q.shape)} and v {tuple(k.shape)}")
    if tokens == 0:
        return

    gqa_attention_kernel[(tokens, heads)](
        q,
        k,
        v,
        out,
        tokens,
        HEADS=heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_T=triton.next_power_of_2(tokens),
        SCALE=1.0 / math.sqrt(head_dim),
        CAUSAL=causal,
        num_warps=4,
    )

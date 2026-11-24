"""Causal grouped-query attention with rope fused, one program per token and kv head.

Concept: the query heads of one kv head share their key and value rows, so a program that
owns a whole group pays for those rows once instead of once per query head. Rotation is
linear in the key and the query, so it can run inside the same program from the same cos
and sine tables that `vla_cu.blocks.rotary` uses, which removes the two torch calls per
layer and keeps the two paths numerically identical.

Task: python -m vla_cu attncross
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
    cos_ptr,
    sin_ptr,
    out_ptr,
    T,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
) -> None:
    """Rotate the group's keys and values once, then mix every query head in the group.

    q: fp32 [T, HEADS * HEAD_DIM] unrotated, k and v: fp32 [T, KV_HEADS * HEAD_DIM],
    cos and sin: fp32 [T, HEAD_DIM // 2], out: fp32 [T, HEADS * HEAD_DIM].
    Mapping: program (token, kv_head) loads the key and value rows up to `token` once,
    rotates them with the per-row tables, then walks the HEADS // KV_HEADS query heads of
    that group, rotating one query row per head, scoring it against the shared key tile,
    softmaxing in fp32 and mixing the shared value tile.
    Expected bottleneck: register pressure and launch count. The key and value traffic
    per layer drops by the group size compared with one program per query head.
    """
    token = tl.program_id(axis=0)
    kv_head = tl.program_id(axis=1)
    share: tl.constexpr = HEADS // KV_HEADS
    half: tl.constexpr = HEAD_DIM // 2
    dims = tl.arange(0, half)
    keys = tl.arange(0, BLOCK_T)
    key_mask = keys < T
    if CAUSAL:
        key_mask = key_mask & (keys <= token)

    cos_query = tl.load(cos_ptr + token * half + dims)
    sin_query = tl.load(sin_ptr + token * half + dims)

    key_offsets = keys[:, None] * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + dims[None, :]
    key_first = tl.load(k_ptr + key_offsets, mask=key_mask[:, None], other=0.0)
    key_second = tl.load(k_ptr + key_offsets + half, mask=key_mask[:, None], other=0.0)
    cos_key = tl.load(cos_ptr + keys[:, None] * half + dims[None, :], mask=key_mask[:, None], other=0.0)
    sin_key = tl.load(sin_ptr + keys[:, None] * half + dims[None, :], mask=key_mask[:, None], other=0.0)
    rotated_first = key_first * cos_key - key_second * sin_key
    rotated_second = key_first * sin_key + key_second * cos_key

    value_dims = tl.arange(0, HEAD_DIM)
    value_offsets = keys[:, None] * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + value_dims[None, :]
    value = tl.load(v_ptr + value_offsets, mask=key_mask[:, None], other=0.0)

    group_base = token * HEADS * HEAD_DIM + kv_head * share * HEAD_DIM
    for index in tl.static_range(share):
        head_base = group_base + index * HEAD_DIM
        query_first = tl.load(q_ptr + head_base + dims)
        query_second = tl.load(q_ptr + head_base + half + dims)
        turned_first = query_first * cos_query - query_second * sin_query
        turned_second = query_first * sin_query + query_second * cos_query
        scores = tl.sum(rotated_first * turned_first[None, :], axis=1) + tl.sum(
            rotated_second * turned_second[None, :], axis=1
        )
        scores = tl.where(key_mask, scores * SCALE, -1.0e30)
        weights = tl.exp(scores - tl.max(scores, axis=0))
        weights = weights / tl.sum(weights, axis=0)
        mixed = tl.sum(value * weights[:, None], axis=0)
        tl.store(out_ptr + head_base + value_dims, mixed)


def gqa_attention_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    out: torch.Tensor,
    head_dim: int = 64,
    causal: bool = True,
) -> None:
    """Write rope and the attention context of q, k and v into `out` with one launch.

    q is [tokens, heads * head_dim] and k, v are [tokens, kv_heads * head_dim], all
    unrotated fp32 contiguous; cosine and sine are [tokens, head_dim // 2] fp32 from
    `vla_cu.blocks.rotary_tables`; out is [tokens, heads * head_dim] fp32. head_dim must
    be a power of two with an even count, and kv_heads must divide heads.
    """
    tokens = q.shape[0]
    require(q.dim() == 2 and k.dim() == 2 and v.dim() == 2 and out.dim() == 2, "expected four 2D tensors")
    require(
        all(t.dtype == torch.float32 for t in (q, k, v, cosine, sine, out)),
        "expected fp32 tensors",
    )
    require(all(t.is_contiguous() for t in (q, k, v, cosine, sine, out)), "expected contiguous tensors")
    check_power_of_two("head_dim", head_dim)
    require(head_dim % 2 == 0, f"head_dim {head_dim} must be even for the half split pairing")
    heads = q.shape[1] // head_dim
    kv_heads = k.shape[1] // head_dim
    require(heads * head_dim == q.shape[1] and kv_heads * head_dim == k.shape[1], "head_dim must divide both widths")
    require(heads % kv_heads == 0, f"{heads} query heads do not group {kv_heads} kv heads")
    require(out.shape == q.shape and v.shape == k.shape, f"expected out {tuple(q.shape)} and v {tuple(k.shape)}")
    require(
        cosine.shape == (tokens, head_dim // 2) and sine.shape == cosine.shape,
        f"expected tables {(tokens, head_dim // 2)}, got {tuple(cosine.shape)} and {tuple(sine.shape)}",
    )
    if tokens == 0:
        return

    gqa_attention_kernel[(tokens, kv_heads)](
        q,
        k,
        v,
        cosine,
        sine,
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

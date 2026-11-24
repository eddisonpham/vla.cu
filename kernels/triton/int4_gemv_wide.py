"""int4 GEMV over int32 words: wide weight loads and no shared memory staging.

Concept: reading the packed buffer as int32, so a lane owns four consecutive words, is
the cheapest way to ask Triton for 16 byte weight loads. At 4096 x 8192 it does not get
them: the listing still shows 4 and 8 byte weight loads and a shared memory reduction,
so this mapping is a measured alternative and not the default (EXP-052).

Task: python -m vla_cu plateau
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.quant import GROUP_SIZE
from vla_cu.validate import check_int4_gemv, check_power_of_two, require

BLOCK_K = 1024
WARPS = 1
BYTES_PER_WORD = 4
NIBBLES_PER_WORD = 8


@triton.jit
def int4_gemv_wide_kernel(
    words_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    N,
    K,
    GROUP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    W_EVICT: tl.constexpr,
    X_EVICT: tl.constexpr,
) -> None:
    """Dot one packed int4 row with one fp32 vector, reading the row as int32 words.

    words: int32 [N, K // 8], scales: fp32 or fp16 [N, K // GROUP], x: fp32 [K], out: fp32 [N].
    Mapping: program r owns output r and walks K in BLOCK_K element steps. The weights of
    a step are one flat run of BLOCK_K // 8 words, so a lane owns four consecutive words
    and the load can reach 16 bytes, then the run is reshaped to [groups, words per
    group] and broadcast against a shift vector to unpack eight nibbles per word. The
    group scale multiplies the group's finished sum. The reshape is what keeps the
    weights narrow in practice: it drives the layout Triton picks for the tile.
    Expected bottleneck: memory, one weight stream plus one cached activation stream.
    """
    row = tl.program_id(axis=0)
    n_groups = K // GROUP
    groups_per_block: tl.constexpr = BLOCK_K // GROUP
    words_per_group: tl.constexpr = GROUP // 8
    block_words: tl.constexpr = BLOCK_K // 8
    flat_lanes = tl.arange(0, block_words)
    group_lanes = tl.arange(0, groups_per_block)
    word_lanes = tl.arange(0, words_per_group)
    nibble_lanes = tl.arange(0, 8)
    shifts = nibble_lanes * 4
    acc = 0.0

    for k in range(0, K, BLOCK_K):
        first_group = k // GROUP
        group_mask = first_group + group_lanes < n_groups
        flat = tl.load(
            words_ptr + row * (K // 8) + k // 8 + flat_lanes,
            mask=k // 8 + flat_lanes < K // 8,
            other=0,
            eviction_policy=W_EVICT,
        )
        words = tl.reshape(flat, (groups_per_block, words_per_group))
        word_mask = group_mask[:, None, None]
        activations = tl.load(
            x_ptr + k + group_lanes[:, None, None] * GROUP + word_lanes[None, :, None] * 8 + nibble_lanes[None, None, :],
            mask=word_mask,
            other=0.0,
            eviction_policy=X_EVICT,
        )
        codes = ((words[:, :, None] >> shifts[None, None, :]) & 0x0F).to(tl.float32) - 8.0
        products = codes * activations
        group_sums = tl.sum(tl.sum(products, axis=2), axis=1)

        scale = tl.load(
            scale_ptr + row * n_groups + first_group + group_lanes,
            mask=group_mask,
            other=0.0,
            eviction_policy=W_EVICT,
        ).to(tl.float32)
        acc += tl.sum(group_sums * scale, axis=0)

    tl.store(out_ptr + row, acc)


def int4_gemv_wide_into(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    warps: int = WARPS,
    hints: bool = True,
) -> None:
    """Write (packed weights dequantized) @ x into `out`, reading the row as int32.

    Assumptions: as `int4_gemv_into`, plus K divisible by 32 so every row start is 16
    byte aligned, and group_size a multiple of 8 so no word straddles a group.
    """
    n_rows, n_cols, _ = check_int4_gemv(packed, scales, x, out, group_size)
    check_power_of_two("block size", block_k)
    require(block_k % group_size == 0, f"block size {block_k} must be a multiple of group size {group_size}")
    require(group_size % NIBBLES_PER_WORD == 0, f"group size {group_size} must be a multiple of {NIBBLES_PER_WORD}")
    require(n_cols % (2 * BYTES_PER_WORD * NIBBLES_PER_WORD) == 0, f"K {n_cols} must be a multiple of 32 for 16 byte row starts")
    if n_rows == 0:
        return

    int4_gemv_wide_kernel[(n_rows,)](
        packed.view(torch.int32),
        scales,
        x,
        out,
        n_rows,
        n_cols,
        GROUP=group_size,
        BLOCK_K=block_k,
        num_warps=warps,
        W_EVICT="evict_first" if hints else "",
        X_EVICT="evict_last" if hints else "",
    )


def int4_gemv_wide(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    warps: int = WARPS,
    hints: bool = True,
) -> torch.Tensor:
    """Allocate the output and return the wide-load dequantized GEMV."""
    out = torch.empty(packed.shape[0], device=packed.device, dtype=torch.float32)
    int4_gemv_wide_into(packed, scales, x, out, group_size=group_size, block_k=block_k, warps=warps, hints=hints)
    return out

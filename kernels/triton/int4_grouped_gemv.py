"""Grouped fused int4 GEMV: every matrix of one layer in a single launch.

Concept: at batch 1 a transformer layer issues one launch per projection, and q, k and
v all read the same activation vector. Concatenating their packed rows into one buffer
turns three launches into one and reads the activation vector once instead of three
times. Scales and outputs stay per matrix, so the layout is unchanged and only the row
range grows.

The assumption that makes the activation vector shared is that each matrix's row count
is a multiple of the row tile. Rows are laid out segment by segment, so a program never
straddles two matrices and its activation base is one scalar load.

Task: python -m vla_cu grouped
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.quant import GROUP_SIZE

BLOCK_K = 1024
ROWS_PER_PROGRAM = 4


@triton.jit
def int4_grouped_gemv_kernel(
    packed_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    segment_ptr,
    x_base_ptr,
    n_rows,
    K,
    GROUP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    W_EVICT: tl.constexpr,
    X_EVICT: tl.constexpr,
) -> None:
    """One int4 GEMV launch over several matrices, each with its own activation base.

    packed: uint8 [sum(Ni), K // 2], scales: fp32 [sum(Ni), K // GROUP], x: fp32 [X],
    out: fp32 [sum(Ni)], segment: int32 [sum(Ni)] matrix id per row, x_base: int32 [S]
    element offset into x per matrix.
    Mapping: identical to the single-matrix kernel, with the row index offset by the
    program's start row and the activation base read once per program from the segment
    table, because rows are concatenated in whole multiples of ROWS.
    Expected bottleneck: memory, weights plus one activation stream per matrix.
    """
    n_groups = K // GROUP
    groups_per_block: tl.constexpr = BLOCK_K // GROUP
    first_row = tl.program_id(axis=0) * ROWS
    rows = first_row + tl.arange(0, ROWS)
    row_mask = rows < n_rows
    segment = tl.load(segment_ptr + first_row)
    x_base = tl.load(x_base_ptr + segment)
    group_lanes = tl.arange(0, groups_per_block)
    byte_lanes = tl.arange(0, GROUP // 2)
    row_bytes = rows[:, None, None] * (K // 2)
    row_scales = rows[:, None] * n_groups
    acc = tl.zeros((ROWS, groups_per_block), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        group_ids = k // GROUP + group_lanes
        group_mask = group_ids < n_groups
        byte_offsets = k // 2 + group_lanes[:, None] * (GROUP // 2) + byte_lanes[None, :]
        weights_mask = row_mask[:, None, None] & group_mask[None, :, None]

        packed = tl.load(
            packed_ptr + row_bytes + byte_offsets[None, :, :],
            mask=weights_mask,
            other=0,
            eviction_policy=W_EVICT,
        ).to(tl.int32)
        low = (packed & 0x0F).to(tl.float32) - 8.0
        high = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0

        activations_low = tl.load(
            x_ptr + x_base + 2 * byte_offsets, mask=group_mask[:, None], other=0.0, eviction_policy=X_EVICT
        )
        activations_high = tl.load(
            x_ptr + x_base + 2 * byte_offsets + 1, mask=group_mask[:, None], other=0.0, eviction_policy=X_EVICT
        )
        products = low * activations_low[None, :, :] + high * activations_high[None, :, :]

        scale = tl.load(
            scale_ptr + row_scales + group_ids[None, :],
            mask=row_mask[:, None] & group_mask[None, :],
            other=0.0,
            eviction_policy=W_EVICT,
        ).to(tl.float32)
        acc += tl.sum(products, axis=2) * scale

    tl.store(out_ptr + rows, tl.sum(acc, axis=1), mask=row_mask)


def int4_grouped_gemv_into(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    segment: torch.Tensor,
    x_base: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    rows: int = ROWS_PER_PROGRAM,
) -> None:
    """Write the dequantized GEMV of several concatenated matrices into one output.

    Assumptions: packed uint8 [R, K // 2], scales fp32 [R, K // group_size], x fp32
    [X] holding every matrix's activation vector back to back, out fp32 [R], segment
    int32 [R], x_base int32 [S]. Every matrix's row count must be a multiple of `rows`
    and K must be divisible by `block_k` and by group_size.
    """
    total_rows = packed.shape[0]
    if total_rows == 0:
        return
    grid = (triton.cdiv(total_rows, rows),)
    int4_grouped_gemv_kernel[grid](
        packed,
        scales,
        x,
        out,
        segment,
        x_base,
        total_rows,
        packed.shape[1] * 2,
        GROUP=group_size,
        BLOCK_K=block_k,
        ROWS=rows,
        W_EVICT="evict_first",
        X_EVICT="evict_last",
    )

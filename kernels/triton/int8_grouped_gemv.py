"""Grouped int8 GEMV: every int8 projection of one layer in a single launch.

Concept: the int8 path has one scale per output row and no group structure, so the only
thing a grouped launch has to select is each matrix's activation base. A segment table
does that with one int32 load per program, and projections that share an activation
vector stop paying one launch each for it.

Rows are laid out matrix by matrix, so a program owns one row and never straddles two
matrices. There is no split-K here: a single row per program is the mapping the int8
kernel already uses, and its grid fills this GPU above 26 rows.

Task: python -m vla_cu attn
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.validate import check_int8_grouped, check_power_of_two

BLOCK_K = 1024


@triton.jit
def int8_grouped_gemv_kernel(
    w_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    segment_ptr,
    x_base_ptr,
    K,
    BLOCK_K: tl.constexpr,
) -> None:
    """Dot one int8 row with that row's matrix's activation vector, then apply the row scale.

    w: int8 [R, K] row-major, scales: fp32 [R], x: fp32 [X], out: fp32 [R],
    segment: int32 [R], x_base: int32 [S].
    Mapping: program i owns output row i, reads its matrix id once, and walks K in
    BLOCK_K element steps. Every row's bytes are contiguous, so the loads coalesce.
    Expected bottleneck: memory, weights only, R * K bytes, plus one activation stream
    per matrix rather than one per row.
    """
    row = tl.program_id(axis=0)
    segment = tl.load(segment_ptr + row)
    x_base = tl.load(x_base_ptr + segment)
    cols = tl.arange(0, BLOCK_K)
    row_start = w_ptr + row * K
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offsets = k + cols
        mask = offsets < K
        weights = tl.load(row_start + offsets, mask=mask, other=0).to(tl.float32)
        activations = tl.load(x_ptr + x_base + offsets, mask=mask, other=0.0)
        acc += weights * activations

    scale = tl.load(scale_ptr + row)
    tl.store(out_ptr + row, tl.sum(acc, axis=0) * scale)


def int8_grouped_gemv_into(
    w: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    segment: torch.Tensor,
    x_base: torch.Tensor,
    block_k: int = BLOCK_K,
) -> None:
    """Write the dequantized GEMV of several concatenated int8 matrices into one output.

    Assumptions: w int8 [R, K], scales fp32 [R], x fp32 [X] holding every matrix's
    activation vector back to back, out fp32 [R], segment int32 [R], x_base int32 [S],
    all contiguous on one CUDA device. X must be at least K.
    """
    total_rows, n_cols = check_int8_grouped(w, scales, x, out, segment, x_base)
    check_power_of_two("block size", block_k)
    if total_rows == 0:
        return
    int8_grouped_gemv_kernel[(total_rows,)](w, scales, x, out, segment, x_base, n_cols, BLOCK_K=block_k)

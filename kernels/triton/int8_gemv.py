"""int8 GEMV with one scale per output row, dequantized inside the kernel.

Concept: at batch 1 a linear layer is a matrix-vector product, so every weight is
used once and the kernel is memory-bound. Weights stay int8 in memory and are
converted to fp32 in registers, so DRAM only ever sees the small form. The row
scale is constant along the reduction, so it multiplies the finished dot product
once instead of scaling every term.

Two mappings. The default gives each output row to one program. The split-K variant
gives one row to SPLIT programs that combine partial sums with atomics, which is the
fix for layers with fewer rows than the GPU has SMs.

Task: python -m vla_cu gemv, python -m vla_cu tune
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.validate import check_int8_gemv, check_power_of_two, require

BLOCK_K = 1024


@triton.jit
def int8_gemv_kernel(
    w_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    K,
    BLOCK_K: tl.constexpr,
) -> None:
    """Dot one int8 weight row with an fp32 vector, then apply that row's scale.

    w: int8 [N, K] row-major, scales: fp32 [N], x: fp32 [K], out: fp32 [N].
    Mapping: program i owns output i and walks its row in BLOCK_K element steps,
    accumulating into registers that are reduced once at the end. Row i's bytes are
    contiguous, so the loads coalesce, and every program re-reads x, which is K
    floats and stays in cache. Expected bottleneck: memory, weights only, N*K bytes.
    """
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_K)
    row_start = w_ptr + row * K
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offsets = k + cols
        mask = offsets < K
        weights = tl.load(row_start + offsets, mask=mask, other=0).to(tl.float32)
        activations = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += weights * activations

    scale = tl.load(scale_ptr + row)
    tl.store(out_ptr + row, tl.sum(acc, axis=0) * scale)


@triton.jit
def int8_gemv_split_kernel(
    w_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    K,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
) -> None:
    """Split one row across SPLIT programs and combine partial dots with atomics.

    Mapping: program (row, part) handles every SPLIT-th block of BLOCK_K elements
    starting at part * BLOCK_K, so the parts interleave and stay balanced even when K
    is not a multiple of SPLIT * BLOCK_K. Each part adds its scaled partial sum to
    out[row] atomically, so `out` must be zeroed before the launch. Atomic order
    varies between runs, so the last few bits can differ from the single program
    version. Expected bottleneck: memory, until the row count fills the SMs.
    """
    row = tl.program_id(axis=0)
    part = tl.program_id(axis=1)
    cols = tl.arange(0, BLOCK_K)
    row_start = w_ptr + row * K
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for k in range(part * BLOCK_K, K, SPLIT * BLOCK_K):
        offsets = k + cols
        mask = offsets < K
        weights = tl.load(row_start + offsets, mask=mask, other=0).to(tl.float32)
        activations = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += weights * activations

    scale = tl.load(scale_ptr + row)
    tl.atomic_add(out_ptr + row, tl.sum(acc, axis=0) * scale)


def int8_gemv_into(
    w: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    block_k: int = BLOCK_K,
    split: int = 1,
) -> None:
    """Write scales * (w @ x) into `out`, without allocating.

    Assumptions: w is int8 [N, K], scales is fp32 [N], x is fp32 [K], out is fp32
    [N], all contiguous on one CUDA device. K may be any length. `block_k` must be a
    power of two and `split` at least 1; split above 1 zeroes `out` first because the
    partial sums are combined with atomics.
    """
    n_rows, n_cols = check_int8_gemv(w, scales, x, out)
    check_power_of_two("block size", block_k)
    require(split >= 1, f"expected split >= 1, got {split}")
    if n_rows == 0:
        return

    if split == 1:
        int8_gemv_kernel[(n_rows,)](w, scales, x, out, n_cols, BLOCK_K=block_k)
        return

    out.zero_()
    int8_gemv_split_kernel[(n_rows, split)](w, scales, x, out, n_cols, BLOCK_K=block_k, SPLIT=split)


def int8_gemv(
    w: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    block_k: int = BLOCK_K,
    split: int = 1,
) -> torch.Tensor:
    """Allocate the output and return scales * (w @ x)."""
    out = torch.empty(w.shape[0], device=w.device, dtype=torch.float32)
    int8_gemv_into(w, scales, x, out, block_k=block_k, split=split)
    return out

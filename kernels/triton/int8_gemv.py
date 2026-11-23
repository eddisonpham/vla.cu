"""int8 GEMV with one scale per output row, dequantized inside the kernel.

Concept: at batch 1 a linear layer is a matrix-vector product, so every weight is
used once and the kernel is memory-bound. Weights stay int8 in memory and are
converted to fp32 in registers, so DRAM only ever sees the small form. The scale is
constant along the reduction, so it multiplies the finished dot product once
instead of scaling every term.

Task: python -m vla_cu gemv
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_K = 1024


@triton.jit
def int8_gemv_kernel(
    w_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Dot one int8 weight row with a fp32 vector, then apply that row's scale.

    w: int8 [N, K] row-major, scales: fp32 [N], x: fp32 [K], out: fp32 [N].
    Mapping: program i owns output i and walks its row in BLOCK_K element steps,
    accumulating into a register vector that is reduced once at the end. Row i's
    bytes are contiguous, so the loads coalesce. Every program re-reads all of x,
    which is K floats and stays resident in cache after the first rows. Expected
    bottleneck: memory, weights only, N*K bytes plus one scale load per program.
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


def int8_gemv_into(
    w: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Write scales * (w @ x) into `out`, without allocating.

    Assumptions: w is int8 [N, K] contiguous, scales is fp32 [N] contiguous, x is
    fp32 [K] contiguous, out is fp32 [N] contiguous, all on the CUDA device. K may
    be any positive length, not only a multiple of BLOCK_K.
    """
    if w.dim() != 2 or scales.dim() != 1 or x.dim() != 1:
        raise ValueError(f"expected w [N, K], scales [N], x [K]; got {tuple(w.shape)}, {tuple(scales.shape)}, {tuple(x.shape)}")
    if w.shape != (scales.shape[0], x.shape[0]) or out.shape != scales.shape:
        raise ValueError(f"shape mismatch: w {tuple(w.shape)}, scales {tuple(scales.shape)}, x {tuple(x.shape)}, out {tuple(out.shape)}")
    if w.dtype != torch.int8:
        raise ValueError(f"expected int8 weights, got {w.dtype}")
    if scales.dtype != torch.float32 or x.dtype != torch.float32 or out.dtype != torch.float32:
        raise ValueError(f"expected fp32 scales, x and out; got {scales.dtype}, {x.dtype}, {out.dtype}")
    if not (w.is_contiguous() and scales.is_contiguous() and x.is_contiguous() and out.is_contiguous()):
        raise ValueError("expected contiguous tensors")
    if w.device.type != "cuda" or scales.device != w.device or x.device != w.device or out.device != w.device:
        raise ValueError("expected all tensors on the same CUDA device")

    n_rows, n_cols = w.shape
    if n_rows == 0:
        return

    int8_gemv_kernel[(n_rows,)](w, scales, x, out, K=n_cols, BLOCK_K=BLOCK_K)


def int8_gemv(w: torch.Tensor, scales: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Allocate the output and return scales * (w @ x)."""
    out = torch.empty(w.shape[0], device=w.device, dtype=torch.float32)
    int8_gemv_into(w, scales, x, out)
    return out

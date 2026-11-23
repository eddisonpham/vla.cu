"""Fused 4-bit dequantize and GEMV with one scale per group of 64 weights.

Concept: the weights arrive as packed nibbles and never become fp32 in memory. The
kernel reads one byte, turns it into two weights, multiplies each by its activation
and the scale of its group, and accumulates. Weights therefore stream at half a byte
per parameter, which is the entire point of the project at batch 2 elements per byte.

Layout matches `vla_cu.quant`: element 2i is the low nibble, element 2i+1 the high
nibble, value is (nibble - 8) * group_scale.

Task: python -m vla_cu matrix
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.quant import GROUP_SIZE

BLOCK_K = 1024
ROWS_PER_PROGRAM = 4


@triton.jit
def int4_gemv_kernel(
    packed_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    N,
    K,
    GROUP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
) -> None:
    """Dequantize packed int4 rows in registers and dot them with one fp32 vector.

    packed: uint8 [N, K // 2], scales: fp32 [N, K // GROUP], x: fp32 [K], out: fp32 [N].
    Mapping: program p owns outputs p*ROWS to p*ROWS+ROWS-1 and walks K in BLOCK_K
    element steps. Each step loads BLOCK_K // 2 bytes per row and splits every byte
    into its low and high nibble, so one load feeds two weights; a pair of lanes no
    longer fetches the same byte twice. The activation pair is loaded once per step
    and reused across all ROWS rows, which is the point of the tiling: every program
    would otherwise re-read the whole vector and make it the dominant traffic.
    The block is a whole number of groups, so both nibbles of a byte share a scale.
    Expected bottleneck: memory, weights and one activation stream, provided ROWS is
    large enough to amortize the activation; too large and accumulators spill.
    """
    rows = tl.program_id(axis=0) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < N
    lanes = tl.arange(0, BLOCK_K // 2)
    n_groups = K // GROUP
    row_bytes = rows[:, None] * (K // 2)
    row_scales = rows[:, None] * n_groups
    acc = tl.zeros((ROWS, BLOCK_K // 2), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        byte_offsets = k // 2 + lanes
        byte_mask = byte_offsets < K // 2
        mask = row_mask[:, None] & byte_mask[None, :]
        packed = tl.load(packed_ptr + row_bytes + byte_offsets[None, :], mask=mask, other=0).to(tl.int32)
        low = (packed & 0x0F).to(tl.float32) - 8.0
        high = ((packed >> 4) & 0x0F).to(tl.float32) - 8.0
        scale = tl.load(scale_ptr + row_scales + ((2 * byte_offsets) // GROUP)[None, :], mask=mask, other=0.0)
        activations_low = tl.load(x_ptr + 2 * byte_offsets, mask=byte_mask, other=0.0)
        activations_high = tl.load(x_ptr + 2 * byte_offsets + 1, mask=byte_mask, other=0.0)
        acc += (low * activations_low[None, :] + high * activations_high[None, :]) * scale

    tl.store(out_ptr + rows, tl.sum(acc, axis=1), mask=row_mask)


def int4_gemv_into(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    rows: int = ROWS_PER_PROGRAM,
) -> None:
    """Write (packed weights dequantized) @ x into `out`, no fp32 weights in memory.

    Assumptions: packed is uint8 [N, K // 2] contiguous, scales is fp32
    [N, K // group_size] contiguous, x is fp32 [K] contiguous, out is fp32 [N]
    contiguous, all on one CUDA device. K must be divisible by group_size, and
    block_k must be a power of two that is a multiple of group_size.
    """
    if packed.dim() != 2 or scales.dim() != 2 or x.dim() != 1 or out.dim() != 1:
        raise ValueError(f"expected packed [N, K // 2], scales [N, K // G], x [K], out [N]; got {tuple(packed.shape)}, {tuple(scales.shape)}, {tuple(x.shape)}, {tuple(out.shape)}")
    if packed.dtype != torch.uint8:
        raise ValueError(f"expected uint8 packed weights, got {packed.dtype}")
    if scales.dtype != torch.float32 or x.dtype != torch.float32 or out.dtype != torch.float32:
        raise ValueError(f"expected fp32 scales, x and out; got {scales.dtype}, {x.dtype}, {out.dtype}")

    n_rows, n_bytes = packed.shape
    n_cols = n_bytes * 2
    if x.shape[0] != n_cols or out.shape[0] != n_rows:
        raise ValueError(f"shape mismatch: packed {tuple(packed.shape)} implies K {n_cols}, got x {tuple(x.shape)} and out {tuple(out.shape)}")
    if group_size <= 0 or n_cols % group_size:
        raise ValueError(f"K {n_cols} is not divisible by group size {group_size}")
    if scales.shape != (n_rows, n_cols // group_size):
        raise ValueError(f"expected scales {(n_rows, n_cols // group_size)}, got {tuple(scales.shape)}")
    if block_k % group_size:
        raise ValueError(f"block size {block_k} must be a multiple of group size {group_size}")
    if rows < 1 or rows & (rows - 1):
        raise ValueError(f"expected a power of two row tile, got {rows}")
    if not (packed.is_contiguous() and scales.is_contiguous() and x.is_contiguous() and out.is_contiguous()):
        raise ValueError("expected contiguous tensors")
    if packed.device.type != "cuda" or scales.device != packed.device or x.device != packed.device or out.device != packed.device:
        raise ValueError("expected all tensors on the same CUDA device")
    if n_rows == 0:
        return

    grid = (triton.cdiv(n_rows, rows),)
    int4_gemv_kernel[grid](packed, scales, x, out, n_rows, n_cols, GROUP=group_size, BLOCK_K=block_k, ROWS=rows)


def int4_gemv(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    rows: int = ROWS_PER_PROGRAM,
) -> torch.Tensor:
    """Allocate the output and return the dequantized GEMV."""
    out = torch.empty(packed.shape[0], device=packed.device, dtype=torch.float32)
    int4_gemv_into(packed, scales, x, out, group_size=group_size, block_k=block_k, rows=rows)
    return out

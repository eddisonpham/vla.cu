"""Fused 4-bit dequantize and GEMV with one scale per group of 64 weights.

Concept: the weights arrive as packed nibbles and never become fp32 in memory. The
kernel reads one byte, turns it into two weights, multiplies each by its activation,
and accumulates. Weights therefore stream at half a byte per parameter, which is the
entire point of the project at two elements per byte.

The group structure drives the loop shape. A group is 64 consecutive weights, which
is 32 packed bytes, so the kernel reduces a whole group before it multiplies by that
group's scale. That gives one scale load and one multiply per 64 weights instead of
one per weight, and it shrinks the accumulator by the group size, which is what
allows a wide row tile.

Layout matches `vla_cu.quant`: element 2i is the low nibble, element 2i+1 the high
nibble, value is (nibble - 8) * group_scale. Scales may be fp32 or fp16.

Eviction hints are on by default: each packed byte is read once, so it is marked
evict_first and kept out of the way, while the activation vector is reused by every
program and is marked evict_last to stay resident.

Task: python -m vla_cu matrix
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.quant import GROUP_SIZE

BLOCK_K = 1024
ROWS_PER_PROGRAM = 4
SCALE_DTYPES = (torch.float32, torch.float16)


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
    W_EVICT: tl.constexpr,
    X_EVICT: tl.constexpr,
) -> None:
    """Dequantize packed int4 rows in registers and dot them with one fp32 vector.

    packed: uint8 [N, K // 2], scales: fp32 or fp16 [N, K // GROUP], x: fp32 [K], out: fp32 [N].
    Mapping: program p owns outputs p*ROWS to p*ROWS+ROWS-1 and walks K in BLOCK_K
    element steps. A step covers BLOCK_K // GROUP whole groups, and the tile is shaped
    [ROWS, groups, GROUP // 2 bytes] so the reduction runs inside each group first.
    One scale per group then multiplies the group's finished sum, so the scale is
    loaded once per group per row instead of being gathered per byte.
    The activation pair is loaded once per step and reused across all ROWS rows: every
    program would otherwise re-read the whole vector, and there are N / ROWS programs.
    Expected bottleneck: memory, weights plus one activation stream. A wider ROWS tile
    cuts activation traffic but multiplies the transient products tile, so ROWS is the
    knob to sweep. Too wide and the products tile spills out of registers.
    """
    n_groups = K // GROUP
    groups_per_block: tl.constexpr = BLOCK_K // GROUP
    rows = tl.program_id(axis=0) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < N
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
            x_ptr + 2 * byte_offsets, mask=group_mask[:, None], other=0.0, eviction_policy=X_EVICT
        )
        activations_high = tl.load(
            x_ptr + 2 * byte_offsets + 1, mask=group_mask[:, None], other=0.0, eviction_policy=X_EVICT
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


def int4_gemv_into(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    rows: int = ROWS_PER_PROGRAM,
    hints: bool = True,
) -> None:
    """Write (packed weights dequantized) @ x into `out`, no fp32 weights in memory.

    Assumptions: packed is uint8 [N, K // 2] contiguous, scales is fp32 or fp16
    [N, K // group_size] contiguous, x is fp32 [K] contiguous, out is fp32 [N]
    contiguous, all on one CUDA device. K must be divisible by group_size, and
    block_k must be a power of two that is a multiple of group_size.
    """
    if packed.dim() != 2 or scales.dim() != 2 or x.dim() != 1 or out.dim() != 1:
        raise ValueError(f"expected packed [N, K // 2], scales [N, K // G], x [K], out [N]; got {tuple(packed.shape)}, {tuple(scales.shape)}, {tuple(x.shape)}, {tuple(out.shape)}")
    if packed.dtype != torch.uint8:
        raise ValueError(f"expected uint8 packed weights, got {packed.dtype}")
    if scales.dtype not in SCALE_DTYPES:
        raise ValueError(f"expected fp32 or fp16 scales, got {scales.dtype}")
    if x.dtype != torch.float32 or out.dtype != torch.float32:
        raise ValueError(f"expected fp32 x and out; got {x.dtype}, {out.dtype}")

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
    int4_gemv_kernel[grid](
        packed,
        scales,
        x,
        out,
        n_rows,
        n_cols,
        GROUP=group_size,
        BLOCK_K=block_k,
        ROWS=rows,
        W_EVICT="evict_first" if hints else "",
        X_EVICT="evict_last" if hints else "",
    )


def int4_gemv(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    rows: int = ROWS_PER_PROGRAM,
    hints: bool = True,
) -> torch.Tensor:
    """Allocate the output and return the dequantized GEMV."""
    out = torch.empty(packed.shape[0], device=packed.device, dtype=torch.float32)
    int4_gemv_into(packed, scales, x, out, group_size=group_size, block_k=block_k, rows=rows, hints=hints)
    return out

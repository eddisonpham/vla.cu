"""4-bit packing with per-group scales, plus the reference dequantize.

Concept: two weights share one byte. Element 2i lives in the low nibble, element
2i+1 in the high nibble. A nibble is an unsigned 0..15 and the value is
(nibble - 8) * scale, with one scale per group_size consecutive elements along K.
This module defines the format the kernels must match. It is never fast, and nothing
here belongs in a forward pass.

Task: python -m vla_cu gemv, python -m vla_cu matrix
"""

from __future__ import annotations

import torch

GROUP_SIZE = 64
NIBBLE_OFFSET = 8
INT4_MAX = 7
SCALE_DTYPES = (torch.float32, torch.float16)


def pack_int4(
    w: torch.Tensor,
    group_size: int = GROUP_SIZE,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp32 weights [N, K] into packed uint8 [N, K // 2] and scales [N, K // group].

    Assumptions: K is divisible by group_size and group_size is even, so every group
    is a whole number of bytes and no nibble straddles a group boundary. The scale is
    the group's largest absolute value over INT4_MAX, so the range is [-8, 7] and a
    group of zeros gets scale 1 and dequantizes to zero. Codes are chosen with fp32
    scales and only the stored table is narrowed, so fp16 costs 5e-4 relative on the
    scale and nothing on the codes.
    """
    if w.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got {tuple(w.shape)}")
    if group_size <= 0 or group_size % 2:
        raise ValueError(f"expected an even positive group size, got {group_size}")
    if w.shape[1] % group_size:
        raise ValueError(f"K {w.shape[1]} is not divisible by group size {group_size}")
    if scale_dtype not in SCALE_DTYPES:
        raise ValueError(f"expected fp32 or fp16 scales, got {scale_dtype}")

    n_rows, n_cols = w.shape
    n_groups = n_cols // group_size
    groups = w.reshape(n_rows, n_groups, group_size)
    amax = groups.abs().amax(dim=2)
    scales = torch.where(amax > 0, amax / INT4_MAX, torch.ones_like(amax))

    codes = torch.round(groups / scales[:, :, None]) + NIBBLE_OFFSET
    codes = codes.clamp(0, 2 * NIBBLE_OFFSET - 1).to(torch.uint8).reshape(n_rows, n_cols)

    low = codes[:, 0::2].to(torch.int16)
    high = codes[:, 1::2].to(torch.int16)
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, scales.to(scale_dtype).contiguous()


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Return fp32 weights [N, K] with the same layout the kernel reads.

    Assumptions: packed is uint8 [N, K // 2], scales is fp32 [N, K // group_size],
    group_size matches the packing call.
    """
    if packed.dim() != 2 or packed.dtype != torch.uint8:
        raise ValueError(f"expected uint8 [N, K // 2], got {packed.dtype} {tuple(packed.shape)}")
    if scales.dim() != 2 or scales.shape[0] != packed.shape[0]:
        raise ValueError(f"expected scales [N, K // group], got {tuple(scales.shape)}")

    n_rows, n_bytes = packed.shape
    n_cols = n_bytes * 2
    if scales.shape[1] * group_size != n_cols:
        raise ValueError(f"{n_cols} weights do not divide into {scales.shape[1]} groups of {group_size}")

    codes = packed.to(torch.int32)
    low = (codes & 0x0F) - NIBBLE_OFFSET
    high = ((codes >> 4) & 0x0F) - NIBBLE_OFFSET
    values = torch.stack((low, high), dim=2).reshape(n_rows, n_cols).to(torch.float32)
    per_element_scale = scales.repeat_interleave(group_size, dim=1)
    return values * per_element_scale

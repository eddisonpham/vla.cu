"""Format-level input checks shared by every Triton and CUDA kernel wrapper.

Concept: a kernel handed the wrong dtype or shape does not fail, it reads memory and
returns plausible garbage. Each format therefore has one description of a valid call,
and every implementation of that format checks the same thing before it launches.

Raises ValueError, never TypeError, so a caller catches one exception for a bad call.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vla_cu.quant import GROUP_SIZE

SCALE_DTYPES = (torch.float32, torch.float16)


def require(condition: bool, message: str) -> None:
    """Raise ValueError with `message` unless `condition` holds."""
    if not condition:
        raise ValueError(message)


def check_power_of_two(name: str, value: int) -> None:
    """Require a positive power of two, the only tile width the kernels can mask."""
    require(value > 0 and not value & (value - 1), f"{name} must be a positive power of two, got {value}")


def check_tensors(*tensors: Tensor) -> None:
    """Require one CUDA device and contiguous storage, which the address math assumes."""
    require(all(t.is_cuda for t in tensors), "expected every tensor on a CUDA device")
    require(all(t.device == tensors[0].device for t in tensors), "expected every tensor on one device")
    require(all(t.is_contiguous() for t in tensors), "expected contiguous tensors")


def check_int8_gemv(w: Tensor, scales: Tensor, x: Tensor, out: Tensor | None = None) -> tuple[int, int]:
    """Validate an int8 GEMV call and return (N, K).

    Assumes w is int8 [N, K], scales fp32 [N], x fp32 [K], and out fp32 [N] when given.
    """
    tensors = (w, scales, x) if out is None else (w, scales, x, out)
    require(w.dim() == 2 and scales.dim() == 1 and x.dim() == 1, f"expected w [N, K], scales [N], x [K]; got {tuple(w.shape)}, {tuple(scales.shape)}, {tuple(x.shape)}")
    require(w.dtype == torch.int8, f"expected int8 weights, got {w.dtype}")
    require(scales.dtype == torch.float32 and x.dtype == torch.float32, f"expected fp32 scales and x; got {scales.dtype}, {x.dtype}")
    n_rows, n_cols = w.shape
    require(scales.shape == (n_rows,) and x.shape == (n_cols,), f"shape mismatch: w {tuple(w.shape)}, scales {tuple(scales.shape)}, x {tuple(x.shape)}")
    if out is not None:
        require(out.dim() == 1 and out.shape == (n_rows,) and out.dtype == torch.float32, f"expected fp32 out [N], got {out.dtype} {tuple(out.shape)}")
    check_tensors(*tensors)
    return n_rows, n_cols


def check_int8_grouped(
    w: Tensor,
    scales: Tensor,
    x: Tensor,
    out: Tensor,
    segment: Tensor,
    x_base: Tensor,
) -> tuple[int, int]:
    """Validate a grouped int8 GEMV call and return (rows, K).

    Assumes w int8 [R, K], scales fp32 [R], x fp32 [X], out fp32 [R], segment int32 [R],
    x_base int32 [segments], with X large enough to hold every matrix's vector.
    """
    require(w.dim() == 2 and scales.dim() == 1 and x.dim() == 1 and out.dim() == 1, f"expected w [R, K], scales [R], x [X], out [R]; got {tuple(w.shape)}, {tuple(scales.shape)}, {tuple(x.shape)}, {tuple(out.shape)}")
    require(segment.dim() == 1 and x_base.dim() == 1, f"expected segment [R] and x_base [S]; got {tuple(segment.shape)}, {tuple(x_base.shape)}")
    require(w.dtype == torch.int8, f"expected int8 weights, got {w.dtype}")
    require(segment.dtype == torch.int32 and x_base.dtype == torch.int32, f"expected int32 segment tables; got {segment.dtype}, {x_base.dtype}")
    require(scales.dtype == torch.float32 and x.dtype == torch.float32 and out.dtype == torch.float32, f"expected fp32 scales, x and out; got {scales.dtype}, {x.dtype}, {out.dtype}")
    total_rows, n_cols = w.shape
    require(scales.shape == (total_rows,) and out.shape == (total_rows,) and segment.shape == (total_rows,), f"expected one scale, output and segment entry per packed row, got {total_rows} rows")
    require(x.shape[0] >= n_cols, f"expected an activation vector of at least {n_cols}, got {x.shape[0]}")
    check_tensors(w, scales, x, out, segment, x_base)
    return total_rows, n_cols


def check_int4_gemv(
    packed: Tensor,
    scales: Tensor,
    x: Tensor,
    out: Tensor | None = None,
    group_size: int = GROUP_SIZE,
) -> tuple[int, int, int]:
    """Validate an int4 GEMV call and return (N, K, groups).

    Assumes packed uint8 [N, K // 2], scales fp32 or fp16 [N, K // group_size],
    x fp32 [K], and out fp32 [N] when given. K must divide by group_size.
    """
    tensors = (packed, scales, x) if out is None else (packed, scales, x, out)
    require(packed.dim() == 2 and scales.dim() == 2 and x.dim() == 1, f"expected packed [N, K // 2], scales [N, groups], x [K]; got {tuple(packed.shape)}, {tuple(scales.shape)}, {tuple(x.shape)}")
    require(packed.dtype == torch.uint8, f"expected uint8 packed weights, got {packed.dtype}")
    require(scales.dtype in SCALE_DTYPES, f"expected fp32 or fp16 scales, got {scales.dtype}")
    require(x.dtype == torch.float32, f"expected fp32 activations, got {x.dtype}")
    n_rows, n_bytes = packed.shape
    n_cols = n_bytes * 2
    require(group_size > 0, f"expected a positive group size, got {group_size}")
    require(n_cols % group_size == 0, f"K {n_cols} is not divisible by group size {group_size}")
    n_groups = n_cols // group_size
    require(x.shape == (n_cols,), f"expected x [{n_cols}], got {tuple(x.shape)}")
    require(scales.shape == (n_rows, n_groups), f"expected scales {(n_rows, n_groups)}, got {tuple(scales.shape)}")
    if out is not None:
        require(out.dim() == 1 and out.shape == (n_rows,) and out.dtype == torch.float32, f"expected fp32 out [N], got {out.dtype} {tuple(out.shape)}")
    check_tensors(*tensors)
    return n_rows, n_cols, n_groups


def check_int4_gemm(
    packed: Tensor,
    scales: Tensor,
    x: Tensor,
    out: Tensor,
    group_size: int = GROUP_SIZE,
) -> tuple[int, int, int, int]:
    """Validate an int4 GEMM call and return (tokens, N, K, groups).

    Assumes packed uint8 [N, K // 2], scales fp32 or fp16 [N, K // group_size],
    x fp32 [M, K], out fp32 [M, N]. K must divide by group_size.
    """
    require(packed.dim() == 2 and scales.dim() == 2 and x.dim() == 2 and out.dim() == 2, f"expected packed [N, K // 2], scales [N, groups], x [M, K], out [M, N]")
    require(packed.dtype == torch.uint8, f"expected uint8 packed weights, got {packed.dtype}")
    require(scales.dtype in SCALE_DTYPES, f"expected fp32 or fp16 scales, got {scales.dtype}")
    require(x.dtype == torch.float32 and out.dtype == torch.float32, f"expected fp32 x and out; got {x.dtype}, {out.dtype}")
    n_rows, n_bytes = packed.shape
    n_cols = n_bytes * 2
    n_tokens = x.shape[0]
    require(group_size > 0, f"expected a positive group size, got {group_size}")
    require(n_cols % group_size == 0, f"K {n_cols} is not divisible by group size {group_size}")
    n_groups = n_cols // group_size
    require(x.shape == (n_tokens, n_cols), f"expected x [(M), {n_cols}], got {tuple(x.shape)}")
    require(out.shape == (n_tokens, n_rows), f"expected out [{(n_tokens)}, {n_rows}], got {tuple(out.shape)}")
    require(scales.shape == (n_rows, n_groups), f"expected scales {(n_rows, n_groups)}, got {tuple(scales.shape)}")
    check_tensors(packed, scales, x, out)
    return n_tokens, n_rows, n_cols, n_groups


def check_int4_grouped(
    packed: Tensor,
    scales: Tensor,
    x: Tensor,
    out: Tensor,
    segment: Tensor,
    x_base: Tensor,
    group_size: int = GROUP_SIZE,
) -> tuple[int, int]:
    """Validate a grouped int4 GEMV call and return (total rows, K).

    Assumes packed uint8 [R, K // 2], scales fp32 or fp16 [R, K // group_size],
    x fp32 [X], out fp32 [R], segment int32 [R], x_base int32 [segments], with X large
    enough to hold every matrix's activation vector.
    """
    require(packed.dim() == 2 and scales.dim() == 2 and x.dim() == 1 and out.dim() == 1, f"expected packed [R, K // 2], scales [R, groups], x [X], out [R]; got {tuple(packed.shape)}, {tuple(scales.shape)}, {tuple(x.shape)}, {tuple(out.shape)}")
    require(segment.dim() == 1 and x_base.dim() == 1, f"expected segment [R] and x_base [S]; got {tuple(segment.shape)}, {tuple(x_base.shape)}")
    require(packed.dtype == torch.uint8, f"expected uint8 packed weights, got {packed.dtype}")
    require(segment.dtype == torch.int32 and x_base.dtype == torch.int32, f"expected int32 segment tables; got {segment.dtype}, {x_base.dtype}")
    require(scales.dtype in SCALE_DTYPES, f"expected fp32 or fp16 scales, got {scales.dtype}")
    require(x.dtype == torch.float32 and out.dtype == torch.float32, f"expected fp32 x and out; got {x.dtype}, {out.dtype}")
    total_rows, n_bytes = packed.shape
    n_cols = n_bytes * 2
    require(group_size > 0, f"expected a positive group size, got {group_size}")
    require(n_cols % group_size == 0, f"K {n_cols} is not divisible by group size {group_size}")
    n_groups = n_cols // group_size
    require(scales.shape == (total_rows, n_groups), f"expected scales {(total_rows, n_groups)}, got {tuple(scales.shape)}")
    require(out.shape == (total_rows,) and segment.shape == (total_rows,), f"expected one output and one segment entry per packed row, got {total_rows} rows")
    require(x.shape[0] >= n_cols, f"expected an activation vector of at least {n_cols}, got {x.shape[0]}")
    check_tensors(packed, scales, x, out, segment, x_base)
    return total_rows, n_cols

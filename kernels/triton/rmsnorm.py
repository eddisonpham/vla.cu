"""RMSNorm and residual add, one program per row, for a token or a token chunk.

Concept: RMSNorm scales each row by the inverse root mean square of that row and then by
a learned per-feature weight. One program owns one row, so a chunk of tokens is one launch
with `tokens` programs and a single token is the same kernel with one program. The
residual add and the add-then-norm fusion are row kernels for the same reason, which is
what stops the per-token launch tail from scaling with the token count.

EPS is the model's `rms_norm_eps`, 1e-05 from the SmolVLM2 text config, and the fp32
reference in `bench/scale.py` imports the same constant, so the two sides cannot drift.

Task: python -m vla_cu scale
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.validate import require

EPS = 1e-05


@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, K, BLOCK: tl.constexpr, EPS: tl.constexpr) -> None:
    """Write x * weight / sqrt(mean(x^2) + eps) for one row of K fp32 values.

    The block is the next power of two above K, so a width like 720 loads 720 lanes and
    leaves the rest masked. The mean runs over the row, never over the chunk.
    """
    offsets = tl.arange(0, BLOCK)
    mask = offsets < K
    base = tl.program_id(axis=0) * K
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    inverse_rms = 1.0 / tl.sqrt(tl.sum(x * x) / K + EPS)
    tl.store(out_ptr + base + offsets, x * inverse_rms * weight, mask=mask)


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, K, BLOCK: tl.constexpr) -> None:
    """Write a + b for one row of K fp32 values, which is the residual connection."""
    offsets = tl.arange(0, BLOCK)
    mask = offsets < K
    base = tl.program_id(axis=0) * K
    total = tl.load(a_ptr + base + offsets, mask=mask, other=0.0) + tl.load(b_ptr + base + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + base + offsets, total, mask=mask)


@triton.jit
def add_norm_kernel(
    a_ptr,
    b_ptr,
    weight_ptr,
    added_ptr,
    normed_ptr,
    K,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    """Write a + b and the RMSNorm of that sum for one row of K fp32 values.

    `added` receives the new residual and `normed` receives added * weight /
    sqrt(mean(added^2) + eps), so a residual plus its norm costs one launch instead of
    two. Both stores read a sum that is already in registers.
    """
    offsets = tl.arange(0, BLOCK)
    mask = offsets < K
    base = tl.program_id(axis=0) * K
    total = tl.load(a_ptr + base + offsets, mask=mask, other=0.0) + tl.load(b_ptr + base + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    inverse_rms = 1.0 / tl.sqrt(tl.sum(total * total) / K + EPS)
    tl.store(added_ptr + base + offsets, total, mask=mask)
    tl.store(normed_ptr + base + offsets, total * inverse_rms * weight, mask=mask)


def _rows(name: str, tensor: torch.Tensor, width: int) -> int:
    """Require contiguous fp32 rows of `width`, 1D or 2D, and return the row count."""
    require(tensor.dim() in (1, 2), f"{name} must be 1D or 2D, got {tuple(tensor.shape)}")
    require(tensor.dtype == torch.float32, f"{name} must be fp32, got {tensor.dtype}")
    require(tensor.shape[-1] == width, f"{name} has {tensor.shape[-1]} elements per row, expected {width}")
    require(tensor.is_contiguous(), f"{name} must be contiguous")
    return 1 if tensor.dim() == 1 else tensor.shape[0]


def rmsnorm_into(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Write the RMSNorm of `x` [rows, K] or [K] into `out`, using the `weight` [K]."""
    width = weight.shape[0]
    rows = _rows("x", x, width)
    _rows("weight", weight, width)
    _rows("out", out, width)
    require(out.shape == x.shape, f"expected out {tuple(x.shape)}, got {tuple(out.shape)}")
    if rows == 0:
        return
    rmsnorm_kernel[(rows,)](x, weight, out, width, BLOCK=triton.next_power_of_2(width), EPS=EPS)


def add_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Write the elementwise sum of `a` and `b` [rows, K] or [K] into `out`."""
    width = a.shape[-1]
    rows = _rows("a", a, width)
    _rows("b", b, width)
    _rows("out", out, width)
    require(b.shape == a.shape and out.shape == a.shape, f"expected three {tuple(a.shape)} tensors")
    if rows == 0:
        return
    add_kernel[(rows,)](a, b, out, width, BLOCK=triton.next_power_of_2(width))


def add_norm_into(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: torch.Tensor,
    added: torch.Tensor,
    normed: torch.Tensor,
) -> None:
    """Write a + b into `added` and its RMSNorm into `normed`, over every row, in one launch."""
    width = weight.shape[0]
    rows = _rows("a", a, width)
    _rows("b", b, width)
    _rows("weight", weight, width)
    _rows("added", added, width)
    _rows("normed", normed, width)
    require(b.shape == a.shape and added.shape == a.shape and normed.shape == a.shape, f"expected three {tuple(a.shape)} tensors")
    if rows == 0:
        return
    add_norm_kernel[(rows,)](a, b, weight, added, normed, width, BLOCK=triton.next_power_of_2(width), EPS=EPS)

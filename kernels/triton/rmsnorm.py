"""RMSNorm and residual add at batch 1, one program per vector.

Concept: RMSNorm scales a vector by the inverse root mean square and then by a learned
per-feature weight. At one token the whole vector fits in one program, so the reduction
is a single sum and the kernel is launch bound rather than bandwidth bound. The
residual is its own pass here, so a normed layer costs norm, up, down, add. EPS is the
model's `rms_norm_eps`, 1e-05 from the SmolVLM2 text config, and the fp32 reference in
`bench/scale.py` imports the same constant, so the two sides cannot drift.

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
    """Write x * weight / sqrt(mean(x^2) + eps) for one fp32 vector of length K.

    The block is the next power of two above K, so a width like 720 loads 720 lanes
    masked and 304 idle ones. One program, one pass over the vector.
    """
    offsets = tl.arange(0, BLOCK)
    mask = offsets < K
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    inverse_rms = 1.0 / tl.sqrt(tl.sum(x * x) / K + EPS)
    tl.store(out_ptr + offsets, x * inverse_rms * weight, mask=mask)


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, K, BLOCK: tl.constexpr) -> None:
    """Write a + b for two fp32 vectors of length K, which is the residual connection."""
    offsets = tl.arange(0, BLOCK)
    mask = offsets < K
    total = tl.load(a_ptr + offsets, mask=mask, other=0.0) + tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, total, mask=mask)


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
    """Write a + b and the RMSNorm of that sum for one fp32 vector of length K.

    `added` receives the new residual and `normed` receives added * weight /
    sqrt(mean(added^2) + eps), so a residual plus its norm costs one launch instead of
    two. One program, one pass: both stores read the sum that is already in registers.
    """
    offsets = tl.arange(0, BLOCK)
    mask = offsets < K
    total = tl.load(a_ptr + offsets, mask=mask, other=0.0) + tl.load(b_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    inverse_rms = 1.0 / tl.sqrt(tl.sum(total * total) / K + EPS)
    tl.store(added_ptr + offsets, total, mask=mask)
    tl.store(normed_ptr + offsets, total * inverse_rms * weight, mask=mask)


def _check_vector(name: str, vector: torch.Tensor, length: int) -> None:
    """Require a contiguous fp32 vector of `length` on the device the caller uses."""
    require(vector.dim() == 1, f"{name} must be 1D, got {tuple(vector.shape)}")
    require(vector.dtype == torch.float32, f"{name} must be fp32, got {vector.dtype}")
    require(vector.shape[0] == length, f"{name} has {vector.shape[0]} elements, expected {length}")
    require(vector.is_contiguous(), f"{name} must be contiguous")


def rmsnorm_into(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Write the RMSNorm of `x` [K] into `out`, using the per-feature `weight` [K]."""
    length = x.shape[0]
    _check_vector("x", x, length)
    _check_vector("weight", weight, length)
    _check_vector("out", out, length)
    rmsnorm_kernel[(1,)](x, weight, out, length, BLOCK=triton.next_power_of_2(length), EPS=EPS)


def add_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Write the elementwise sum of `a` and `b` [K] into `out`."""
    length = a.shape[0]
    _check_vector("a", a, length)
    _check_vector("b", b, length)
    _check_vector("out", out, length)
    add_kernel[(1,)](a, b, out, length, BLOCK=triton.next_power_of_2(length))


def add_norm_into(
    a: torch.Tensor,
    b: torch.Tensor,
    weight: torch.Tensor,
    added: torch.Tensor,
    normed: torch.Tensor,
) -> None:
    """Write a + b into `added` and its RMSNorm into `normed`, each [K], in one launch."""
    length = a.shape[0]
    _check_vector("a", a, length)
    _check_vector("b", b, length)
    _check_vector("weight", weight, length)
    _check_vector("added", added, length)
    _check_vector("normed", normed, length)
    add_norm_kernel[(1,)](a, b, weight, added, normed, length, BLOCK=triton.next_power_of_2(length), EPS=EPS)

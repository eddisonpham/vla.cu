"""Loader and Python wrapper for the CUDA C++ int8 GEMV.

Concept: the same warp-per-row mapping as the int4 kernel, with 16 weights per lane
instead of 8 and streaming loads on both, so a whole block can run on native kernels
without the Triton launcher in the loop. That is what makes a CUDA graph of a block
worth capturing.

Task: python -m vla_cu fastpath
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

CSRC = Path(__file__).resolve().parent.parent / "csrc" / "int8_gemv.cu"
_EXTENSION = None


def extension():
    """Compile the extension on first use and return the cached module.

    Like the int4 loader, this needs nvcc, a host compiler and ninja, which on Windows
    means a shell where vcvars64.bat has run. A missing compiler arrives as
    CalledProcessError, so both failure types are wrapped.
    """
    global _EXTENSION
    if _EXTENSION is None:
        try:
            _EXTENSION = load(
                name="vla_cu_int8_gemv",
                sources=[str(CSRC)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
            )
        except (RuntimeError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"could not build {CSRC.name}: {error}. On Windows, call vcvars64.bat in this shell first (see agents/HARDWARE.md)."
            ) from error
    return _EXTENSION


def int8_gemv_cuda_into(w: Tensor, scales: Tensor, x: Tensor, out: Tensor) -> None:
    """Write scales * (w @ x) into `out` with the CUDA kernel, allocation free.

    Assumptions: w int8 [N, K] contiguous, scales fp32 [N], x fp32 [K] 16-byte aligned,
    out fp32 [N], all on one CUDA device. K must be a multiple of 16 so every row base
    stays aligned for the 16-byte weight load; a scalar tail handles K not divisible by 512.
    """
    if w.dtype != torch.int8:
        raise ValueError(f"expected int8 weights, got {w.dtype}")
    if scales.dtype != torch.float32 or x.dtype != torch.float32 or out.dtype != torch.float32:
        raise ValueError(f"expected fp32 scales, x and out; got {scales.dtype}, {x.dtype}, {out.dtype}")
    if w.dim() != 2 or scales.dim() != 1 or x.dim() != 1 or out.dim() != 1:
        raise ValueError("expected w [N, K], scales [N], x [K], out [N]")
    if w.shape[0] != scales.shape[0] or w.shape[1] != x.shape[0] or out.shape[0] != w.shape[0]:
        raise ValueError(f"shape mismatch: w {tuple(w.shape)}, scales {tuple(scales.shape)}, x {tuple(x.shape)}, out {tuple(out.shape)}")
    extension().int8_gemv_launch(w, scales, x, out)


def int8_gemv_cuda(w: Tensor, scales: Tensor, x: Tensor) -> Tensor:
    """Allocate the output and return scales * (w @ x) computed by the CUDA kernel."""
    out = torch.empty(w.shape[0], device=w.device, dtype=torch.float32)
    int8_gemv_cuda_into(w, scales, x, out)
    return out

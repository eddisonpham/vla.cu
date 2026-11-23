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

from vla_cu.validate import check_int8_gemv, require

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
    n_rows, n_cols = check_int8_gemv(w, scales, x, out)
    require(n_cols % 16 == 0, f"K must be a multiple of 16 so every row stays 16-byte aligned, got {n_cols}")
    require(x.data_ptr() % 16 == 0, "activations must be 16-byte aligned")
    require(w.data_ptr() % 16 == 0, "weights must be 16-byte aligned")
    if n_rows == 0 or n_cols == 0:
        return
    extension().int8_gemv_launch(w, scales, x, out)


def int8_gemv_cuda(w: Tensor, scales: Tensor, x: Tensor) -> Tensor:
    """Allocate the output and return scales * (w @ x) computed by the CUDA kernel."""
    out = torch.empty(w.shape[0], device=w.device, dtype=torch.float32)
    int8_gemv_cuda_into(w, scales, x, out)
    return out

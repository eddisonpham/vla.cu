"""Loader and Python wrapper for the CUDA C++ port of the fused int4 GEMV.

Concept: `torch.utils.cpp_extension.load` compiles a .cu file with nvcc and the host
compiler at import time and caches the result, so a hand-written kernel stays a
normal Python call. That is enough for a learning project; CMake only appears when
the build itself becomes the subject.

Task: python -m vla_cu cuda
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

from vla_cu.quant import GROUP_SIZE

CSRC = Path(__file__).resolve().parent.parent / "csrc" / "int4_gemv.cu"
_EXTENSION = None


def extension():
    """Compile the extension on first use and return the cached module.

    A build needs nvcc, a host compiler and ninja, and on Windows the host compiler is
    only on PATH inside a shell where vcvars64.bat has been called. The error says so,
    because the underlying message is only about ninja.
    """
    global _EXTENSION
    if _EXTENSION is None:
        try:
            _EXTENSION = load(
                name="vla_cu_int4_gemv",
                sources=[str(CSRC)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"could not build {CSRC.name}: {error}. On Windows, call vcvars64.bat in this shell first (see agents/HARDWARE.md)."
            ) from error
    return _EXTENSION


def int4_gemv_cuda_into(
    packed: Tensor,
    scales: Tensor,
    x: Tensor,
    out: Tensor,
    group_size: int = GROUP_SIZE,
) -> None:
    """Write the dequantized GEMV of packed weights into `out` with the CUDA kernel.

    Assumptions: packed uint8 [N, K // 2], scales fp32 [N, K // 64], x fp32 [K],
    out fp32 [N], all contiguous on one CUDA device. The group size is fixed at 64
    because one warp segment of eight lanes maps exactly onto one group.
    """
    if group_size != GROUP_SIZE:
        raise ValueError(f"the CUDA kernel is written for group size {GROUP_SIZE}, got {group_size}")
    if packed.dtype != torch.uint8:
        raise ValueError(f"expected uint8 packed weights, got {packed.dtype}")
    if packed.dim() != 2 or scales.dim() != 2 or x.dim() != 1 or out.dim() != 1:
        raise ValueError("expected packed [N, K // 2], scales [N, groups], x [K], out [N]")
    extension().int4_gemv_launch(packed, scales, x, out)


def int4_gemv_cuda(packed: Tensor, scales: Tensor, x: Tensor, group_size: int = GROUP_SIZE) -> Tensor:
    """Allocate the output and return the dequantized GEMV computed by the CUDA kernel."""
    out = torch.empty(packed.shape[0], device=packed.device, dtype=torch.float32)
    int4_gemv_cuda_into(packed, scales, x, out, group_size=group_size)
    return out

"""The CUDA int8 GEMV against the Triton int8 GEMV, awkward widths included."""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from kernels.triton.int8_gemv import int8_gemv
from vla_cu.cuda_int8 import int8_gemv_cuda

HOST_COMPILER = (
    shutil.which("cl")
    if os.name == "nt"
    else shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
)
pytestmark = pytest.mark.skipif(
    shutil.which("nvcc") is None or HOST_COMPILER is None,
    reason="no CUDA build toolchain on PATH; on Windows, run inside vcvars64.bat",
)

SHAPES = ((1, 16), (3, 960), (17, 1040), (64, 2560))


@pytest.mark.parametrize(("n_out", "n_in"), SHAPES)
def test_matches_triton(n_out: int, n_in: int) -> None:
    """The 16-wide fast loop and the scalar tail must reproduce the reference exactly."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randint(-127, 128, (n_out, n_in), device="cuda", dtype=torch.int8, generator=generator)
    scales = torch.rand(n_out, device="cuda", generator=generator) * 0.02 + 1e-3
    x = torch.randn(n_in, device="cuda", generator=generator)

    out = int8_gemv_cuda(w, scales, x)

    assert out.shape == (n_out,)
    torch.testing.assert_close(out, int8_gemv(w, scales, x), rtol=1e-4, atol=1e-3)


def test_rejects_length_that_breaks_vector_alignment() -> None:
    """A row base that is not 16-byte aligned would fault inside the 16-byte load."""
    w = torch.zeros((4, 1025), device="cuda", dtype=torch.int8)
    scales = torch.ones(4, device="cuda")
    x = torch.zeros(1025, device="cuda")

    with pytest.raises(RuntimeError):
        int8_gemv_cuda(w, scales, x)


def test_rejects_float_weights() -> None:
    """Handing fp32 weights to the int8 path would reinterpret bytes, not complain later."""
    w = torch.zeros((4, 64), device="cuda", dtype=torch.float32)
    scales = torch.ones(4, device="cuda")
    x = torch.zeros(64, device="cuda")

    with pytest.raises(ValueError):
        int8_gemv_cuda(w, scales, x)

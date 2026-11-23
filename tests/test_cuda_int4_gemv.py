"""The CUDA C++ int4 GEMV against dequantize-then-matmul in float64.

The port has its own thread mapping, so it needs its own correctness gate rather than
a comparison against the Triton kernel alone.
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from vla_cu.cuda_int4 import VARIANTS
from vla_cu.quant import dequantize_int4, pack_int4

HOST_COMPILER = (
    shutil.which("cl")
    if os.name == "nt"
    else shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
)
pytestmark = pytest.mark.skipif(
    shutil.which("nvcc") is None or HOST_COMPILER is None,
    reason="no CUDA build toolchain on PATH; on Windows, run inside vcvars64.bat",
)

SHAPES = ((1, 64), (3, 128), (17, 320), (64, 960), (128, 2048))


def reference(packed: torch.Tensor, scales: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Dequantize to float64, then matmul, so the reference carries no fp32 error."""
    weights = dequantize_int4(packed, scales).to(torch.float64)
    return (weights @ x.to(torch.float64)).to(torch.float32)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(("n_out", "n_in"), SHAPES)
def test_matches_dequantize_then_matmul(n_out: int, n_in: int, variant: str) -> None:
    """Warp per row, uint32 weight loads and the row tail all have to agree with the reference."""
    from vla_cu.cuda_int4 import int4_gemv_cuda

    generator = torch.Generator(device="cuda").manual_seed(11)
    w = torch.randn((n_out, n_in), device="cuda", generator=generator) * 0.05
    x = torch.randn(n_in, device="cuda", generator=generator)
    packed, scales = pack_int4(w)

    out = int4_gemv_cuda(packed, scales, x, variant=variant)

    assert out.shape == (n_out,)
    torch.testing.assert_close(out, reference(packed, scales, x), rtol=1e-4, atol=1e-3)


def test_agrees_with_the_triton_kernel() -> None:
    """Two independent mappings of the same format must land on the same numbers."""
    from kernels.triton.int4_gemv import int4_gemv
    from vla_cu.cuda_int4 import int4_gemv_cuda

    generator = torch.Generator(device="cuda").manual_seed(12)
    w = torch.randn((300, 1024), device="cuda", generator=generator) * 0.05
    x = torch.randn(1024, device="cuda", generator=generator)
    packed, scales = pack_int4(w)

    torch.testing.assert_close(int4_gemv_cuda(packed, scales, x), int4_gemv(packed, scales, x), rtol=1e-4, atol=1e-3)


def test_rejects_a_group_size_the_thread_map_cannot_serve() -> None:
    """One warp segment of eight lanes covers exactly one 64-weight group, nothing else."""
    from vla_cu.cuda_int4 import int4_gemv_cuda

    w = torch.randn((4, 128), device="cuda") * 0.05
    packed, scales = pack_int4(w)
    x = torch.randn(128, device="cuda")

    with pytest.raises(ValueError):
        int4_gemv_cuda(packed, scales, x, group_size=32)

"""Every kernel entry point rejects the same classes of bad input before it launches."""

from __future__ import annotations

import pytest
import torch

from kernels.triton.int4_gemm import int4_gemm_into
from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from kernels.triton.int8_gemv import int8_gemv_into
from kernels.triton.int8_grouped_gemv import int8_grouped_gemv_into
from vla_cu.quant import GROUP_SIZE, pack_int4

NAMES = ("int8_gemv", "int4_gemv", "int4_gemm", "int4_grouped", "int8_grouped")
N_ROWS = 8
N_COLS = 128


def valid_tensors(device: str = "cuda") -> dict[str, torch.Tensor]:
    """Return one valid activation, scale table and output set for the 8 x 128 shape."""
    packed, scales = pack_int4(torch.zeros((N_ROWS, N_COLS), device=device))
    return {
        "packed": packed,
        "scales": scales,
        "w": torch.zeros((N_ROWS, N_COLS), device=device, dtype=torch.int8),
        "s8": torch.ones(N_ROWS, device=device),
        "out": torch.empty(N_ROWS, device=device),
        "segment": torch.zeros(N_ROWS, device=device, dtype=torch.int32),
        "x_base": torch.zeros(1, device=device, dtype=torch.int32),
    }


def call(name: str, t: dict[str, torch.Tensor], x: torch.Tensor) -> None:
    """Run one entry point with a substituted activation tensor."""
    if name == "int8_gemv":
        int8_gemv_into(t["w"], t["s8"], x, t["out"])
    elif name == "int4_gemv":
        int4_gemv_into(t["packed"], t["scales"], x, t["out"])
    elif name == "int4_gemm":
        int4_gemm_into(t["packed"], t["scales"], x.reshape(1, -1), torch.empty((1, N_ROWS), device=x.device))
    elif name == "int8_grouped":
        int8_grouped_gemv_into(t["w"], t["s8"], x, t["out"], t["segment"], t["x_base"])
    else:
        int4_grouped_gemv_into(t["packed"], t["scales"], x, t["out"], t["segment"], t["x_base"])


@pytest.mark.parametrize("name", NAMES)
def test_rejects_a_non_contiguous_activation(name: str) -> None:
    """A strided vector would make the kernel read the wrong bytes, not fail."""
    strided = torch.zeros((N_COLS, 2), device="cuda").t()[0]
    assert not strided.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        call(name, valid_tensors(), strided)


@pytest.mark.parametrize("name", NAMES)
def test_rejects_cpu_tensors(name: str) -> None:
    """A CPU pointer handed to a kernel launch is a crash, so the wrapper must refuse it."""
    with pytest.raises(ValueError, match="CUDA"):
        call(name, valid_tensors("cpu"), torch.zeros(N_COLS))


def test_int4_gemm_rejects_a_block_n_that_is_not_a_power_of_two() -> None:
    """The step tiles are masked with a power of two, so 96 is not a width the kernel can express."""
    t = valid_tensors()
    with pytest.raises(ValueError, match="power of two"):
        int4_gemm_into(
            t["packed"],
            t["scales"],
            torch.zeros((1, N_COLS), device="cuda"),
            torch.empty((1, N_ROWS), device="cuda"),
            tile=(16, 96, 64, 2),
        )


def test_cuda_wrapper_rejects_an_unaligned_row_width() -> None:
    """The 16-byte int8 load needs K divisible by 16, which the Triton kernel does not."""
    from vla_cu.cuda_int8 import int8_gemv_cuda_into

    w = torch.zeros((N_ROWS, 24), device="cuda", dtype=torch.int8)
    with pytest.raises(ValueError, match="16"):
        int8_gemv_cuda_into(w, torch.ones(N_ROWS, device="cuda"), torch.zeros(24, device="cuda"), torch.empty(N_ROWS, device="cuda"))


def test_every_group_size_must_divide_the_width() -> None:
    """A ragged last group is not representable by the current format (DEC-030)."""
    w = torch.zeros((N_ROWS, 720), device="cuda")
    with pytest.raises(ValueError, match="divisible"):
        pack_int4(w, group_size=GROUP_SIZE)

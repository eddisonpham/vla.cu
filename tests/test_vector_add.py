"""Correctness tests for the Triton vector add against PyTorch."""

import pytest
import torch

from kernels.triton.vector_add import vector_add

SIZES = (0, 1, 1023, 1024, 1025, 1_000_003)


@pytest.mark.parametrize("n_elements", SIZES)
def test_matches_pytorch(n_elements: int) -> None:
    """Every size, including empty and non-multiples of the block, matches exactly."""
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)

    out = vector_add(x, y)
    torch.cuda.synchronize()

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert torch.equal(out, x + y)


def test_rejects_mismatched_shapes() -> None:
    """A shape mismatch is a caller error, not a garbage result."""
    x = torch.randn(8, device="cuda")
    y = torch.randn(9, device="cuda")

    with pytest.raises(ValueError):
        vector_add(x, y)

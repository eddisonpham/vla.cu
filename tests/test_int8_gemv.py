"""The fused int8 GEMV against a dequantize-then-matmul reference in float64.

Sizes are chosen to break assumptions: one element, K not a multiple of BLOCK_K,
prime lengths, more than one output row, and zero rows.
"""

from __future__ import annotations

import pytest
import torch

from kernels.triton.int8_gemv import int8_gemv

SHAPES = ((1, 1), (1, 1025), (3, 64), (17, 257), (64, 1000), (0, 128))


def reference(w: torch.Tensor, scales: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Dequantize to float64, then matmul, so the comparison has no reference error."""
    w_float = w.to(torch.float64) * scales.to(torch.float64)[:, None]
    return (w_float @ x.to(torch.float64)).to(torch.float32)


@pytest.mark.parametrize(("n_out", "n_in"), SHAPES)
def test_matches_dequantize_then_matmul(n_out: int, n_in: int) -> None:
    """Fused and naive paths agree elementwise, including scales far from one."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randint(-127, 128, (n_out, n_in), device="cuda", dtype=torch.int8, generator=generator)
    scales = torch.rand(n_out, device="cuda", generator=generator) * 0.02 + 0.001
    x = torch.randn(n_in, device="cuda", generator=generator)

    out = int8_gemv(w, scales, x)

    assert out.shape == (n_out,)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, reference(w, scales, x), rtol=1e-4, atol=1e-4)


def test_zero_scales_give_zero() -> None:
    """A scale of zero silences its row, which catches a scale applied to the wrong row."""
    w = torch.randint(-127, 128, (5, 300), device="cuda", dtype=torch.int8)
    scales = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], device="cuda")
    x = torch.ones(300, device="cuda")

    out = int8_gemv(w, scales, x)

    assert out[1].item() == 0.0
    assert out[3].item() == 0.0
    torch.testing.assert_close(out, reference(w, scales, x), rtol=1e-4, atol=1e-4)


def test_rejects_mismatched_shapes() -> None:
    """A vector whose length disagrees with the weight rows must fail, not read past it."""
    w = torch.zeros((4, 8), device="cuda", dtype=torch.int8)
    scales = torch.zeros(4, device="cuda")
    x = torch.zeros(9, device="cuda")

    with pytest.raises(ValueError):
        int8_gemv(w, scales, x)


def test_rejects_non_int8_weights() -> None:
    """fp32 weights would silently pass the launch, so the wrapper must refuse them."""
    w = torch.zeros((4, 8), device="cuda", dtype=torch.float32)
    scales = torch.zeros(4, device="cuda")
    x = torch.zeros(8, device="cuda")

    with pytest.raises(ValueError):
        int8_gemv(w, scales, x)

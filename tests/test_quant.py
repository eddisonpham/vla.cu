"""Packing layout and quantization error for the 4-bit format."""

from __future__ import annotations

import pytest
import torch

from vla_cu.quant import GROUP_SIZE, dequantize_int4, pack_int4


def test_dequantize_reads_low_nibble_first() -> None:
    """The kernel assumes element 2i is the low nibble, so the reference must too."""
    packed = torch.tensor([[0x0F, 0x88]], device="cuda", dtype=torch.uint8)
    scales = torch.tensor([[1.0]], device="cuda")

    out = dequantize_int4(packed, scales, group_size=4)

    assert out.tolist() == [[7.0, -8.0, 0.0, 0.0]]


def test_zero_group_quantizes_to_zero() -> None:
    """An all-zero group must not divide by a zero scale."""
    w = torch.zeros((2, GROUP_SIZE), device="cuda")

    packed, scales = pack_int4(w)

    assert torch.all(scales == 1.0)
    assert torch.equal(dequantize_int4(packed, scales), w)


@pytest.mark.parametrize(("n_out", "n_in"), ((1, 64), (16, 960), (64, 2048)))
def test_error_stays_within_half_a_scale(n_out: int, n_in: int) -> None:
    """Round to nearest gives at most half a quantization step of error per weight."""
    w = torch.randn((n_out, n_in), device="cuda") * 0.05

    packed, scales = pack_int4(w)
    approx = dequantize_int4(packed, scales)
    step = scales.repeat_interleave(GROUP_SIZE, dim=1)

    assert packed.shape == (n_out, n_in // 2)
    assert scales.shape == (n_out, n_in // GROUP_SIZE)
    assert ((w - approx).abs() <= step / 2 + 1e-7).all()


def test_rejects_non_divisible_length() -> None:
    """A K that is not a whole number of groups would put one scale on two groups."""
    w = torch.zeros((1, 100), device="cuda")

    with pytest.raises(ValueError):
        pack_int4(w)

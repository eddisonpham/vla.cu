"""The tower's packed projections, which the 64 image prefix rows are built from."""

from __future__ import annotations

import pytest
import torch

from vla_cu.vision import _Projection


@pytest.mark.parametrize(("bits", "floor"), [(8, 0.999), (4, 0.99)])
def test_packed_projection_tracks_the_fp32_product(bits: int, floor: float) -> None:
    """A packed projection must hold cosine with the fp32 product it replaces."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn((256, 768), device="cuda", generator=generator)
    bias = torch.randn(256, device="cuda", generator=generator)
    x = torch.randn((1024, 768), device="cuda", generator=generator)

    out = _Projection(weight, bias, bits)(x)
    expected = x @ weight.T + bias

    assert out.shape == expected.shape
    held = torch.nn.functional.cosine_similarity(out.reshape(-1), expected.reshape(-1), dim=0)
    assert held > floor

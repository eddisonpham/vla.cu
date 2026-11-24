"""The tower's packed projections, which the 64 image prefix rows are built from."""

from __future__ import annotations

import pytest
import torch

from vla_cu.quant import dequantize_int4, pack_int4
from vla_cu.vision import _Projection


@pytest.mark.parametrize(("bits", "group", "floor"), [(8, 64, 0.999), (4, 64, 0.99), (4, 32, 0.99), (4, 16, 0.99)])
def test_packed_projection_tracks_the_fp32_product(bits: int, group: int, floor: float) -> None:
    """A packed projection must hold cosine with the fp32 product it replaces."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn((256, 768), device="cuda", generator=generator)
    bias = torch.randn(256, device="cuda", generator=generator)
    x = torch.randn((1024, 768), device="cuda", generator=generator)

    out = _Projection(weight, bias, bits, group_size=group)(x)
    expected = x @ weight.T + bias

    assert out.shape == expected.shape
    held = torch.nn.functional.cosine_similarity(out.reshape(-1), expected.reshape(-1), dim=0)
    assert held > floor


def test_a_smaller_group_never_loses_accuracy() -> None:
    """Sixteen weights per scale must read at least as well as sixty four on the same matrix."""
    generator = torch.Generator(device="cuda").manual_seed(1)
    weight = torch.randn((256, 768), device="cuda", generator=generator)
    x = torch.randn((1024, 768), device="cuda", generator=generator)
    expected = x @ weight.T

    wide = _Projection(weight, None, 4, group_size=64)(x)
    narrow = _Projection(weight, None, 4, group_size=16)(x)
    held = [
        torch.nn.functional.cosine_similarity(out.reshape(-1), expected.reshape(-1), dim=0)
        for out in (wide, narrow)
    ]

    assert held[1] >= held[0]


def test_exact_activations_are_the_dequantized_weight_product() -> None:
    """Activation bits 32 must be the fp32 matmul of the dequantized 4-bit weight."""
    generator = torch.Generator(device="cuda").manual_seed(2)
    weight = torch.randn((256, 768), device="cuda", generator=generator)
    bias = torch.randn(256, device="cuda", generator=generator)
    x = torch.randn((1024, 768), device="cuda", generator=generator)

    out = _Projection(weight, bias, 4, group_size=16, activation_bits=32)(x)
    packed, scales = pack_int4(weight, group_size=16)
    expected = x @ dequantize_int4(packed, scales, 16).T + bias

    torch.testing.assert_close(out, expected)


def test_int8_activations_keep_most_of_the_product() -> None:
    """Rounding each activation row to int8 must stay close to the exact product."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    weight = torch.randn((256, 768), device="cuda", generator=generator)
    x = torch.randn((1024, 768), device="cuda", generator=generator)

    exact = _Projection(weight, None, 4, group_size=16, activation_bits=32)(x)
    rounded = _Projection(weight, None, 4, group_size=16, activation_bits=8)(x)
    held = torch.nn.functional.cosine_similarity(rounded.reshape(-1), exact.reshape(-1), dim=0)

    assert held > 0.99

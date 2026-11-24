"""The pack file's projection round trip, which is what a warm start replays."""

from __future__ import annotations

import pytest
import torch

from vla_cu.vision import _Projection


@pytest.mark.parametrize(("bits", "group"), [(32, 64), (8, 64), (4, 64), (4, 16)])
def test_a_projection_survives_a_state_round_trip(bits: int, group: int) -> None:
    """A projection rebuilt from its stored tensors must produce exactly the same output."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn((256, 768), device="cuda", generator=generator)
    bias = torch.randn(256, device="cuda", generator=generator)
    x = torch.randn((64, 768), device="cuda", generator=generator)

    before = _Projection(weight, bias, bits, group_size=group)
    after = _Projection.from_state(before.state(), bits, group)

    assert after.bytes == before.bytes
    assert torch.equal(after(x), before(x))

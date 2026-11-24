"""The connector's pixel shuffle order, which the image row order depends on."""

from __future__ import annotations

import pytest
import torch

from vla_cu.vision import pixel_shuffle


@pytest.mark.cpu
def test_pixel_shuffle_folds_height_major_blocks() -> None:
    """A 4 by 4 patch grid at scale 2 must fold into height block, then width block order."""
    x = torch.arange(16, dtype=torch.float32).view(1, 16, 1)
    expected = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15], dtype=torch.float32).view(1, 4, 4)

    out = pixel_shuffle(x, 2)

    assert out.shape == (1, 4, 4)
    torch.testing.assert_close(out, expected)


@pytest.mark.cpu
def test_pixel_shuffle_channel_order_matches_the_connector() -> None:
    """Two channels must stay innermost inside each folded row."""
    x = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]).view(1, 4, 2)
    expected = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]).view(1, 1, 8)

    out = pixel_shuffle(x, 2)

    assert out.shape == (1, 1, 8)
    torch.testing.assert_close(out, expected)

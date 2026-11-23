"""The grouped int8 GEMV against one launch per matrix, with per-matrix activation bases."""

from __future__ import annotations

import pytest
import torch

from kernels.triton.int8_gemv import int8_gemv
from kernels.triton.int8_grouped_gemv import int8_grouped_gemv_into

ROWS = (8, 12)
WIDTH = 128


def test_matches_one_launch_per_matrix() -> None:
    """A segment table must select each matrix's own weights, scale and activation base."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    first = torch.randint(-127, 128, (ROWS[0], WIDTH), device="cuda", dtype=torch.int8, generator=generator)
    second = torch.randint(-127, 128, (ROWS[1], WIDTH), device="cuda", dtype=torch.int8, generator=generator)
    weights = torch.cat((first, second), dim=0)
    scales = torch.rand(sum(ROWS), device="cuda", generator=generator) * 0.02 + 1e-3
    x_first = torch.randn(WIDTH, device="cuda", generator=generator)
    x_second = torch.randn(WIDTH, device="cuda", generator=generator)

    out = torch.empty(sum(ROWS), device="cuda")
    segment = torch.cat(
        (
            torch.zeros(ROWS[0], device="cuda", dtype=torch.int32),
            torch.ones(ROWS[1], device="cuda", dtype=torch.int32),
        )
    )
    x_base = torch.tensor([0, WIDTH], device="cuda", dtype=torch.int32)

    int8_grouped_gemv_into(weights, scales, torch.cat((x_first, x_second)), out, segment, x_base)

    expected = torch.cat(
        (
            int8_gemv(weights[: ROWS[0]], scales[: ROWS[0]], x_first),
            int8_gemv(weights[ROWS[0] :], scales[ROWS[0] :], x_second),
        )
    )
    assert torch.equal(out, expected)


def test_rejects_a_short_activation_buffer() -> None:
    """A grouped call may read any matrix's vector, so the buffer must hold one of them."""
    weights = torch.zeros((8, WIDTH), device="cuda", dtype=torch.int8)
    scales = torch.ones(8, device="cuda")
    out = torch.empty(8, device="cuda")
    segment = torch.zeros(8, device="cuda", dtype=torch.int32)
    x_base = torch.zeros(1, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="activation"):
        int8_grouped_gemv_into(weights, scales, torch.zeros(WIDTH // 2, device="cuda"), out, segment, x_base)


def test_rejects_a_segment_table_that_does_not_match_the_rows() -> None:
    """Every row needs one segment entry, or the kernel would read past the table."""
    weights = torch.zeros((8, WIDTH), device="cuda", dtype=torch.int8)
    scales = torch.ones(8, device="cuda")
    out = torch.empty(8, device="cuda")
    x_base = torch.zeros(1, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="segment"):
        int8_grouped_gemv_into(weights, scales, torch.zeros(WIDTH, device="cuda"), out, torch.zeros(7, device="cuda", dtype=torch.int32), x_base)

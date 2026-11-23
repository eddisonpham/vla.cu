"""The grouped int4 GEMV against one launch per matrix, with per-matrix activation bases."""

from __future__ import annotations

import pytest
import torch

from kernels.triton.int4_gemv import int4_gemv
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from vla_cu.quant import pack_int4

ROWS = (8, 12)
WIDTH = 128


def test_matches_one_launch_per_matrix() -> None:
    """A segment table must select each matrix's own weights, scales and activation base."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    first = torch.randn((ROWS[0], WIDTH), device="cuda", generator=generator) * 0.05
    second = torch.randn((ROWS[1], WIDTH), device="cuda", generator=generator) * 0.05
    x_first = torch.randn(WIDTH, device="cuda", generator=generator)
    x_second = torch.randn(WIDTH, device="cuda", generator=generator)

    packed, scales = pack_int4(torch.cat((first, second), dim=0))
    out = torch.empty(sum(ROWS), device="cuda")
    segment = torch.cat(
        (
            torch.zeros(ROWS[0], device="cuda", dtype=torch.int32),
            torch.ones(ROWS[1], device="cuda", dtype=torch.int32),
        )
    )
    x_base = torch.tensor([0, WIDTH], device="cuda", dtype=torch.int32)

    int4_grouped_gemv_into(packed, scales, torch.cat((x_first, x_second)), out, segment, x_base)

    expected = torch.cat(
        (
            int4_gemv(packed[: ROWS[0]], scales[: ROWS[0]], x_first),
            int4_gemv(packed[ROWS[0] :], scales[ROWS[0] :], x_second),
        )
    )
    assert torch.equal(out, expected)


def test_rejects_a_segment_table_that_does_not_match_the_rows() -> None:
    """Every packed row needs one segment entry, or the kernel would read past the table."""
    packed, scales = pack_int4(torch.randn((8, WIDTH), device="cuda") * 0.05)
    x = torch.randn(WIDTH, device="cuda")
    out = torch.empty(8, device="cuda")
    segment = torch.zeros(7, device="cuda", dtype=torch.int32)
    x_base = torch.zeros(1, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError):
        int4_grouped_gemv_into(packed, scales, x, out, segment, x_base)


def test_rejects_float_packed_weights() -> None:
    """Reinterpreting fp32 bytes as packed nibbles would produce garbage, not an error."""
    packed = torch.zeros((8, WIDTH // 2), device="cuda", dtype=torch.float32)
    scales = torch.ones((8, WIDTH // 64), device="cuda")
    x = torch.zeros(WIDTH, device="cuda")
    out = torch.empty(8, device="cuda")
    segment = torch.zeros(8, device="cuda", dtype=torch.int32)
    x_base = torch.zeros(1, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError):
        int4_grouped_gemv_into(packed, scales, x, out, segment, x_base)

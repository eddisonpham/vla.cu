"""The int8 GEMM against the int8 GEMV loop and a dense reference."""

from __future__ import annotations

import pytest
import torch

from kernels.triton.int8_gemm import (
    LONG_TILE,
    SHARED_LIMIT_BYTES,
    TILES,
    int8_gemm,
    int8_gemm_into,
    shared_memory_bytes,
    tile_for,
)
from kernels.triton.int8_gemv import int8_gemv_into
from vla_cu.quant import quantize_int8

SHAPES = ((1, 64, 64), (5, 96, 128), (50, 720, 720), (64, 2560, 960))


def reference(tokens: int, n_out: int, n_in: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return int8 codes, scales, activations and the fp32 answer for one shape."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((n_out, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
    x = torch.randn((tokens, n_in), device="cuda", generator=generator)
    codes, scales = quantize_int8(w)
    dense = codes.to(torch.float32) * scales[:, None]
    return codes, scales, x, x @ dense.transpose(0, 1)


@pytest.mark.parametrize(("tokens", "n_out", "n_in"), SHAPES)
def test_matches_the_dense_reference(tokens: int, n_out: int, n_in: int) -> None:
    """The GEMM must agree with the exact dequantized product of the same codes."""
    codes, scales, x, expected = reference(tokens, n_out, n_in)

    out = int8_gemm(codes, scales, x)

    similarity = torch.nn.functional.cosine_similarity(out.flatten(), expected.flatten(), dim=0)
    assert similarity.item() > 0.9995
    assert torch.allclose(out, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(("tokens", "n_out", "n_in"), SHAPES)
def test_matches_the_gemv_loop(tokens: int, n_out: int, n_in: int) -> None:
    """Both paths hold the same codes and row scales, so they must agree closely."""
    codes, scales, x, _ = reference(tokens, n_out, n_in)
    loop = torch.empty((tokens, n_out), device="cuda")
    for token in range(tokens):
        int8_gemv_into(codes, scales, x[token], loop[token])

    out = int8_gemm(codes, scales, x)

    similarity = torch.nn.functional.cosine_similarity(out.flatten(), loop.flatten(), dim=0)
    assert similarity.item() > 0.9999


def test_every_menu_tile_fits_shared_memory() -> None:
    """A menu entry that stages more than the GPU has would fail at launch, not in review."""
    assert all(shared_memory_bytes(tile) <= SHARED_LIMIT_BYTES for tile in TILES)


def test_tile_for_gives_a_long_chunk_enough_programs() -> None:
    """A 50 token chunk on a 16 row tile gives four row blocks instead of one padded one."""
    assert tile_for(50) == LONG_TILE
    assert [tile_for(tokens)[0] for tokens in (1, 16, 17, 200)] == [16, 16, 32, 16]


def test_rejects_a_shape_mismatch() -> None:
    """The activation width must equal the code row length."""
    codes, scales = quantize_int8(torch.zeros((4, 64), device="cuda"))
    out = torch.zeros((2, 4), device="cuda")

    with pytest.raises(ValueError):
        int8_gemm_into(codes, scales, torch.zeros((2, 32), device="cuda"), out)

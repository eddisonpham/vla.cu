"""Tests for the int4 GEMM against the GEMV loop and a dense reference."""

from __future__ import annotations

import pytest
import torch

from kernels.triton.int4_gemm import (
    SHARED_LIMIT_BYTES,
    TILES,
    block_m_for,
    int4_gemm,
    int4_gemm_into,
    int4_stream_k_into,
    shared_memory_bytes,
    stream_k_buffers,
    stream_k_plan,
    tile_for,
)
from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.quant import dequantize_int4, pack_int4

SHAPES = ((1, 64, 64, 64), (3, 17, 128, 16), (50, 1920, 720, 16), (64, 2560, 960, 64))


def reference(tokens: int, n_out: int, n_in: int, group: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed weights, scales, activations and the fp32 answer for one shape."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((n_out, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
    x = torch.randn((tokens, n_in), device="cuda", generator=generator)
    packed, scales = pack_int4(w, group_size=group)
    dense = dequantize_int4(packed, scales, group_size=group)
    return packed, scales, x, x @ dense.transpose(0, 1)


@pytest.mark.parametrize(("tokens", "n_out", "n_in", "group"), SHAPES)
def test_matches_dense_reference(tokens: int, n_out: int, n_in: int, group: int) -> None:
    """The fused GEMM must agree with fp32 weights to within one fp16 rounding."""
    packed, scales, x, expected = reference(tokens, n_out, n_in, group)

    out = int4_gemm(packed, scales, x, group_size=group)

    similarity = torch.nn.functional.cosine_similarity(out.flatten(), expected.flatten(), dim=0)
    assert similarity.item() > 0.9995
    assert torch.allclose(out, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(("tokens", "n_out", "n_in", "group"), SHAPES)
def test_matches_gemv_loop(tokens: int, n_out: int, n_in: int, group: int) -> None:
    """Both low-bit paths dequantize the same weights, so they must agree closely."""
    packed, scales, x, _ = reference(tokens, n_out, n_in, group)
    loop = torch.empty((tokens, n_out), device="cuda")
    for token in range(tokens):
        int4_gemv_into(packed, scales, x[token], loop[token], group_size=group)

    out = int4_gemm(packed, scales, x, group_size=group)

    similarity = torch.nn.functional.cosine_similarity(out.flatten(), loop.flatten(), dim=0)
    assert similarity.item() > 0.9999


def test_block_m_rounds_up_to_a_tensor_core_tile() -> None:
    """Rows are padded to a power of two, with 16 as the tensor core floor."""
    assert [block_m_for(m) for m in (1, 15, 16, 17, 50, 64)] == [16, 16, 16, 32, 64, 64]


def test_rejects_step_that_breaks_the_group() -> None:
    """A step must hold whole groups, or a group would straddle two loop iterations."""
    w = torch.zeros((4, 128), device="cuda")
    packed, scales = pack_int4(w, group_size=64)
    x = torch.zeros((2, 128), device="cuda")
    out = torch.zeros((2, 4), device="cuda")

    with pytest.raises(ValueError):
        int4_gemm_into(packed, scales, x, out, group_size=64, tile=(16, 64, 48, 2))


def test_every_tile_preset_fits_shared_memory() -> None:
    """A menu entry that stages more than the GPU has would fail at launch, not in review."""
    assert all(shared_memory_bytes(tile) <= SHARED_LIMIT_BYTES for tile in TILES)


def test_tile_for_covers_the_token_count_with_the_smallest_row_tile() -> None:
    """A chunk pads to the next preset, and a long one keeps the last preset of its family."""
    assert [tile_for(tokens, 1920)[0] for tokens in (1, 16, 17, 50, 64, 200)] == [16, 16, 32, 64, 64, 64]
    assert [tile_for(tokens, 720)[0] for tokens in (1, 17, 50, 200)] == [16, 32, 32, 32]
    assert [tile_for(tokens, 1920)[1] for tokens in (1, 50)] == [128, 128]
    assert [tile_for(tokens, 720)[1] for tokens in (1, 50)] == [64, 64]


def split_buffers(tokens: int, n_out: int, splits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the partials and the zeroed arrival counters a one launch split needs."""
    block_m, block_n, _, _ = tile_for(tokens, n_out)
    tiles = -(-tokens // block_m) * -(-n_out // block_n)
    return torch.empty((splits, tokens, n_out), device="cuda"), torch.zeros(tiles, device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("splits", (2, 4))
def test_stream_k_matches_the_single_launch(splits: int) -> None:
    """The in launch recombine must sum in a fixed order and repeat bit for bit."""
    packed, scales, x, expected = reference(50, 720, 720, 16)
    whole = int4_gemm(packed, scales, x, group_size=16)
    partial, counters = split_buffers(50, 720, splits)
    split = torch.empty((50, 720), device="cuda")

    int4_gemm_into(packed, scales, x, split, group_size=16, splits=splits, partial=partial, counters=counters)
    first = split.clone()
    int4_gemm_into(packed, scales, x, split, group_size=16, splits=splits, partial=partial, counters=counters)

    assert torch.equal(split, first)
    assert torch.allclose(split, whole, atol=1e-3, rtol=1e-3)
    similarity = torch.nn.functional.cosine_similarity(split.flatten(), expected.flatten(), dim=0)
    assert similarity.item() > 0.9995


def test_split_k_needs_a_workspace() -> None:
    """A split launch writes partials, so a missing buffer must fail before it launches."""
    packed, scales, x, _ = reference(50, 720, 720, 16)
    out = torch.empty((50, 720), device="cuda")

    with pytest.raises(ValueError, match="partial"):
        int4_gemm_into(packed, scales, x, out, group_size=16, splits=2)


def test_stream_k_survives_a_cuda_graph() -> None:
    """A split launch uses atomics and counters, so it must still capture and replay exactly."""
    packed, scales, x, _ = reference(50, 720, 720, 16)
    partial, counters = split_buffers(50, 720, 2)
    out = torch.empty((50, 720), device="cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        int4_gemm_into(packed, scales, x, out, group_size=16, splits=2, partial=partial, counters=counters)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()

    expected = int4_gemm(packed, scales, x, group_size=16)
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3)


def test_split_k_needs_counters() -> None:
    """The last arrival is found through the counters, so a missing table must fail early."""
    packed, scales, x, _ = reference(50, 720, 720, 16)
    partial, _ = split_buffers(50, 720, 2)
    out = torch.empty((50, 720), device="cuda")

    with pytest.raises(ValueError, match="counters"):
        int4_gemm_into(packed, scales, x, out, group_size=16, splits=2, partial=partial)


@pytest.mark.parametrize("waves", (1, 2, 64))
def test_balanced_stream_k_matches_the_single_launch(waves: int) -> None:
    """A contiguous work slice must match the single launch, repeat bit for bit and reset its counters."""
    packed, scales, x, expected = reference(50, 720, 720, 16)
    whole = int4_gemm(packed, scales, x, group_size=16)
    partial, counters, _ = stream_k_buffers(50, 720, 720)
    out = torch.empty((50, 720), device="cuda")

    int4_stream_k_into(packed, scales, x, out, group_size=16, waves=waves, partial=partial, counters=counters)
    first = out.clone()
    int4_stream_k_into(packed, scales, x, out, group_size=16, waves=waves, partial=partial, counters=counters)

    assert torch.equal(out, first)
    assert torch.allclose(out, whole, atol=1e-3, rtol=1e-3)
    assert int(counters.sum()) == 0
    similarity = torch.nn.functional.cosine_similarity(out.flatten(), expected.flatten(), dim=0)
    assert similarity.item() > 0.9995


def test_balanced_stream_k_balances_a_wide_shape() -> None:
    """The wide expert shape has fewer tiles than SMs, which is the imbalance this schedule removes."""
    packed, scales, x, _ = reference(50, 1920, 720, 16)
    _, units, programs, tiles, k_tiles = stream_k_plan(50, 1920, 720)
    partial, counters, _ = stream_k_buffers(50, 1920, 720)
    out = torch.empty((50, 1920), device="cuda")

    int4_stream_k_into(packed, scales, x, out, group_size=16, partial=partial, counters=counters)

    assert tiles == 15 and units == tiles * k_tiles
    assert programs > tiles
    assert torch.allclose(out, int4_gemm(packed, scales, x, group_size=16), atol=1e-3, rtol=1e-3)


def test_balanced_stream_k_survives_a_cuda_graph() -> None:
    """The handoff uses atomics and counters, so it must still capture and replay exactly."""
    packed, scales, x, _ = reference(50, 720, 720, 16)
    partial, counters, _ = stream_k_buffers(50, 720, 720)
    out = torch.empty((50, 720), device="cuda")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        int4_stream_k_into(packed, scales, x, out, group_size=16, partial=partial, counters=counters)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.allclose(out, int4_gemm(packed, scales, x, group_size=16), atol=1e-3, rtol=1e-3)


def test_rejects_shape_mismatch() -> None:
    """The activation width must equal twice the packed row length."""
    w = torch.zeros((4, 64), device="cuda")
    packed, scales = pack_int4(w, group_size=64)

    with pytest.raises(ValueError):
        int4_gemm(packed, scales, torch.zeros((2, 32), device="cuda"), group_size=64)

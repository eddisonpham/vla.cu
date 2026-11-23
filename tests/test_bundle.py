"""The packed-weight bundle: bit-exact round trip, and refusal of incompatible files."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kernels.triton.int8_gemv import int8_gemv
from vla_cu.bundle import Bundle, PackedMatrix, load_bundle, pack_matrix, save_bundle


def test_round_trip_is_bit_exact(tmp_path: Path) -> None:
    """A saved matrix must come back with the same codes, the same scales and its metadata."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((16, 128), device="cuda", generator=generator) * 0.05
    bundle = Bundle({"int4": pack_matrix(w, bits=4), "int8": pack_matrix(w, bits=8)}, {"source": "test"})

    size = save_bundle(tmp_path / "weights.bin", bundle)
    loaded = load_bundle(tmp_path / "weights.bin")

    assert size == (tmp_path / "weights.bin").stat().st_size
    assert loaded.metadata == {"source": "test"}
    assert set(loaded.matrices) == {"int4", "int8"}
    for name, matrix in bundle.matrices.items():
        assert torch.equal(loaded.matrices[name].data, matrix.data.cpu())
        assert torch.equal(loaded.matrices[name].scales, matrix.scales.cpu())
        assert loaded.matrices[name].bits == matrix.bits


def test_loaded_matrix_still_runs_the_kernel(tmp_path: Path) -> None:
    """A bundle that loads but multiplies differently is worse than no bundle."""
    generator = torch.Generator(device="cuda").manual_seed(1)
    w = torch.randn((24, 128), device="cuda", generator=generator) * 0.05
    x = torch.randn(128, device="cuda", generator=generator)
    matrix = pack_matrix(w, bits=8)
    save_bundle(tmp_path / "one.bin", Bundle({"up": matrix}))
    loaded = load_bundle(tmp_path / "one.bin").matrices["up"]

    assert torch.equal(int8_gemv(loaded.data.cuda(), loaded.scales.cuda(), x), int8_gemv(matrix.data, matrix.scales, x))


def test_rejects_an_unknown_format_version(tmp_path: Path) -> None:
    """A future archive must be refused by version, not decoded as if it were current."""
    path = tmp_path / "future.bin"
    torch.save({"format_version": 99, "metadata": "{}"}, path)
    with pytest.raises(ValueError, match="version"):
        load_bundle(path)


def test_rejects_a_matrix_whose_shapes_do_not_match_its_format() -> None:
    """The declaration is the contract, so a mismatched tensor must fail at construction."""
    with pytest.raises(ValueError):
        PackedMatrix(data=torch.zeros((4, 32), dtype=torch.uint8), scales=torch.zeros((4, 2)), bits=4, group_size=64, rows=4, cols=128)


def test_rejects_a_file_with_no_matrices(tmp_path: Path) -> None:
    """An empty archive would load as a model with no weights, so it is an error."""
    path = tmp_path / "empty.bin"
    save_bundle(path, Bundle({"only": pack_matrix(torch.zeros((4, 64), dtype=torch.float32))}))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    stripped = {key: value for key, value in payload.items() if not key.endswith(".manifest")}
    torch.save(stripped, path)
    with pytest.raises(ValueError, match="no matrices"):
        load_bundle(path)

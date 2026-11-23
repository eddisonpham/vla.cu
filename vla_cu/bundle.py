"""Versioned files for packed weights: quantize once, load without the fp32 source.

Concept: packing is a one time conversion, and a saved bundle has to carry its format so
a loader refuses an incompatible one instead of decoding garbage. The archive holds raw
tensors plus a JSON manifest, is written with `torch.save` and read with
`weights_only=True`, so it needs no new dependency and cannot execute code on load.

Shapes are described once, by `PackedMatrix`, and checked in both directions.

Task: python -m vla_cu bundle
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8

FORMAT_VERSION = 1
BITS = (4, 8)
SCALE_DTYPES = (torch.float32, torch.float16)


@dataclass(frozen=True)
class PackedMatrix:
    """One quantized matrix and the layout needed to interpret it.

    `bits` 4 stores uint8 [rows, cols // 2] with scales [rows, cols // group_size]; bits
    8 stores int8 [rows, cols] with one scale per row.
    """

    data: Tensor
    scales: Tensor
    bits: int
    group_size: int
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.bits not in BITS:
            raise ValueError(f"expected 4 or 8 bits, got {self.bits}")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"expected a non-empty matrix, got {self.rows} x {self.cols}")
        if self.bits == 4:
            if self.cols % self.group_size:
                raise ValueError(f"{self.cols} columns do not divide into groups of {self.group_size}")
            expected_data = (self.rows, self.cols // 2)
            expected_scales = (self.rows, self.cols // self.group_size)
            data_dtype = torch.uint8
        else:
            expected_data = (self.rows, self.cols)
            expected_scales = (self.rows,)
            data_dtype = torch.int8
        if self.data.dtype != data_dtype or tuple(self.data.shape) != expected_data:
            raise ValueError(f"expected {data_dtype} {expected_data}, got {self.data.dtype} {tuple(self.data.shape)}")
        if self.scales.dtype not in SCALE_DTYPES or tuple(self.scales.shape) != expected_scales:
            raise ValueError(f"expected {expected_scales} scales, fp32 or fp16, got {self.scales.dtype} {tuple(self.scales.shape)}")

    @property
    def nbytes(self) -> int:
        """Return the stored bytes of the codes and the scale table."""
        return self.data.numel() * self.data.element_size() + self.scales.numel() * self.scales.element_size()


@dataclass(frozen=True)
class Bundle:
    """A named set of packed matrices plus free-form string metadata."""

    matrices: dict[str, PackedMatrix]
    metadata: dict[str, str] = field(default_factory=dict)

    def nbytes(self) -> int:
        """Return the stored bytes of every matrix in the bundle."""
        return sum(matrix.nbytes for matrix in self.matrices.values())


def pack_matrix(w: Tensor, bits: int = 8, group_size: int = GROUP_SIZE) -> PackedMatrix:
    """Quantize fp32 weights [rows, cols] at `bits` and describe the result."""
    if w.dim() != 2:
        raise ValueError(f"expected a 2D weight tensor, got {tuple(w.shape)}")
    if bits == 4:
        data, scales = pack_int4(w, group_size=group_size)
    else:
        data, scales = quantize_int8(w)
    return PackedMatrix(data=data, scales=scales, bits=bits, group_size=group_size, rows=w.shape[0], cols=w.shape[1])


def _check_name(name: str) -> None:
    """Reject a matrix name that cannot form a manifest key."""
    allowed = set("._-")
    if not name or not all(character.isalnum() or character in allowed for character in name):
        raise ValueError(f"expected an alphanumeric matrix name, got {name!r}")


def save_bundle(path: str | Path, bundle: Bundle) -> int:
    """Write `bundle` to `path`, creating its parent directory, and return the file size.

    Every tensor is moved to CPU first, so the archive is portable and loads without a
    GPU. The manifest carries the format version, the bit width and the shapes.
    """
    if not bundle.matrices:
        raise ValueError("expected at least one matrix")
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "metadata": json.dumps(bundle.metadata, sort_keys=True),
    }
    for name, matrix in bundle.matrices.items():
        _check_name(name)
        payload[f"{name}.data"] = matrix.data.detach().cpu()
        payload[f"{name}.scales"] = matrix.scales.detach().cpu()
        payload[f"{name}.manifest"] = json.dumps(
            {"bits": matrix.bits, "group_size": matrix.group_size, "rows": matrix.rows, "cols": matrix.cols},
            sort_keys=True,
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target.stat().st_size


def load_bundle(path: str | Path) -> Bundle:
    """Read a bundle to CPU tensors, checking its version and every layout before returning it."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a bundle dictionary, got {type(payload).__name__}")
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"expected format version {FORMAT_VERSION}, got {version}")
    metadata = json.loads(payload["metadata"])
    matrices: dict[str, PackedMatrix] = {}
    for key in payload:
        if not key.endswith(".manifest"):
            continue
        name = key[: -len(".manifest")]
        manifest = json.loads(payload[key])
        matrices[name] = PackedMatrix(
            data=payload[f"{name}.data"],
            scales=payload[f"{name}.scales"],
            bits=int(manifest["bits"]),
            group_size=int(manifest["group_size"]),
            rows=int(manifest["rows"]),
            cols=int(manifest["cols"]),
        )
    if not matrices:
        raise ValueError("the bundle holds no matrices")
    return Bundle(matrices=matrices, metadata=metadata)

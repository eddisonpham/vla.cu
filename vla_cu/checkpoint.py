"""Read single tensors out of a .safetensors file without loading the whole checkpoint.

Concept: the format is an 8 byte little-endian header length, a JSON header that gives
each tensor a dtype, a shape and a byte range, then the raw bytes. One seek and one read
bring one matrix into memory, which is how an 865 MiB checkpoint becomes three layers
without a new dependency.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

DTYPES: dict[str, torch.dtype] = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


@dataclass(frozen=True)
class TensorInfo:
    """Where one tensor lives in the file, and how to read its bytes."""

    dtype: torch.dtype
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        """Return the stored bytes of this tensor."""
        return self.end - self.begin


@dataclass(frozen=True)
class Checkpoint:
    """An indexed safetensors file: the header length plus one entry per tensor."""

    path: Path
    header_bytes: int
    tensors: dict[str, TensorInfo]

    def info(self, name: str) -> TensorInfo:
        """Return the table entry for `name`, or raise a KeyError listing the count."""
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{name} is not in {self.path.name}, which holds {len(self.tensors)} tensors") from None

    def read(self, name: str) -> Tensor:
        """Return one tensor as an owned CPU tensor of its stored dtype."""
        info = self.info(name)
        with self.path.open("rb") as handle:
            handle.seek(8 + self.header_bytes + info.begin)
            buffer = handle.read(info.nbytes)
        if len(buffer) != info.nbytes:
            raise ValueError(f"{name} ended after {len(buffer)} of {info.nbytes} bytes")
        tensor = torch.frombuffer(bytearray(buffer), dtype=info.dtype).reshape(info.shape)
        return tensor.clone()


def open_checkpoint(path: str | Path) -> Checkpoint:
    """Parse the header of a safetensors file and return its tensor table.

    Only the header is read, so this is cheap on a checkpoint that does not fit in RAM.
    Each range is checked against its shape and dtype, because a wrong offset would
    otherwise decode as plausible but wrong numbers.
    """
    target = Path(path)
    with target.open("rb") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_bytes))

    tensors: dict[str, TensorInfo] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"{name} has unsupported dtype {entry['dtype']}")
        begin, end = (int(value) for value in entry["data_offsets"])
        info = TensorInfo(dtype, tuple(int(size) for size in entry["shape"]), begin, end)
        expected = info.dtype.itemsize * torch.tensor(info.shape).prod().item()
        if info.nbytes != expected:
            raise ValueError(f"{name} spans {info.nbytes} bytes, but its shape needs {expected}")
        tensors[name] = info
    return Checkpoint(path=target, header_bytes=header_bytes, tensors=tensors)

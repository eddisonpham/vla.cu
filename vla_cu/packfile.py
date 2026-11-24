"""One file holding the packed tower and the 16 backbone layers, so a new process starts warm.

Concept: a prefix build reads 172.87 MB of bf16 tower weights and 315 MB of bf16 text
weights, then quantizes and casts them, which is about 0.85 s of the first call in a
process. This file stores what the two caches hold, already quantized and already fp32, so a
later process memory maps it and copies each tensor to the GPU without a read, a cast or a
quantize. The round trip is bit exact, so a pack file cannot move a number.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from vla_cu.checkpoint import Checkpoint

PACK_BITS = 8
PACKED_LAYERS = 16

_LOADED: dict[Path, dict[str, Tensor]] = {}


def pack_path(checkpoint: Checkpoint) -> Path:
    """Return the pack file beside the checkpoint: model.safetensors -> model.pack.int8.pt."""
    return checkpoint.path.with_name(f"{checkpoint.path.stem}.pack.int{PACK_BITS}.pt")


def save_pack(path: Path, tensors: dict[str, Tensor]) -> int:
    """Write `tensors` to one file and return its size in bytes.

    Every tensor moves to the CPU first, so the file is a plain CPU archive that a later
    process can map instead of read. An existing file is removed, because Windows refuses to
    open a mapped file for writing.
    """
    if path.exists():
        path.unlink()
    torch.save({name: tensor.detach().to("cpu") for name, tensor in tensors.items()}, path)
    return path.stat().st_size


def load_pack(path: Path) -> dict[str, Tensor]:
    """Return the file's tensors memory mapped, mapping the file at most once per process."""
    held = _LOADED.get(path)
    if held is None:
        _LOADED[path] = held = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    return held


def forget_packs() -> None:
    """Drop the mapped archives, so the next call maps the file again, as a new process would."""
    _LOADED.clear()

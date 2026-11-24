"""Build one pack file for the tower and the 16 text layers, then time a cold start against it.

Concept: a new process pays 172.87 MB of bf16 tower plus 315 MB of bf16 text weights before
it can build a prefix, which is the 0.85 s first call. The pack file holds exactly what the
two caches hold, so the correctness check is a bit exact round trip and the number is wall
clock, taken with both caches cleared in this process. The file is warm in the page cache
here because this run wrote it; a first cold touch of it is a disk read and costs seconds.

Task: python -m vla_cu prepack
"""

from __future__ import annotations

import time
from typing import Callable

import torch
from torch import Tensor

from bench.real_layers import CHECKPOINT
from vla_cu.bench_utils import enforce_memory_budget, warm_gpu
from vla_cu.checkpoint import Checkpoint, open_checkpoint
from vla_cu.packfile import PACK_BITS, forget_packs, pack_path, save_pack
from vla_cu.prefix import (
    LAYERS,
    backbone_weights,
    cache_from_rows,
    forget_backbone,
    prefix_inputs,
    weights_state,
)
from vla_cu.vision import forget_tower, tower_for

FRAMES = 3


def elapsed(fn: Callable[[], None]) -> float:
    """Return wall clock seconds for one call, synchronized on both sides."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def build(checkpoint: Checkpoint) -> tuple[object, list[list[Tensor]], float]:
    """Clear both caches, then time one tower build and one backbone read."""
    forget_tower()
    forget_backbone()
    forget_packs()
    torch.cuda.empty_cache()
    seconds = elapsed(lambda: (tower_for(checkpoint, bits=PACK_BITS), backbone_weights(checkpoint, layers=LAYERS)))
    return tower_for(checkpoint, bits=PACK_BITS), backbone_weights(checkpoint, layers=LAYERS), seconds


def main() -> int:
    """Write the pack file, then compare a cold start from the checkpoint with one from the pack."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    path = pack_path(checkpoint)
    forget_packs()
    path.unlink(missing_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    frame = torch.rand((3, 512, 512), device="cuda", generator=generator) * 2 - 1
    print(f"GPU {torch.cuda.get_device_name(0)} | {checkpoint.path.name} | {LAYERS} text layers | tower at int{PACK_BITS}")
    print("every read here is warm in the OS file cache, so the disk is not part of the number")
    print("any existing pack is deleted first, so the checkpoint column cannot read one")

    build(checkpoint)
    tower, text, checkpoint_seconds = build(checkpoint)
    rows_before = tower.rows(frame)
    cache_before = cache_from_rows(checkpoint, prefix_inputs(checkpoint), layers=LAYERS)
    tensors = {f"tower.{name}": value for name, value in tower.state().items()}
    tensors.update(weights_state(text))
    write_seconds = elapsed(lambda: save_pack(path, tensors))
    n_bytes = path.stat().st_size
    print(f"pack written in {write_seconds * 1e3:.2f} ms, {n_bytes / 1e6:.2f} MB, {len(tensors)} tensors")

    del tensors, tower, text
    build(checkpoint)
    tower, text, packed_seconds = build(checkpoint)
    rows_after = tower.rows(frame)
    cache_after = cache_from_rows(checkpoint, prefix_inputs(checkpoint), layers=LAYERS)
    held = cache_before[0] + cache_before[1]
    reloaded = cache_after[0] + cache_after[1]
    exact = all(torch.equal(left, right) for left, right in zip(held, reloaded)) and torch.equal(rows_before, rows_after)
    print()
    print(f"{'second read':>16}  {'ms':>9}")
    print(f"{'from checkpoint':>16}  {checkpoint_seconds * 1e3:>9.2f}")
    print(f"{'from pack':>16}  {packed_seconds * 1e3:>9.2f}")
    print(f"{'speedup':>16}  {checkpoint_seconds / packed_seconds:>8.2f}x")
    print("both columns are a second read in this process, so both files were in the page cache")
    print(f"tower bytes {tower.bytes / 1e6:.2f} MB | text bytes {sum(t.numel() * 4 for layer in text for t in layer) / 1e6:.2f} MB")
    print(f"round trip bit exact: {'yes' if exact else 'NO'}, {len(held)} cache tensors and one 64 x 960 row block compared")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

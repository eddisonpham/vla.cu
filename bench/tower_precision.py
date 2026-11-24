"""Where the 241 row prefill spends its time: the SigLIP tower at 32, 8 and 4 bits.

Concept: the tower is 197 of the checkpoint's 500 tensors and it runs once per frame, so
this task separates the weight read from the patch compute and asks whether packing the
projections moves the number that matters. Wall clock is used and not CUDA events, because
a prefill that waits on the disk would read as zero GPU time.

Task: python -m vla_cu tower
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import torch

from bench.real_layers import CHECKPOINT, cosine, relative_error
from vla_cu.bench_utils import enforce_memory_budget, warm_gpu
from vla_cu.checkpoint import open_checkpoint
from vla_cu.prefix import FULL_TOKENS, cache_from_rows, full_prefix_inputs
from vla_cu.vision import BASE, Tower

BITS = (32, 8, 4)
FRAMES = 3
REPEATS = 3


def elapsed(fn: Callable[[], None], repeats: int = REPEATS) -> float:
    """Return the median wall clock seconds per call, synchronized on both sides."""
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> int:
    """Time the tower's read and its patch compute at three precisions, then the whole prefill."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    generator = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.rand((FRAMES, 3, 512, 512), device="cuda", generator=generator) * 2 - 1
    state = torch.zeros(32, device="cuda")
    print(f"GPU {torch.cuda.get_device_name(0)} | {FRAMES} frames of 512, 1024 patches, 12 layers, {FULL_TOKENS} prefix rows")
    stored = sum(info.nbytes for name, info in checkpoint.tensors.items() if name.startswith(f"{BASE}."))
    print(f"wall clock, median of {REPEATS} calls, file reads included; the tower is {stored / 1e6:.2f} MB of bf16 in the checkpoint")

    rows = {}
    towers = {}
    builds = {}
    for bits in BITS:
        tower = Tower(checkpoint, bits)
        torch.cuda.synchronize()
        builds[bits] = elapsed(lambda: Tower(checkpoint, bits))
        towers[bits] = tower
        frames_seconds = elapsed(lambda: [tower.rows(frame) for frame in frames])
        rebuild_seconds = elapsed(lambda: [Tower(checkpoint, bits).rows(frame) for frame in frames])
        rows[bits] = torch.stack([tower.rows(frame) for frame in frames])
        cosine_value = cosine(rows[bits].reshape(-1), rows[32].reshape(-1))
        error = relative_error(rows[bits].reshape(-1), rows[32].reshape(-1))
        label = f"int{bits}" if bits < 32 else "fp32"
        print(
            f"{label:>5}  build {builds[bits] * 1e3:8.2f} ms  {FRAMES} frames {frames_seconds * 1e3:8.2f} ms  "
            f"rebuilt {rebuild_seconds * 1e3:8.2f} ms  {tower.bytes / 1e6:7.2f} MB  cos {cosine_value:8.5f}  "
            f"rel L2 {error:8.2e}"
        )

    prefix_rows, visible = full_prefix_inputs(checkpoint, frames, state, tower_bits=8)
    text_seconds = elapsed(lambda: cache_from_rows(checkpoint, prefix_rows, visible))
    print()
    print(f"text cache, 16 backbone layers over {FULL_TOKENS} rows: {text_seconds * 1e3:.2f} ms, weights read included")
    print(f"{'bits':>5}  {'vision':>10}  {'text':>10}  {'prefill':>10}  {'cos':>8}")
    for bits in BITS:
        shared = builds[bits] + elapsed(lambda: [towers[bits].rows(frame) for frame in frames])
        print(
            f"{bits:>5}  {shared * 1e3:>10.2f}  {text_seconds * 1e3:>10.2f}  "
            f"{(shared + text_seconds) * 1e3:>10.2f}  {cosine(rows[bits].reshape(-1), rows[32].reshape(-1)):>8.5f}"
        )
    print()
    print("the read is the same number of checkpoint bytes at every width, so packing saves compute, not traffic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

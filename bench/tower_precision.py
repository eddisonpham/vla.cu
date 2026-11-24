"""Where the 241 row prefill spends its time, and what the tower's precision and group buy.

Concept: the tower is 197 of the checkpoint's 500 tensors and it runs once per frame, so
this task separates the weight read from the patch compute, then sweeps the 4-bit group
size, which is the one accuracy lever a packed weight has left. Wall clock is used and not
CUDA events, because a prefill that waits on the disk would read as zero GPU time.

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
from vla_cu.quant import GROUP_SIZE
from vla_cu.vision import BASE, Tower, tower_for

VARIANTS = (
    ("fp32", 32, GROUP_SIZE),
    ("int8", 8, GROUP_SIZE),
    ("int4 g64", 4, 64),
    ("int4 g32", 4, 32),
    ("int4 g16", 4, 16),
)
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
    """Time the tower at five precisions and group sizes, then the whole prefill."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    generator = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.rand((FRAMES, 3, 512, 512), device="cuda", generator=generator) * 2 - 1
    state = torch.zeros(32, device="cuda")
    print(f"GPU {torch.cuda.get_device_name(0)} | {FRAMES} frames of 512, 1024 patches, 12 layers, {FULL_TOKENS} prefix rows")
    stored = sum(info.nbytes for name, info in checkpoint.tensors.items() if name.startswith(f"{BASE}."))
    print(f"wall clock, median of {REPEATS} calls, file reads included; the tower is {stored / 1e6:.2f} MB of bf16 in the checkpoint")
    print(f"patch cosine is one frame at the 1024 patches, row cosine is all {FRAMES} frames at the 64 connector rows")

    rows: dict[str, torch.Tensor] = {}
    patches: dict[str, torch.Tensor] = {}
    towers: dict[str, Tower] = {}
    builds: dict[str, float] = {}
    frame_times: dict[str, float] = {}
    print()
    print(f"{'variant':>9}  {'build':>9}  {'3 frames':>9}  {'rebuilt':>9}  {'MB':>7}  {'patch cos':>9}  {'row cos':>8}  {'rel L2':>8}")
    for label, bits, group in VARIANTS:
        tower = Tower(checkpoint, bits, group_size=group)
        torch.cuda.synchronize()
        builds[label] = elapsed(lambda: Tower(checkpoint, bits, group_size=group))
        frame_times[label] = elapsed(lambda: [tower.rows(frame) for frame in frames])
        rebuild_seconds = elapsed(lambda: [Tower(checkpoint, bits, group_size=group).rows(frame) for frame in frames])
        towers[label] = tower
        rows[label] = torch.stack([tower.rows(frame) for frame in frames])
        patches[label] = tower.hidden(frames[0])
        print(
            f"{label:>9}  {builds[label] * 1e3:>8.2f}ms  {frame_times[label] * 1e3:>8.2f}ms  {rebuild_seconds * 1e3:>8.2f}ms  "
            f"{tower.bytes / 1e6:>7.2f}  {cosine(patches[label].reshape(-1), patches['fp32'].reshape(-1)):>9.5f}  "
            f"{cosine(rows[label].reshape(-1), rows['fp32'].reshape(-1)):>8.5f}  "
            f"{relative_error(rows[label].reshape(-1), rows['fp32'].reshape(-1)):>8.2e}"
        )

    cold = elapsed(lambda: tower_for(checkpoint, 8, group_size=GROUP_SIZE), repeats=1)
    warm = elapsed(lambda: tower_for(checkpoint, 8, group_size=GROUP_SIZE))
    prefix_rows, visible = full_prefix_inputs(checkpoint, frames, state, tower_bits=8)
    text_seconds = elapsed(lambda: cache_from_rows(checkpoint, prefix_rows, visible))
    print()
    print(f"tower cache, int8: first call {cold * 1e3:.2f} ms, later calls in the same process {warm * 1e3:.3f} ms")
    print(f"text cache, 16 backbone layers over {FULL_TOKENS} rows: {text_seconds * 1e3:.2f} ms, weights read included")
    print(f"warm prefill is what an observation pays after the first, with the tower already held")
    print(f"{'variant':>9}  {'frames':>9}  {'text':>9}  {'first':>9}  {'warm':>9}")
    for label, _, _ in VARIANTS:
        frames_seconds = frame_times[label]
        print(
            f"{label:>9}  {frames_seconds * 1e3:>8.2f}ms  {text_seconds * 1e3:>8.2f}ms  "
            f"{(builds[label] + frames_seconds + text_seconds) * 1e3:>8.2f}ms  {(frames_seconds + text_seconds) * 1e3:>8.2f}ms"
        )
    print()
    print("the read is the same number of checkpoint bytes at every width, so packing saves compute, not traffic")
    print("a smaller group costs scale bytes per row and buys accuracy, which is the only lever int4 has left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

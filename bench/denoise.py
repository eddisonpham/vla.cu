"""The 10 step denoising loop with the prefix cache held across steps.

Concept: a policy observation is one prefill and ten expert passes, so the cache is the
fixed cost of a chunk and the per step cost is what a step of denoising costs. This task
builds the 241 row cache once, then replays a captured expert graph ten times with a new
action guess in the input buffer each time, and compares that with rebuilding the cache
every step. Wall clock is used for the build because it waits on the checkpoint; CUDA
events time the replays.

Task: python -m vla_cu denoise
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import torch

from bench.real_layers import CHECKPOINT
from vla_cu.bench_utils import enforce_memory_budget, time_call, warm_gpu
from vla_cu.build import layer_from_checkpoint
from vla_cu.checkpoint import open_checkpoint
from vla_cu.prefix import FULL_TOKENS, TOKENS, cache_from_rows, full_prefix_inputs

LAYERS = 16
STEPS = 10
WARMUP = 5
ITERATIONS = 20


def elapsed(fn: Callable[[], None], repeats: int = 3) -> float:
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
    """Build the prefix once, run ten graphed expert steps, and price the rebuild path."""
    enforce_memory_budget()
    warm_gpu()
    checkpoint = open_checkpoint(CHECKPOINT)
    generator = torch.Generator(device="cuda").manual_seed(0)
    frames = torch.rand((3, 3, 512, 512), device="cuda", generator=generator) * 2 - 1
    state = torch.zeros(32, device="cuda")
    print(f"GPU {torch.cuda.get_device_name(0)} | {STEPS} steps of {TOKENS} action tokens on {FULL_TOKENS} prefix rows")
    print("frames are seeded noise, the state is zeros, and the tower runs at int8")

    def build() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return the 16 layer key and value cache for the full prefix."""
        rows, visible = full_prefix_inputs(checkpoint, frames, state, tower_bits=8)
        return cache_from_rows(checkpoint, rows, visible)

    build_seconds = elapsed(build)
    cached_keys, cached_values = build()
    n_bytes = sum(tensor.numel() * 4 for tensor in cached_keys + cached_values)
    print(f"prefix build {build_seconds * 1e3:.2f} ms once per observation, cache {n_bytes / 1e6:.2f} MB held")

    expert = [
        layer_from_checkpoint(
            checkpoint,
            index,
            tokens=TOKENS,
            down_bits=8,
            part="expert",
            other_tokens=FULL_TOKENS if index % 2 else TOKENS,
        )
        for index in range(LAYERS)
    ]
    hidden = expert[0].hidden_width
    guesses = torch.randn((STEPS, TOKENS, hidden), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    action = torch.empty((TOKENS, hidden), device="cuda")
    first, second = torch.empty_like(action), torch.empty_like(action)
    result = torch.empty_like(action)

    def run_expert() -> None:
        """Run the 16 expert layers from `action` into `result`, crossing on the odd layers."""
        source, other = first, second
        source.copy_(action)
        for index, layer in enumerate(expert):
            if index % 2 == 0:
                layer.into(source, other)
            else:
                layer.into(source, other, cached_keys[index], cached_values[index])
            source, other = other, source
        result.copy_(source)

    def step(index: int) -> None:
        """Write one new action guess into the input buffer and replay the expert."""
        action.copy_(guesses[index])
        graph.replay()

    run_expert()
    torch.cuda.synchronize()
    eager_result = result.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_expert()
    action.zero_()
    graph.replay()
    torch.cuda.synchronize()
    exact = torch.equal(result, eager_result)

    eager_step = time_call(run_expert, WARMUP, ITERATIONS)
    eager_seconds = time_call(lambda: [run_expert() for _ in range(STEPS)], WARMUP, ITERATIONS)
    graph_step = time_call(lambda: step(0), WARMUP, ITERATIONS)
    held_seconds = time_call(lambda: [step(index) for index in range(STEPS)], WARMUP, ITERATIONS)
    rebuild_seconds = STEPS * (build_seconds + graph_step)

    print(f"expert graph bit exact against eager: {'yes' if exact else 'NO'}")
    print()
    print(f"{'path':>22}  {'ms':>9}")
    print(f"{'one step, eager':>22}  {eager_step * 1e3:>9.3f}")
    print(f"{'one step, graph':>22}  {graph_step * 1e3:>9.3f}")
    print(f"{'10 steps, eager':>22}  {eager_seconds * 1e3:>9.3f}")
    print(f"{'10 steps, graph':>22}  {held_seconds * 1e3:>9.3f}")
    print()
    print(f"{'chunk, cache held':>22}  {(build_seconds + held_seconds) * 1e3:>9.3f}  = build once + 10 steps")
    print(f"{'chunk, cache rebuilt':>22}  {rebuild_seconds * 1e3:>9.3f}  = 10 x (build + step), from the measured build")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

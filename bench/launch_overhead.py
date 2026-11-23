"""Measure what one kernel launch costs, from Python and as a CUDA graph node.

At batch size 1 a policy is hundreds of tiny kernels, so launch cost can rival the
compute. One 1-element kernel is timed four ways: a queued Python loop, a loop that
synchronizes every launch, the same loop timed with CUDA events, and a CUDA graph
that holds all 1000 launches in one submission.

Task: python -m vla_cu launch
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from vla_cu.bench_utils import enforce_memory_budget, time_call

N_LAUNCHES = 1000
GRAPH_REPLAYS = 20


@triton.jit
def touch_kernel(ptr) -> None:
    """Read one float, add one, write it back. One program, one warp."""
    tl.store(ptr, tl.load(ptr) + 1.0)


def main() -> int:
    """Print microseconds per launch for the four submission methods."""
    enforce_memory_budget()
    x = torch.zeros(1, device="cuda")

    def launch_once() -> None:
        touch_kernel[(1,)](x)

    launch_once()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(N_LAUNCHES):
        launch_once()
    torch.cuda.synchronize()
    queued_wall_s = (time.perf_counter() - start) / N_LAUNCHES

    start = time.perf_counter()
    for _ in range(N_LAUNCHES):
        launch_once()
        torch.cuda.synchronize()
    round_trip_s = (time.perf_counter() - start) / N_LAUNCHES

    queued_events_s = time_call(launch_once, warmup=10, iterations=N_LAUNCHES)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(N_LAUNCHES):
            launch_once()
    replay_s = time_call(graph.replay, warmup=5, iterations=GRAPH_REPLAYS) / N_LAUNCHES

    rows = (
        ("python loop, queued, one sync", queued_wall_s),
        ("python loop, sync each launch", round_trip_s),
        ("cuda events, same queued loop", queued_events_s),
        ("cuda graph, 1000 nodes replayed", replay_s),
    )

    print(f"GPU {torch.cuda.get_device_name(0)} | {N_LAUNCHES} launches of a 1-element kernel")
    print(f"{'method':<34} {'us/launch':>10} {'launches/s':>12}")
    for name, seconds in rows:
        print(f"{name:<34} {seconds * 1e6:>10.2f} {1.0 / seconds:>12.0f}")

    graph_saving_s = queued_wall_s - replay_s
    print()
    print(f"graph saves {graph_saving_s * 1e6:.2f} us per launch")
    print(f"a 400 kernel chain saves {400 * graph_saving_s * 1e3:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Where a captured Int4Linear call pays, from a tiny layer to a 134 MB matrix.

Concept: a CUDA graph removes host dispatch, which is a fixed cost per call, while the
GEMV body reads the packed weight, which grows with the matrix. Eager and both captured
calls are paired at the same shape, so the ratio separates the two costs and the gap
between a copied input and one bound to the caller's buffer prices the copy. Achieved GB/s
is reported against the packed weight plus scales, the bytes one token must read.

Task: python -m vla_cu graphed
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_pair,
    warm_gpu,
)
from vla_cu.linear import Int4Linear

SHAPES = ((256, 640), (768, 1920), (960, 2560), (4096, 4096), (8192, 4096))
REPEATS = 5
ITERATIONS = 50
WARMUP = 10


class Result(NamedTuple):
    """One shape's paired timing, the bytes it reads and whether replay was exact."""

    eager_ms: float
    copy_ms: float
    bound_ms: float
    ratio: float
    stored_mb: float
    bandwidth: float
    exact: bool


def measure(out_features: int, in_features: int, generator: torch.Generator) -> Result:
    """Time eager against graphed for one [out_features, in_features] weight at one token."""
    weight = torch.randn((out_features, in_features), device="cuda", generator=generator) * (2.0 / in_features**0.5)
    x = torch.randn(in_features, device="cuda", generator=generator)
    layer = Int4Linear(weight)
    copying = layer.graphed()
    bound = layer.graphed(x)
    exact = torch.equal(copying(x), layer(x)) and torch.equal(bound(x), layer(x))

    copy_timing = time_pair(lambda: layer(x), lambda: copying(x), ITERATIONS, REPEATS, WARMUP)
    bound_timing = time_pair(lambda: layer(x), lambda: bound(x), ITERATIONS, REPEATS, WARMUP)
    stored_bytes = layer.packed.numel() + layer.scales.numel() * 4
    return Result(
        eager_ms=bound_timing.a * 1e3,
        copy_ms=copy_timing.b * 1e3,
        bound_ms=bound_timing.b * 1e3,
        ratio=bound_timing.ratio,
        stored_mb=stored_bytes / 1e6,
        bandwidth=bandwidth_gb_s(stored_bytes, bound_timing.a),
        exact=exact,
    )


def main() -> int:
    """Time eager against graphed at every shape and print the ratio and bandwidth."""
    enforce_memory_budget()
    warm_gpu()
    generator = torch.Generator(device="cuda").manual_seed(0)
    print(f"GPU {torch.cuda.get_device_name(0)} | one token | warmup {WARMUP}, repeats {REPEATS}, {ITERATIONS} iterations")
    print(
        f"{'out':>5}  {'in':>5}  {'int4 MB':>8}  {'eager ms':>9}  {'copy ms':>8}  {'bound ms':>8}  "
        f"{'ratio':>6}  {'GB/s':>7}  {'peak':>6}"
    )
    for out_features, in_features in SHAPES:
        result = measure(out_features, in_features, generator)
        print(
            f"{out_features:>5}  {in_features:>5}  {result.stored_mb:>8.2f}  {result.eager_ms:>9.4f}  "
            f"{result.copy_ms:>8.4f}  {result.bound_ms:>8.4f}  {result.ratio:>5.2f}x  "
            f"{result.bandwidth:>7.1f}  {100.0 * fraction_of_peak(result.bandwidth, PEAK_BANDWIDTH_GB_S):>5.1f}%"
        )
        if not result.exact:
            print("  graphed replay is not bit exact at this shape")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

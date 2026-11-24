"""Paired timing of the chunk kernels against the per-token loops they replaced.

Concept: this laptop part throttles between runs, so two tables measured minutes apart
disagree for reasons that have nothing to do with the code. `time_pair` alternates the
old and the new path inside one process, so both read the same clock state and the ratio
survives. The predecessors are the exact loops the layer used before the chunk kernels:
one row kernel per token, and one MLP per token.

Task: python -m vla_cu pairedtail
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from kernels.triton.rmsnorm import add_into, add_norm_into, rmsnorm_into
from vla_cu.bench_utils import enforce_memory_budget, time_pair, warm_gpu
from vla_cu.blocks import MlpBlock, MlpTokenBlock

TOKENS = (2, 3, 4, 8, 50)
HIDDEN = 720
INTERMEDIATE = 1920
GROUP_FOR_DOWN = 64
WARMUP = 5
ROW_ITERATIONS = 100
MLP_ITERATIONS = 20
REPEATS = 5


def report(
    name: str,
    tokens: int,
    loop_path: Callable[[], None],
    chunk_path: Callable[[], None],
    read: Callable[[], Tensor],
    launches: tuple[int, int],
    iterations: int,
) -> None:
    """Check the two paths agree on one run, then time them alternating and print a row.

    `read` returns the buffer the last call wrote, so both paths are compared on the same
    outputs without the paths themselves allocating anything inside the timed region.
    """
    loop_path()
    expected = read().clone()
    chunk_path()
    actual = read().clone()
    if torch.equal(expected, actual):
        verdict = "exact"
    else:
        verdict = "close" if torch.allclose(expected, actual, atol=1e-3, rtol=1e-3) else "DIFF"
    timing = time_pair(loop_path, chunk_path, iterations, REPEATS, WARMUP)
    print(
        f"{name:>9}  {tokens:>6}  {timing.a * 1e3:>8.3f}  {timing.b * 1e3:>9.3f}  {timing.ratio:>7.2f}x  "
        f"{timing.ratio_low:.2f} to {timing.ratio_high:.2f}  {launches[0]:>4} against {launches[1]:<3} {verdict:>6}"
    )


def main() -> int:
    """Time each chunk kernel against its per-token predecessor, paired in one process."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | hidden {HIDDEN}, intermediate {INTERMEDIATE}")
    print("paired: the loop and the chunk alternate inside one process, so both read one clock state")
    print(f"warmup {WARMUP}, repeats {REPEATS}, {ROW_ITERATIONS} iterations for rows, {MLP_ITERATIONS} for the MLP")
    print()
    print(
        f"{'path':>9}  {'tokens':>6}  {'loop ms':>8}  {'chunk ms':>9}  {'speedup':>8}  {'range':>13}  "
        f"{'launches':>18}  {'same':>6}"
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(HIDDEN, device="cuda", generator=generator) + 1.0
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * (1.0 / HIDDEN**0.5)
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * (1.0 / INTERMEDIATE**0.5)

    for tokens in TOKENS:
        x = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
        other = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
        out = torch.empty_like(x)
        normed = torch.empty_like(x)
        added = torch.empty_like(x)
        one = MlpBlock(up, down, down_bits=4, group_size=GROUP_FOR_DOWN)
        many = MlpTokenBlock(up, down, down_bits=4, group_size=GROUP_FOR_DOWN, tokens=tokens)

        def norm_loop() -> None:
            for token in range(tokens):
                rmsnorm_into(x[token], weight, out[token])

        def norm_chunk() -> None:
            rmsnorm_into(x, weight, out)

        def add_loop() -> None:
            for token in range(tokens):
                add_into(x[token], other[token], out[token])

        def add_chunk() -> None:
            add_into(x, other, out)

        def add_norm_loop() -> None:
            for token in range(tokens):
                add_norm_into(x[token], other[token], weight, added[token], normed[token])

        def add_norm_chunk() -> None:
            add_norm_into(x, other, weight, added, normed)

        def mlp_loop() -> None:
            for token in range(tokens):
                one.into(x[token], out[token])

        def mlp_chunk() -> None:
            many.into(x, out)

        report("norm", tokens, norm_loop, norm_chunk, lambda: out, (tokens, 1), ROW_ITERATIONS)
        report("add", tokens, add_loop, add_chunk, lambda: out, (tokens, 1), ROW_ITERATIONS)
        report(
            "add norm",
            tokens,
            add_norm_loop,
            add_norm_chunk,
            lambda: torch.stack((added[0], normed[0])),
            (tokens, 1),
            ROW_ITERATIONS,
        )
        report("mlp", tokens, mlp_loop, mlp_chunk, lambda: out, (3 * tokens, 3), MLP_ITERATIONS)
        print()
        del x, other, out, normed, added, one, many

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

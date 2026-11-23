"""Randomized correctness sweep across every kernel, seeded and repeatable.

Concept: a test suite checks the cases someone thought of; a fuzz sweep checks the
shapes nobody wrote down. Each case builds random weights, runs the kernel and a float64
reference, and records the worst error. One case outside its tolerance fails the run, so
this is a gate rather than a report.

The seed is fixed and printed. A failure names the shape and the seed, and the same
sequence rebuilds it.

Task: python -m vla_cu fuzz
"""

from __future__ import annotations

import random

import torch

from kernels.triton.int4_gemm import int4_gemm
from kernels.triton.int4_gemv import int4_gemv
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from kernels.triton.int8_gemv import int8_gemv
from kernels.triton.int8_grouped_gemv import int8_grouped_gemv_into
from vla_cu.bench_utils import enforce_memory_budget
from vla_cu.quant import dequantize_int4, pack_int4

SEED = 0
CASES = 32


def relative_error(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Return (max absolute error, relative L2 error) against a float64 reference."""
    diff = (out.double() - ref.double()).abs()
    relative = diff.norm() / ref.double().norm().clamp_min(1e-12)
    return diff.max().item(), relative.item()


def fuzz_int4_gemv(rng: random.Random, generator: torch.Generator) -> tuple[str, float, float]:
    """One random int4 GEMV case against dequantize-then-matmul in float64."""
    group = rng.choice((16, 32, 64))
    n_cols = group * rng.randint(1, 32)
    n_rows = rng.randint(1, 64)
    block_k = rng.choice([value for value in (128, 256, 1024) if value % group == 0])
    w = torch.randn((n_rows, n_cols), device="cuda", generator=generator) * 0.05
    x = torch.randn(n_cols, device="cuda", generator=generator)
    packed, scales = pack_int4(w, group_size=group)
    out = int4_gemv(packed, scales, x, group_size=group, block_k=block_k)
    ref = dequantize_int4(packed, scales, group_size=group).double() @ x.double()
    return f"N={n_rows} K={n_cols} group={group} block_k={block_k}", *relative_error(out, ref)


def fuzz_int8_gemv(rng: random.Random, generator: torch.Generator) -> tuple[str, float, float]:
    """One random int8 GEMV case, split-K included."""
    n_cols = rng.randint(1, 4096)
    n_rows = rng.randint(1, 64)
    split = rng.choice((1, 2, 4, 8))
    block_k = rng.choice((128, 1024, 2048))
    w = torch.randint(-127, 128, (n_rows, n_cols), device="cuda", dtype=torch.int8, generator=generator)
    scales = torch.rand(n_rows, device="cuda", generator=generator) * 0.02 + 1e-3
    x = torch.randn(n_cols, device="cuda", generator=generator)
    out = int8_gemv(w, scales, x, block_k=block_k, split=split)
    ref = (w.double() * scales.double()[:, None]) @ x.double()
    return f"N={n_rows} K={n_cols} split={split} block_k={block_k}", *relative_error(out, ref)


def fuzz_int4_gemm(rng: random.Random, generator: torch.Generator) -> tuple[str, float, float]:
    """One random int4 GEMM case, token count and padding included."""
    group = rng.choice((16, 64))
    n_cols = group * rng.randint(1, 24)
    n_rows = rng.randint(1, 96)
    tokens = rng.randint(1, 64)
    w = torch.randn((n_rows, n_cols), device="cuda", generator=generator) * (1.0 / n_cols**0.5)
    x = torch.randn((tokens, n_cols), device="cuda", generator=generator)
    packed, scales = pack_int4(w, group_size=group)
    out = int4_gemm(packed, scales, x, group_size=group)
    dense = dequantize_int4(packed, scales, group_size=group)
    return f"M={tokens} N={n_rows} K={n_cols} group={group}", *relative_error(out, x.double() @ dense.double().t())


def fuzz_int4_grouped(rng: random.Random, generator: torch.Generator) -> tuple[str, float, float]:
    """One grouped int4 case against one launch per matrix, which must be bit identical."""
    group = rng.choice((16, 32, 64))
    n_cols = group * rng.randint(1, 16)
    first_rows = 4 * rng.randint(1, 8)
    second_rows = 4 * rng.randint(1, 8)
    first = torch.randn((first_rows, n_cols), device="cuda", generator=generator) * 0.05
    second = torch.randn((second_rows, n_cols), device="cuda", generator=generator) * 0.05
    packed, scales = pack_int4(torch.cat((first, second), dim=0), group_size=group)
    x_first = torch.randn(n_cols, device="cuda", generator=generator)
    x_second = torch.randn(n_cols, device="cuda", generator=generator)
    out = torch.empty(first_rows + second_rows, device="cuda")
    segment = torch.cat(
        (
            torch.zeros(first_rows, device="cuda", dtype=torch.int32),
            torch.ones(second_rows, device="cuda", dtype=torch.int32),
        )
    )
    x_base = torch.tensor([0, n_cols], device="cuda", dtype=torch.int32)
    int4_grouped_gemv_into(packed, scales, torch.cat((x_first, x_second)), out, segment, x_base, group_size=group)
    expected = torch.cat(
        (
            int4_gemv(packed[:first_rows], scales[:first_rows], x_first, group_size=group),
            int4_gemv(packed[first_rows:], scales[first_rows:], x_second, group_size=group),
        )
    )
    return f"R={first_rows}+{second_rows} K={n_cols} group={group}", *relative_error(out, expected)


def fuzz_int8_grouped(rng: random.Random, generator: torch.Generator) -> tuple[str, float, float]:
    """One grouped int8 case against one launch per matrix.

    The two mappings reduce the same values in the same order, but the extra table loads
    change register allocation, so the compiler may reassociate the last few adds.
    """
    n_cols = rng.randint(1, 512)
    first_rows = rng.randint(1, 16)
    second_rows = rng.randint(1, 16)
    weights = torch.randint(-127, 128, (first_rows + second_rows, n_cols), device="cuda", dtype=torch.int8, generator=generator)
    scales = torch.rand(first_rows + second_rows, device="cuda", generator=generator) * 0.02 + 1e-3
    x_first = torch.randn(n_cols, device="cuda", generator=generator)
    x_second = torch.randn(n_cols, device="cuda", generator=generator)
    out = torch.empty(first_rows + second_rows, device="cuda")
    segment = torch.cat(
        (
            torch.zeros(first_rows, device="cuda", dtype=torch.int32),
            torch.ones(second_rows, device="cuda", dtype=torch.int32),
        )
    )
    x_base = torch.tensor([0, n_cols], device="cuda", dtype=torch.int32)
    int8_grouped_gemv_into(weights, scales, torch.cat((x_first, x_second)), out, segment, x_base)
    expected = torch.cat(
        (
            int8_gemv(weights[:first_rows], scales[:first_rows], x_first),
            int8_gemv(weights[first_rows:], scales[first_rows:], x_second),
        )
    )
    return f"R={first_rows}+{second_rows} K={n_cols}", *relative_error(out, expected)


SUITES = (
    ("int4_gemv", fuzz_int4_gemv, 1e-3, 1e-4),
    ("int8_gemv", fuzz_int8_gemv, 1e-3, 1e-4),
    ("int4_gemm", fuzz_int4_gemm, 2e-2, 2e-2),
    ("int4_grouped", fuzz_int4_grouped, 1e-6, 1e-6),
    ("int8_grouped", fuzz_int8_grouped, 1e-4, 1e-5),
)


def main() -> int:
    """Run every suite and return 1 if any case breaks its tolerance."""
    enforce_memory_budget()
    rng = random.Random(SEED)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    print(f"GPU {torch.cuda.get_device_name(0)} | seed {SEED} | {CASES} cases per kernel")
    print()
    print(f"{'kernel':>14}  {'cases':>5}  {'worst abs':>10}  {'worst rel':>10}  {'worst shape':>34}  {'verdict':>7}")

    failures = 0
    for name, suite, atol, rtol in SUITES:
        worst_abs = 0.0
        worst_rel = 0.0
        worst_label = ""
        failed = 0
        for _ in range(CASES):
            label, max_abs, relative = suite(rng, generator)
            if max_abs > worst_abs:
                worst_abs = max_abs
                worst_label = label
            worst_rel = max(worst_rel, relative)
            if max_abs > atol or relative > rtol:
                failed += 1
                print(f"FAIL {name} {label}: max abs {max_abs:.3e}, relative {relative:.3e}")
        failures += failed
        print(f"{name:>14}  {CASES:>5}  {worst_abs:>10.3e}  {worst_rel:>10.3e}  {worst_label:>34}  {'ok' if failed == 0 else 'FAIL':>7}")

    print()
    print(f"{len(SUITES) * CASES} cases, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

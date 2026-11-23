"""The whole ladder on identical weights: fp32, bf16, int8, int4.

Every format sees the same weights and activations, so the differences are bytes
streamed and work done per element. Accuracy is printed next to speed because int4
buys bandwidth with precision, and a table without it would be half a result.

Task: python -m vla_cu matrix
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int8_gemv import int8_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    warm_gpu,
)
from vla_cu.quant import GROUP_SIZE, pack_int4

SIZES = ((1024, 1024), (4096, 2048), (8192, 16384))
INT4_ROW_TILES = (1, 2, 4, 8, 16, 32)
INT4_BLOCK_SIZES = (512, 1024)
WARMUP = 5
ITERATIONS = 20


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the L2 norm of the error divided by the L2 norm of the reference.

    Elementwise relative error is useless here: an output that lands near zero makes
    the ratio huge while the vector as a whole is accurate to a few percent.
    """
    return (torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)).item()


def bytes_moved(fmt: str, n_out: int, n_in: int) -> int:
    """Bytes one call of `fmt` moves: weights, activations, scales, output."""
    weights = {"fp32": 4, "bf16": 2, "int8": 1, "int4": 0.5}[fmt] * n_out * n_in
    scales = {"fp32": 0, "bf16": 0, "int8": 4 * n_out, "int4": 4 * n_out * n_in // GROUP_SIZE}[fmt]
    element = 2 if fmt == "bf16" else 4
    return int(weights + scales + element * n_in + element * n_out)


def int4_tuning_sweep(n_out: int, n_in: int) -> None:
    """Print int4 time for each (row tile, block size) pair, the kernel's two knobs.

    The row tile sets how many output rows reuse one activation load, and the block
    size sets how many groups one program reduces per step. Both move register
    pressure and activation traffic, so they are swept together.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((n_out, n_in), device="cuda", generator=generator) * 0.05
    x = torch.randn(n_in, device="cuda", generator=generator)
    out = torch.empty(n_out, device="cuda")
    packed, scales = pack_int4(w)
    n_bytes = bytes_moved("int4", n_out, n_in)

    print(f"int4 tuning at {n_out} x {n_in}, {n_bytes / 1e6:.1f} MB moved")
    print(f"{'rows':>5}  " + "  ".join(f"block {block_k:>4} (ms, GB/s)" for block_k in INT4_BLOCK_SIZES))
    for rows in INT4_ROW_TILES:
        cells = []
        for block_k in INT4_BLOCK_SIZES:
            seconds = time_call(lambda: int4_gemv_into(packed, scales, x, out, block_k=block_k, rows=rows), WARMUP, ITERATIONS)
            cells.append(f"{seconds * 1e3:>10.3f} {bandwidth_gb_s(n_bytes, seconds):>7.1f}")
        print(f"{rows:>5}  " + "  ".join(cells))
    print()


def int4_ablation(n_out: int, n_in: int) -> None:
    """Time int4 at one shape with the two changes this session added, one at a time.

    Hospitality matters here: the same kernel, the same weights, only the eviction hint
    and the scale width differ, so the columns attribute cost to one change each. Both
    are checked against the fp32 matrix-vector product before being timed.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((n_out, n_in), device="cuda", generator=generator) * 0.05
    x = torch.randn(n_in, device="cuda", generator=generator)
    out = torch.empty(n_out, device="cuda")
    exact = torch.mv(w, x)
    packed32, scales32 = pack_int4(w)
    packed16, scales16 = pack_int4(w, scale_dtype=torch.float16)

    cases = (
        ("fp32 scales, no hint", scales32, False, n_out * n_in // 2 + 4 * n_out * n_in // GROUP_SIZE + 4 * n_in + 4 * n_out),
        ("fp32 scales, hints", scales32, True, n_out * n_in // 2 + 4 * n_out * n_in // GROUP_SIZE + 4 * n_in + 4 * n_out),
        ("fp16 scales, hints", scales16, True, n_out * n_in // 2 + 2 * n_out * n_in // GROUP_SIZE + 4 * n_in + 4 * n_out),
    )

    print(f"int4 ablations at {n_out} x {n_in}, one change per row")
    print(f"{'configuration':>20}  {'MB':>7}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}  {'rel L2 err':>11}")
    for label, scales, hints, n_bytes in cases:
        int4_gemv_into(packed32, scales, x, out, hints=hints)
        torch.cuda.synchronize()
        error = relative_error(out.clone(), exact)
        seconds = time_call(lambda: int4_gemv_into(packed32, scales, x, out, hints=hints), WARMUP, ITERATIONS)
        gbs = bandwidth_gb_s(n_bytes, seconds)
        print(
            f"{label:>20}  {n_bytes / 1e6:>7.1f}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  "
            f"{fraction_of_peak(gbs):>6.1%}  {error:>11.2e}"
        )
    print()
    del w, x, out, exact, packed32, scales32, packed16, scales16


def run_shape(n_out: int, n_in: int) -> list[tuple[str, int, float, float]]:
    """Measure every format at one shape and return (format, bytes, seconds, error)."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((n_out, n_in), device="cuda", generator=generator) * 0.05
    x = torch.randn(n_in, device="cuda", generator=generator)
    out = torch.empty(n_out, device="cuda")

    row_amax = w.abs().amax(dim=1)
    int8_scale = torch.where(row_amax > 0, row_amax / 127.0, torch.ones_like(row_amax))
    w_int8 = torch.round(w / int8_scale[:, None]).clamp(-127, 127).to(torch.int8)
    packed, group_scale = pack_int4(w)

    w_bf16 = w.to(torch.bfloat16)
    x_bf16 = x.to(torch.bfloat16)
    out_bf16 = torch.empty(n_out, device="cuda", dtype=torch.bfloat16)

    def fp32_call() -> None:
        torch.mv(w, x, out=out)

    def bf16_call() -> None:
        torch.mv(w_bf16, x_bf16, out=out_bf16)

    def int8_call() -> None:
        int8_gemv_into(w_int8, int8_scale, x, out)

    def int4_call() -> None:
        int4_gemv_into(packed, group_scale, x, out)

    fp32_call()
    torch.cuda.synchronize()
    exact = out.clone()

    errors = {"fp32": 0.0}
    bf16_call()
    torch.cuda.synchronize()
    errors["bf16"] = relative_error(out_bf16.float(), exact)
    int8_call()
    torch.cuda.synchronize()
    errors["int8"] = relative_error(out.clone(), exact)
    int4_call()
    torch.cuda.synchronize()
    errors["int4"] = relative_error(out.clone(), exact)

    rows = []
    for fmt, call in (("fp32", fp32_call), ("bf16", bf16_call), ("int8", int8_call), ("int4", int4_call)):
        seconds = time_call(call, WARMUP, ITERATIONS)
        rows.append((fmt, bytes_moved(fmt, n_out, n_in), seconds, errors[fmt]))

    del w, w_int8, packed, group_scale, w_bf16, x_bf16, out_bf16, exact
    return rows


def main() -> int:
    """Print one row per format per shape, with GB/s and accuracy side by side."""
    enforce_memory_budget()
    warm_gpu()
    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | group {GROUP_SIZE} | batch 1")
    print()
    print(
        f"{'N x K':>14}  {'fmt':>5}  {'MB':>7}  {'ms':>8}  {'GB/s':>8}  "
        f"{'% DRAM':>7}  {'x fp32':>7}  {'rel L2 err':>11}"
    )

    int4_tuning_sweep(*SIZES[-1])
    int4_ablation(*SIZES[-1])

    for n_out, n_in in SIZES:
        rows = run_shape(n_out, n_in)
        fp32_s = rows[0][2]
        for fmt, n_bytes, seconds, error in rows:
            gbs = bandwidth_gb_s(n_bytes, seconds)
            label = f"{n_out} x {n_in}"
            print(
                f"{label:>14}  {fmt:>5}  {n_bytes / 1e6:>7.1f}  {seconds * 1e3:>8.3f}  "
                f"{gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  {fp32_s / seconds:>6.2f}x  {error:>11.2e}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

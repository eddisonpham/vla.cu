"""A real attention block at batch 1, in int4 and int8, separate against grouped.

Concept: at one token, attention is a matrix multiply, a softmax over a single row, and
a second matrix multiply, and the softmax is the identity because the only visible key
is itself. The three projection matrices Wq, Wk and Wv all read the same x, so a grouped
launch concatenates their rows and reads x once. Both precisions are measured because
int4 is the byte floor while int8 is the accuracy floor, and this block is the first
place the repo can price one against the other.

Task: python -m vla_cu attn
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from kernels.triton.int8_gemv import int8_gemv_into
from kernels.triton.int8_grouped_gemv import int8_grouped_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    time_pair,
    warm_gpu,
)
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8

HIDDEN = 960
MATRICES = 3
WARMUP = 10
ITERATIONS = 50


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the cosine similarity of two vectors."""
    return torch.nn.functional.cosine_similarity(approx, exact, dim=0).item()


def low_bit_bytes() -> tuple[int, int]:
    """Return the bytes the int4 and int8 versions of this block move."""
    int4_qkv = MATRICES * (HIDDEN * HIDDEN // 2 + 4 * HIDDEN * HIDDEN // GROUP_SIZE)
    int4_o = HIDDEN * HIDDEN // 2 + 4 * HIDDEN * HIDDEN // GROUP_SIZE
    int8_qkv = MATRICES * (HIDDEN * HIDDEN + 4 * HIDDEN)
    int8_o = HIDDEN * HIDDEN + 4 * HIDDEN
    return int4_qkv + int4_o + 8 * HIDDEN, int8_qkv + int8_o + 8 * HIDDEN


def main() -> int:
    """Time separate, grouped and graphed qkv plus o at both precisions."""
    enforce_memory_budget()
    warm_gpu()
    generator = torch.Generator(device="cuda").manual_seed(0)
    scale = 1.0 / HIDDEN**0.5
    wq = torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale
    wk = torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale
    wv = torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale
    wo = torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale
    x = torch.randn(HIDDEN, device="cuda", generator=generator)

    qkv = torch.cat((wq, wk, wv), dim=0)
    packed_qkv, scales_qkv = pack_int4(qkv)
    packed_o, scales_o = pack_int4(wo)
    packed_q, scales_q = pack_int4(wq)
    packed_k, scales_k = pack_int4(wk)
    packed_v, scales_v = pack_int4(wv)

    qkv8, scales_qkv8 = quantize_int8(qkv)
    o8, scales_o8 = quantize_int8(wo)
    q8, scales_q8 = quantize_int8(wq)
    k8, scales_k8 = quantize_int8(wk)
    v8, scales_v8 = quantize_int8(wv)

    total_rows = MATRICES * HIDDEN
    segment = torch.arange(MATRICES, device="cuda", dtype=torch.int32).repeat_interleave(HIDDEN)
    x_base = torch.zeros(MATRICES, device="cuda", dtype=torch.int32)

    fp32_qkv = torch.empty(total_rows, device="cuda")
    fp32_out = torch.empty(HIDDEN, device="cuda")
    separate_qkv = torch.empty(total_rows, device="cuda")
    separate_out = torch.empty(HIDDEN, device="cuda")
    grouped_qkv = torch.empty(total_rows, device="cuda")
    grouped_out = torch.empty(HIDDEN, device="cuda")
    separate8_qkv = torch.empty(total_rows, device="cuda")
    separate8_out = torch.empty(HIDDEN, device="cuda")
    grouped8_qkv = torch.empty(total_rows, device="cuda")
    grouped8_out = torch.empty(HIDDEN, device="cuda")

    def fp32_block() -> None:
        torch.mv(wq, x, out=fp32_qkv[:HIDDEN])
        torch.mv(wk, x, out=fp32_qkv[HIDDEN : 2 * HIDDEN])
        torch.mv(wv, x, out=fp32_qkv[2 * HIDDEN :])
        torch.mv(wo, fp32_qkv[2 * HIDDEN :], out=fp32_out)

    def separate_block() -> None:
        int4_gemv_into(packed_q, scales_q, x, separate_qkv[:HIDDEN])
        int4_gemv_into(packed_k, scales_k, x, separate_qkv[HIDDEN : 2 * HIDDEN])
        int4_gemv_into(packed_v, scales_v, x, separate_qkv[2 * HIDDEN :])
        int4_gemv_into(packed_o, scales_o, separate_qkv[2 * HIDDEN :], separate_out)

    def grouped_block() -> None:
        int4_grouped_gemv_into(packed_qkv, scales_qkv, x, grouped_qkv, segment, x_base)
        int4_gemv_into(packed_o, scales_o, grouped_qkv[2 * HIDDEN :], grouped_out)

    def separate8_block() -> None:
        int8_gemv_into(q8, scales_q8, x, separate8_qkv[:HIDDEN])
        int8_gemv_into(k8, scales_k8, x, separate8_qkv[HIDDEN : 2 * HIDDEN])
        int8_gemv_into(v8, scales_v8, x, separate8_qkv[2 * HIDDEN :])
        int8_gemv_into(o8, scales_o8, separate8_qkv[2 * HIDDEN :], separate8_out)

    def grouped8_block() -> None:
        int8_grouped_gemv_into(qkv8, scales_qkv8, x, grouped8_qkv, segment, x_base)
        int8_gemv_into(o8, scales_o8, grouped8_qkv[2 * HIDDEN :], grouped8_out)

    for path in (fp32_block, separate_block, grouped_block, separate8_block, grouped8_block):
        path()
    torch.cuda.synchronize()

    exact = fp32_out.clone()
    separate_snapshot = separate_out.clone()
    grouped_snapshot = grouped_out.clone()
    separate8_snapshot = separate8_out.clone()
    grouped8_snapshot = grouped8_out.clone()

    int4_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(int4_graph):
        grouped_block()
    int4_graph.replay()
    int8_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(int8_graph):
        grouped8_block()
    int8_graph.replay()
    torch.cuda.synchronize()

    int4_graph_out = grouped_out.clone()
    int8_graph_out = grouped8_out.clone()
    int4_bytes, int8_bytes = low_bit_bytes()
    fp32_bytes = 4 * 4 * HIDDEN * HIDDEN + 8 * HIDDEN

    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"attention block at batch 1: q, k, v of {HIDDEN} x {HIDDEN}, then o of {HIDDEN} x {HIDDEN}")
    print(f"softmax over one visible key is the identity, so the block is v = Wv x, out = Wo v")
    print(f"bytes: fp32 {fp32_bytes / 1e6:.2f} MB, int4 {int4_bytes / 1e6:.2f} MB, int8 {int8_bytes / 1e6:.2f} MB")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | int4 group {GROUP_SIZE} | row tile 4")
    print()
    print(f"{'path':>14}  {'launches':>8}  {'ms':>8}  {'GB/s':>8}  {'% PEAK':>7}  {'cos':>8}")

    paths = (
        ("fp32", 4, fp32_block, fp32_bytes, fp32_out),
        ("int4 separate", 4, separate_block, int4_bytes, separate_out),
        ("int4 grouped", 2, grouped_block, int4_bytes, grouped_out),
        ("int4 graph", 2, int4_graph.replay, int4_bytes, int4_graph_out),
        ("int8 separate", 4, separate8_block, int8_bytes, separate8_out),
        ("int8 grouped", 2, grouped8_block, int8_bytes, grouped8_out),
        ("int8 graph", 2, int8_graph.replay, int8_bytes, int8_graph_out),
    )
    timings = {}
    for name, launches, call, n_bytes, result in paths:
        seconds = time_call(call, WARMUP, ITERATIONS)
        timings[name] = seconds
        gbs = bandwidth_gb_s(n_bytes, seconds)
        print(
            f"{name:>14}  {launches:>8}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  "
            f"{cosine(result, exact):>8.5f}"
        )

    int4_pairs = time_pair(fp32_block, int4_graph.replay, ITERATIONS, 5)
    int8_pairs = time_pair(fp32_block, int8_graph.replay, ITERATIONS, 5)

    print()
    print(f"int4 grouped / separate {timings['int4 separate'] / timings['int4 grouped']:.2f}x | graph / separate {timings['int4 separate'] / timings['int4 graph']:.2f}x")
    print(f"int8 grouped / separate {timings['int8 separate'] / timings['int8 grouped']:.2f}x | graph / separate {timings['int8 separate'] / timings['int8 graph']:.2f}x")
    print(f"int4 grouped agrees with separate: {torch.allclose(separate_snapshot, grouped_snapshot, rtol=1e-4, atol=1e-3)}")
    print(f"int8 grouped agrees with separate: {torch.allclose(separate8_snapshot, grouped8_snapshot, rtol=1e-4, atol=1e-3)}")
    print(f"int4 graph reproduces the eager chain bit for bit: {torch.equal(int4_graph_out, grouped_snapshot)}")
    print(f"int8 graph reproduces the eager chain bit for bit: {torch.equal(int8_graph_out, grouped8_snapshot)}")
    print(f"int8 costs {timings['int8 graph'] / timings['int4 graph']:.2f}x the int4 time for {(int8_bytes / int4_bytes):.2f}x the bytes")
    print(f"paired ratio fp32 / int4 graph {int4_pairs.ratio:.2f}x ({int4_pairs.ratio_low:.2f} to {int4_pairs.ratio_high:.2f})")
    print(f"paired ratio fp32 / int8 graph {int8_pairs.ratio:.2f}x ({int8_pairs.ratio_low:.2f} to {int8_pairs.ratio_high:.2f})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A real attention block at batch 1, where q, k and v share one activation vector.

Concept: at one token, attention is a matrix multiply, a softmax over a single row, and
a second matrix multiply, and the softmax is the identity because the only visible key
is itself. What is left is v = Wv x, and out = Wo v. The three projection matrices Wq,
Wk and Wv all read the same x, so three launches carry one activation vector between
them; a grouped launch concatenates their packed rows and reads x once.

The grouped kernel needs every matrix's row count to be a multiple of the row tile, so
960 rows of 960 weights fits without padding. The grouped path is int4 only: there is no
int8 grouped kernel in the repo, which is why this block is measured in int4.

Task: python -m vla_cu attn
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    time_pair,
    warm_gpu,
)
from vla_cu.quant import GROUP_SIZE, pack_int4

HIDDEN = 960
MATRICES = 3
WARMUP = 10
ITERATIONS = 50


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the cosine similarity of two vectors."""
    return torch.nn.functional.cosine_similarity(approx, exact, dim=0).item()


def main() -> int:
    """Time four separate projections against a grouped qkv launch plus one for o."""
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

    total_rows = MATRICES * HIDDEN
    segment = torch.arange(MATRICES, device="cuda", dtype=torch.int32).repeat_interleave(HIDDEN)
    x_base = torch.zeros(MATRICES, device="cuda", dtype=torch.int32)

    fp32_qkv = torch.empty(total_rows, device="cuda")
    fp32_out = torch.empty(HIDDEN, device="cuda")
    separate_qkv = torch.empty(total_rows, device="cuda")
    separate_out = torch.empty(HIDDEN, device="cuda")
    grouped_qkv = torch.empty(total_rows, device="cuda")
    grouped_out = torch.empty(HIDDEN, device="cuda")

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

    fp32_block()
    separate_block()
    grouped_block()
    torch.cuda.synchronize()
    separate_snapshot = separate_out.clone()
    grouped_snapshot = grouped_out.clone()

    grouped_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(grouped_graph):
        grouped_block()
    grouped_graph.replay()
    torch.cuda.synchronize()
    grouped_graph_out = grouped_out.clone()

    low_bit_bytes = MATRICES * (HIDDEN * HIDDEN // 2 + 4 * HIDDEN * HIDDEN // GROUP_SIZE) + HIDDEN * HIDDEN // 2 + 4 * HIDDEN * HIDDEN // GROUP_SIZE + 8 * HIDDEN
    fp32_bytes = 4 * 4 * HIDDEN * HIDDEN + 8 * HIDDEN

    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"attention block at batch 1: q, k, v of {HIDDEN} x {HIDDEN}, then o of {HIDDEN} x {HIDDEN}")
    print(f"softmax over one visible key is the identity, so the block is v = Wv x, out = Wo v")
    print(f"low-bit bytes {low_bit_bytes / 1e6:.2f} MB per block, fp32 {fp32_bytes / 1e6:.2f} MB")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | group {GROUP_SIZE} | row tile 4")
    print()
    print(f"{'path':>12}  {'launches':>8}  {'ms':>8}  {'GB/s':>8}  {'% PEAK':>7}  {'cos':>8}")

    exact = fp32_out.clone()
    paths = (
        ("fp32", 4, fp32_block, fp32_bytes, fp32_out),
        ("separate", 4, separate_block, low_bit_bytes, separate_out),
        ("grouped", 2, grouped_block, low_bit_bytes, grouped_out),
        ("grouped graph", 2, grouped_graph.replay, low_bit_bytes, grouped_graph_out),
    )
    timings = {}
    for name, launches, call, n_bytes, result in paths:
        seconds = time_call(call, WARMUP, ITERATIONS)
        timings[name] = seconds
        gbs = bandwidth_gb_s(n_bytes, seconds)
        print(
            f"{name:>12}  {launches:>8}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  "
            f"{cosine(result, exact):>8.5f}"
        )

    pairs = time_pair(fp32_block, grouped_graph.replay, ITERATIONS, 5)
    print()
    print(f"grouped / separate {timings['separate'] / timings['grouped']:.2f}x | graph / separate {timings['separate'] / timings['grouped graph']:.2f}x")
    print(f"grouped agrees with separate: {torch.allclose(separate_snapshot, grouped_snapshot, rtol=1e-4, atol=1e-3)}")
    print(f"graph reproduces the eager grouped chain bit for bit: {torch.equal(grouped_graph_out, grouped_snapshot)}")
    print(f"paired ratio fp32 / graph {pairs.ratio:.2f}x ({pairs.ratio_low:.2f} to {pairs.ratio_high:.2f}), fp32 {pairs.a * 1e3:.4f} ms")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

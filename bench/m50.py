"""Is the action expert a batch 1 problem? Measure it at its real M.

Concept: SmolVLA predicts a chunk of 50 actions, and the action expert is a transformer
over those 50 tokens, so its linear layers see M = 50, not 1. That changes the kernel
question. A GEMV reads every weight once per call, so 50 calls read the weights 50
times, while one GEMM reads them once and reuses each byte 50 times. The two paths move
different numbers of bytes, so both a time column and a derived byte budget are shown.

The expert width is 0.75 times the VLM width (verified in the LeRobot config) applied
to the published 960 and 2560 of the SmolVLM2-500M backbone, so 720 and 1920. Note that 720 is not a multiple of
the 64-weight group: 720 = 11 * 64 + 16, so this benchmark has to fall back to group 16
for that shape, which puts 4 bytes of scale on every 16 weights instead of every 64.
That is a real constraint on the format, not a benchmark detail (DEC-020).

Task: python -m vla_cu m50
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemm import int4_gemm_into
from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    warm_gpu,
)
from vla_cu.quant import dequantize_int4, pack_int4

TOKENS = 50
EXPERT_HIDDEN = 720
EXPERT_INTERMEDIATE = 1920
GROUP_FOR_HIDDEN = 16
GROUP_FOR_INTERMEDIATE = 64
SHAPES = ((EXPERT_INTERMEDIATE, EXPERT_HIDDEN, GROUP_FOR_HIDDEN), (EXPERT_HIDDEN, EXPERT_INTERMEDIATE, GROUP_FOR_INTERMEDIATE))
WARMUP = 5
ITERATIONS = 20


def main() -> int:
    """Time the per-row GEMV loop against one int4 GEMM and one bf16 GEMM."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s")
    print(f"action expert: M = {TOKENS} tokens, widths {EXPERT_HIDDEN} and {EXPERT_INTERMEDIATE}")
    print(f"warmup {WARMUP} | iterations {ITERATIONS}")
    print()
    print(f"{'N x K':>12}  {'group':>5}  {'path':>16}  {'bytes':>9}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}  {'floor ms':>8}")

    for n_out, n_in, group in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn((n_out, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
        activations = torch.randn((TOKENS, n_in), device="cuda", generator=generator)
        packed, scales = pack_int4(w, group_size=group)
        out = torch.empty((TOKENS, n_out), device="cuda")

        half = dequantize_int4(packed, scales, group_size=group).to(torch.bfloat16)
        half_activations = activations.to(torch.bfloat16)
        half_out = torch.empty((TOKENS, n_out), device="cuda", dtype=torch.bfloat16)

        def gemv_loop() -> None:
            for token in range(TOKENS):
                int4_gemv_into(packed, scales, activations[token], out[token], group_size=group)

        def half_gemm() -> None:
            torch.matmul(half_activations, half.transpose(0, 1), out=half_out)

        def int4_gemm() -> None:
            int4_gemm_into(packed, scales, activations, out, group_size=group)

        gemv_loop()
        half_gemm()
        int4_gemm()
        torch.cuda.synchronize()

        gemv_bytes = TOKENS * (n_out * n_in // 2 + 4 * n_out * n_in // group + 4 * n_in) + 4 * TOKENS * n_out
        gemm_bytes = n_out * n_in * 2 + 4 * n_out * n_in // group + 2 * TOKENS * n_in + 2 * TOKENS * n_out
        int4_bytes = n_out * n_in // 2 + 4 * n_out * n_in // group + 4 * TOKENS * n_in + 4 * TOKENS * n_out
        gemv_seconds = time_call(gemv_loop, WARMUP, ITERATIONS)
        gemm_seconds = time_call(half_gemm, WARMUP, ITERATIONS)
        int4_seconds = time_call(int4_gemm, WARMUP, ITERATIONS)

        for name, n_bytes, seconds in (
            ("gemv loop x50", gemv_bytes, gemv_seconds),
            ("int4 gemm x1", int4_bytes, int4_seconds),
            ("bf16 gemm x1", gemm_bytes, gemm_seconds),
        ):
            gbs = bandwidth_gb_s(n_bytes, seconds)
            print(
                f"{f'{n_out} x {n_in}':>12}  {group:>5}  {name:>16}  {n_bytes / 1e6:>7.1f}MB  {seconds * 1e3:>8.3f}  "
                f"{gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  {n_bytes / PEAK_BANDWIDTH_GB_S / 1e6:>8.3f}"
            )
        print(
            f"{'':>12}  {'':>5}  int4 gemm / gemv {gemv_seconds / int4_seconds:.1f}x on time, "
            f"bytes {gemv_bytes / int4_bytes:.1f}x; bf16 gemm / gemv {gemv_seconds / gemm_seconds:.1f}x, "
            f"bytes {gemv_bytes / gemm_bytes:.1f}x"
        )
        print()
        del w, packed, scales, activations, half, half_activations, out, half_out

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Where the GEMV stops winning: sweep M and time a GEMV loop against one GEMM.

Concept: the two paths move different bytes. A GEMV reads every weight once per call,
so M calls read the packed weights M times. A GEMM reads them once and reuses each byte
across all M tokens. At M = 1 the GEMV wins because it reads the same bytes with no
tensor-core setup; at M = 50 the GEMM wins because it read them one fiftieth as often.
This sweep finds the crossing on the action expert's real shapes.

Both paths are checked against a fp32 reference in the same run, because a crossover
between two wrong kernels is not a result.

Task: python -m vla_cu crossover
"""

from __future__ import annotations

import torch

from kernels.triton.int4_gemm import int4_gemm_into
from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    enforce_memory_budget,
    time_call,
    warm_gpu,
)
from vla_cu.quant import GROUP_SIZE, dequantize_int4, pack_int4

TOKENS = (1, 2, 4, 8, 16, 32, 50, 64)
SHAPES = ((1920, 720, 16), (720, 1920, GROUP_SIZE), (2560, 960, GROUP_SIZE))
WARMUP = 5
ITERATIONS = 20


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the cosine similarity of two tensors flattened to vectors."""
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def gemv_bytes(tokens: int, n_out: int, n_in: int, group: int) -> int:
    """Return the bytes a GEMV loop over `tokens` rows moves, weights read per call."""
    return tokens * (n_out * n_in // 2 + 4 * n_out * n_in // group + 4 * n_in) + 4 * tokens * n_out


def gemm_bytes(tokens: int, n_out: int, n_in: int, group: int) -> int:
    """Return the bytes one GEMM moves: the weights once, plus every activation row."""
    return n_out * n_in // 2 + 4 * n_out * n_in // group + 4 * tokens * n_in + 4 * tokens * n_out


def main() -> int:
    """Time M GEMV calls against one GEMM for M from 1 to 64, and report the crossing."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | tokens {', '.join(str(t) for t in TOKENS)}")
    print()

    for n_out, n_in, group in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn((n_out, n_in), device="cuda", generator=generator) * (1.0 / n_in**0.5)
        packed, scales = pack_int4(w, group_size=group)
        dense = dequantize_int4(packed, scales, group_size=group)
        print(f"{n_out} x {n_in}, group {group}, packed {n_out * n_in / 2 / 1e6:.2f} MB, dense fp32 {n_out * n_in * 4 / 1e6:.2f} MB")
        print(f"{'M':>3}  {'gemv loop':>10}  {'gemm':>8}  {'winner':>7}  {'ratio':>6}  {'bytes':>7}  {'gemv cos':>8}  {'gemm cos':>8}")

        crossing = None
        for tokens in TOKENS:
            activations = torch.randn((tokens, n_in), device="cuda", generator=generator)
            out_gemv = torch.empty((tokens, n_out), device="cuda")
            out_gemm = torch.empty((tokens, n_out), device="cuda")
            reference = activations @ dense.transpose(0, 1)

            def gemv_loop() -> None:
                for token in range(tokens):
                    int4_gemv_into(packed, scales, activations[token], out_gemv[token], group_size=group)

            def gemm() -> None:
                int4_gemm_into(packed, scales, activations, out_gemm, group_size=group)

            gemv_loop()
            gemm()
            torch.cuda.synchronize()
            gemv_seconds = time_call(gemv_loop, WARMUP, ITERATIONS)
            gemm_seconds = time_call(gemm, WARMUP, ITERATIONS)
            if crossing is None and gemm_seconds < gemv_seconds:
                crossing = tokens
            winner = "gemm" if gemm_seconds < gemv_seconds else "gemv"
            ratio = gemv_seconds / gemm_seconds
            print(
                f"{tokens:>3}  {gemv_seconds * 1e3:>10.3f}  {gemm_seconds * 1e3:>8.3f}  {winner:>7}  {ratio:>5.2f}x  "
                f"{gemv_bytes(tokens, n_out, n_in, group) / gemm_bytes(tokens, n_out, n_in, group):>6.1f}x  "
                f"{cosine(out_gemv, reference):>8.5f}  {cosine(out_gemm, reference):>8.5f}"
            )
            del activations, out_gemv, out_gemm, reference

        answer = f"M = {crossing}" if crossing is not None else "not in this range"
        print(f"smallest M where one GEMM beats M GEMV calls: {answer}")
        print()
        del w, packed, scales, dense

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

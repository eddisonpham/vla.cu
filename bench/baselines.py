"""One matrix, one vector, five ways to read it: this repo against torch and bitsandbytes.

Concept: at batch 1 every path here sees the same seeded fp32 matrix and the same fp32
vector, so the differences are stored bytes and the arithmetic each library does inside the
kernel. bitsandbytes nf4 is the nearest installed alternative to this repo's int4. torchao's
weight-only int4 needs fbgemm-gpu-genai, which has no Windows wheel, so it reports the
ImportError instead of a time.

Task: python -m vla_cu baselines
"""

from __future__ import annotations

from typing import Callable, NamedTuple

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
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8

SHAPES = ((2560, 960), (1920, 768), (8192, 16384))
WARMUP = 10
ITERATIONS = 40
PASSES = 2
BNB_BLOCKSIZE = 64


class Result(NamedTuple):
    """One measured path at one shape: what it stores, what it costs, what it loses."""

    label: str
    n_bytes: int
    seconds: float
    cosine: float
    relative: float


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the L2 norm of the error over the L2 norm of the reference."""
    return (torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)).item()


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the cosine similarity of two flattened vectors."""
    return torch.nn.functional.cosine_similarity(approx.reshape(-1), exact.reshape(-1), dim=0).item()


def stored_bytes(label: str, n_out: int, n_in: int) -> int:
    """Return the bytes a path streams: weights, scales, the vector and the output.

    The bitsandbytes row is nominal at 4 bit with one fp32 absmax per 64 weights, its
    default blocksize with double quantizing left on, which stores a few hundred extra
    bytes per matrix rather than per row.
    """
    widths = {"fp32": 4.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5, "bnb nf4": 0.5}
    scales = 4.0 * n_out * n_in / GROUP_SIZE if label in {"int4", "bnb nf4"} else 4.0 * n_out if label == "int8" else 0.0
    element = 2 if label == "bf16" else 4
    return int(widths[label] * n_out * n_in + scales + element * n_in + element * n_out)


def best_of(call: Callable[[], None], iterations: int) -> float:
    """Return the fastest mean of `PASSES` timed runs, which trims a drifting clock."""
    return min(time_call(call, WARMUP, iterations) for _ in range(PASSES))


def bnb_layer(weight: torch.Tensor, compute_dtype: torch.dtype):
    """Return a bitsandbytes nf4 Linear4bit holding `weight` [N, K] with no bias."""
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(
        weight.shape[1], weight.shape[0], bias=False, quant_type="nf4", compute_dtype=compute_dtype
    )
    layer.weight = bnb.nn.Params4bit(weight, requires_grad=False, quant_type="nf4")
    return layer.to("cuda")


def torchao_probe(weight: torch.Tensor, x: torch.Tensor) -> str:
    """Quantize one small linear with torchao's int4 and return a status line, never raising.

    A weight-only int4 path on this platform depends on fbgemm-gpu-genai, so the probe is
    the honest way to report whether the comparison exists at all.
    """
    try:
        from torchao.quantization import Int4WeightOnlyConfig, quantize_

        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False).cuda()
        with torch.no_grad():
            linear.weight.copy_(weight)
        quantize_(linear, Int4WeightOnlyConfig(group_size=GROUP_SIZE))
        out = linear(x[None].to(torch.bfloat16))
        return f"available, one {tuple(out.shape)} call ran"
    except Exception as error:
        return f"unavailable, {type(error).__name__}: {error}"


def measure(n_out: int, n_in: int, iterations: int) -> tuple[list[Result], float]:
    """Build every path on one matrix and return its results plus the fp32 time."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn((n_out, n_in), device="cuda", generator=generator) * 0.05
    vector = torch.randn(n_in, device="cuda", generator=generator)
    out = torch.empty(n_out, device="cuda")
    weight_bf16 = weight.to(torch.bfloat16)
    vector_bf16 = vector.to(torch.bfloat16)
    out_bf16 = torch.empty(n_out, device="cuda", dtype=torch.bfloat16)

    codes8, scales8 = quantize_int8(weight)
    codes4, scales4 = pack_int4(weight)
    layer = bnb_layer(weight, torch.bfloat16)
    bnb_input = vector.to(torch.bfloat16)[None]
    bnb_out = layer(bnb_input)[0]

    def fp32_call() -> None:
        torch.mv(weight, vector, out=out)

    def bf16_call() -> None:
        torch.mv(weight_bf16, vector_bf16, out=out_bf16)

    def int8_call() -> None:
        int8_gemv_into(codes8, scales8, vector, out)

    def int4_call() -> None:
        int4_gemv_into(codes4, scales4, vector, out)

    def bnb_call() -> None:
        layer(bnb_input)

    fp32_call()
    torch.cuda.synchronize()
    exact = out.clone()
    outputs = {}
    for label, call in (("bf16", bf16_call), ("int8", int8_call), ("int4", int4_call), ("bnb nf4", bnb_call)):
        call()
        torch.cuda.synchronize()
        outputs[label] = (out_bf16.float() if label == "bf16" else bnb_out.float() if label == "bnb nf4" else out.clone())

    results = []
    for label, call in (
        ("fp32", fp32_call),
        ("bf16", bf16_call),
        ("int8", int8_call),
        ("int4", int4_call),
        ("bnb nf4", bnb_call),
    ):
        seconds = best_of(call, iterations)
        held = exact if label == "fp32" else outputs[label]
        results.append(
            Result(
                label,
                stored_bytes(label, n_out, n_in),
                seconds,
                1.0 if label == "fp32" else cosine(held, exact),
                0.0 if label == "fp32" else relative_error(held, exact),
            )
        )

    del weight, weight_bf16, codes8, codes4, layer, outputs, exact, out, out_bf16, bnb_out
    return results, results[0].seconds


def main() -> int:
    """Print one table per shape: stored bytes, time, bandwidth, and error against fp32."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | batch 1 | group {GROUP_SIZE}")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | best of {PASSES} passes | one seeded matrix per shape")
    print("the 1920 x 768 row is the expert's 720 wide mlp padded as DEC-044 pads it; smaller matrices are L2 resident")
    print("so their GB/s is cache bandwidth, not DRAM, and the percent column reads low on purpose")
    probe_weight = torch.randn((256, 512), device="cuda")
    print(f"torchao int4 weight-only: {torchao_probe(probe_weight, torch.randn(512, device='cuda'))}")
    del probe_weight
    print()

    for n_out, n_in in SHAPES:
        iterations = ITERATIONS if n_out * n_in <= 4096 * 2048 else ITERATIONS // 2
        results, fp32_seconds = measure(n_out, n_in, iterations)
        print(f"{n_out} x {n_in} weights, one vector")
        print(f"{'path':>9}  {'MB':>7}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}  {'x fp32':>7}  {'cos':>9}  {'rel L2':>9}")
        for result in results:
            gbs = bandwidth_gb_s(result.n_bytes, result.seconds)
            print(
                f"{result.label:>9}  {result.n_bytes / 1e6:>7.1f}  {result.seconds * 1e3:>8.3f}  {gbs:>8.1f}  "
                f"{fraction_of_peak(gbs):>6.1%}  {fp32_seconds / result.seconds:>6.2f}x  "
                f"{result.cosine:>9.5f}  {result.relative:>9.2e}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

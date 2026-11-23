"""One SmolVLA-shaped MLP block end to end, fp32 against fused low-bit.

The block is the transformer MLP at the sizes SmolVLA's 500M backbone uses: hidden
960 into intermediate 2560 and back, batch 1. Those are the repo's assumed layer
shapes (see agents/RESEARCH.md); the config has not been verified against a
checkpoint here, so treat the numbers as that layer, not as a whole model.

Three paths, same weights:
  fp32        torch.mv on the original weights
  dequantize  weights to fp32, then torch.mv, the obvious baseline
  fused       the int8 and int4 kernels through torch custom ops

The dequantize path exists to show what fusion buys: it moves the same low-bit
weights but then writes a large fp32 copy and reads it back.

Task: python -m vla_cu layer
"""

from __future__ import annotations

import torch

from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
)
from vla_cu.ops import int4_gemv_op, int8_gemv_op
from vla_cu.quant import GROUP_SIZE, dequantize_int4, pack_int4

HIDDEN = 960
INTERMEDIATE = 2560
WARMUP = 10
ITERATIONS = 50


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [N, K] fp32 weights to int8 codes with one scale per output row."""
    row_amax = w.abs().amax(dim=1)
    scales = torch.where(row_amax > 0, row_amax / 127.0, torch.ones_like(row_amax))
    return torch.round(w / scales[:, None]).clamp(-127, 127).to(torch.int8), scales


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the cosine similarity of two vectors, the quality gate this repo uses."""
    return torch.nn.functional.cosine_similarity(approx, exact, dim=0).item()


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the L2 norm of the error over the L2 norm of the reference."""
    return (torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)).item()


def main() -> int:
    """Time the three paths for one MLP block and report accuracy beside speed."""
    enforce_memory_budget()
    generator = torch.Generator(device="cuda").manual_seed(0)
    scale = 1.0 / HIDDEN**0.5
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * scale
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * scale
    x = torch.randn(HIDDEN, device="cuda", generator=generator)

    up_int8, up_scale = quantize_int8(up)
    down_packed, down_scale = pack_int4(down)
    activation = torch.nn.functional.silu

    hidden_fp32 = torch.empty(INTERMEDIATE, device="cuda")
    out_fp32 = torch.empty(HIDDEN, device="cuda")

    up_values = up_int8.float() * up_scale[:, None]
    down_values = dequantize_int4(down_packed, down_scale)
    hidden_dequant = torch.empty(INTERMEDIATE, device="cuda")
    out_dequant = torch.empty(HIDDEN, device="cuda")

    def fp32_call() -> None:
        torch.mv(up, x, out=hidden_fp32)
        activation(hidden_fp32, inplace=True)
        torch.mv(down, hidden_fp32, out=out_fp32)

    def dequant_call() -> None:
        torch.mv(up_values, x, out=hidden_dequant)
        activation(hidden_dequant, inplace=True)
        torch.mv(down_values, hidden_dequant, out=out_dequant)

    def fused_call() -> None:
        hidden = activation(int8_gemv_op(up_int8, up_scale, x), inplace=True)
        int4_gemv_op(down_packed, down_scale, hidden)

    fp32_call()
    torch.cuda.synchronize()
    exact = out_fp32.clone()

    dequant_call()
    torch.cuda.synchronize()
    dequant_out = out_dequant.clone()

    fused_hidden = activation(int8_gemv_op(up_int8, up_scale, x), inplace=True)
    fused_out = int4_gemv_op(down_packed, down_scale, fused_hidden).clone()
    torch.cuda.synchronize()

    weight_bytes = {
        "fp32": up.numel() * 4 + down.numel() * 4,
        "dequantize": up.numel() * 4 + down.numel() * 4,
        "fused": up.numel() + down.numel() // 2 + 4 * (down.numel() // GROUP_SIZE) + 4 * INTERMEDIATE,
    }
    paths = (("fp32", fp32_call), ("dequantize", dequant_call), ("fused", fused_call))
    timings = {name: time_call(call, WARMUP, ITERATIONS) for name, call in paths}

    l2_bytes = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | L2 {l2_bytes / 1e6:.1f} MB")
    print(f"MLP block: {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}, batch 1, group {GROUP_SIZE}")
    print(f"fp32 weights {weight_bytes['fp32'] / 1e6:.1f} MB, low-bit weights {weight_bytes['fused'] / 1e6:.1f} MB")
    print(f"warmup {WARMUP} | iterations {ITERATIONS}")
    print()
    print(f"{'path':>11}  {'ms':>8}  {'GB/s':>8}  {'% PEAK':>7}  {'x fp32':>7}  {'cos':>8}  {'rel L2':>9}")

    fp32_seconds = timings["fp32"]
    for name, _ in paths:
        seconds = timings[name]
        n_bytes = weight_bytes[name]
        gbs = bandwidth_gb_s(n_bytes, seconds)
        result = {"fp32": exact, "dequantize": dequant_out, "fused": fused_out}[name]
        print(
            f"{name:>11}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  {fraction_of_peak(gbs):>6.1%}  "
            f"{fp32_seconds / seconds:>6.2f}x  {cosine(result, exact):>8.5f}  {relative_error(result, exact):>9.2e}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

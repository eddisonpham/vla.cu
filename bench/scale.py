"""Iterative scaling: shapes, precision, and stack depth, grown until the GPU says stop.

Concept: a kernel has three ways to stop being fast, and each stage isolates one. Shapes grow
until the launch stops mattering and DRAM becomes the wall. Group size grows the scale table
until bytes per weight dominate accuracy. Depth grows the number of weight streams per token
until a block's cost is the whole model's cost. Nothing here is tuned; each row is the same
kernel with one number changed.

Task: python -m vla_cu scale
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from bench.real_layers import CHECKPOINT, MATRICES, cosine, pad_columns, relative_error, run_int4
from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import (
    PEAK_BANDWIDTH_GB_S,
    bandwidth_gb_s,
    enforce_memory_budget,
    fraction_of_peak,
    time_call,
    warm_gpu,
)
from vla_cu.blocks import MlpBlock
from vla_cu.checkpoint import open_checkpoint
from vla_cu.quant import GROUP_SIZE, dequantize_int4, pack_int4

SHAPES = ((256, 1024), (1024, 1024), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 16384), (16384, 32768))
GROUPS = (64, 32, 16, 8)
STACKS = (1, 2, 4, 8, 16)
EXPERT_LAYERS = 16
WARMUP = 20
ITERATIONS = 100
WORKING_CAP_BYTES = 4 * 1024**3
VERIFY_MAX_ELEMENTS = 8192 * 16384


def shape_sweep() -> None:
    """Time int4 at each shape in the grid, verifying where a dequantized copy still fits.

    Weights are random nibbles rather than a model, because only the byte count sets the
    expected time here. A shape whose working set would breach the cap is reported and skipped.
    """
    print("stage 1: shape sweep, int4 group 64, batch 1")
    print(f"{'N x K':>15}  {'MB':>8}  {'ms':>8}  {'GB/s':>8}  {'% DRAM':>7}  {'rel L2':>9}  {'checked':>7}")
    generator = torch.Generator(device="cuda").manual_seed(0)

    for n_out, n_in in SHAPES:
        working_bytes = n_out * n_in // 2 + 4 * n_out * (n_in // GROUP_SIZE) + 4 * n_out * n_in
        label = f"{n_out} x {n_in}"
        if working_bytes > WORKING_CAP_BYTES:
            print(f"{label:>15}  skipped, {working_bytes / 1e9:.2f} GB over the {WORKING_CAP_BYTES / 1e9:.0f} GB cap")
            continue

        x = torch.randn(n_in, device="cuda", generator=generator)
        packed = torch.randint(0, 256, (n_out, n_in // 2), dtype=torch.uint8, device="cuda", generator=generator)
        scales = torch.rand((n_out, n_in // GROUP_SIZE), device="cuda", generator=generator) * 0.01 + 0.001
        out = torch.empty(n_out, device="cuda")

        checked = n_out * n_in <= VERIFY_MAX_ELEMENTS
        error = float("nan")
        if checked:
            int4_gemv_into(packed, scales, x, out)
            exact = torch.mv(dequantize_int4(packed, scales), x)
            torch.cuda.synchronize()
            error = relative_error(out, exact)
            del exact

        n_bytes = packed.numel() + scales.numel() * 4
        seconds = time_call(lambda: int4_gemv_into(packed, scales, x, out), WARMUP, ITERATIONS)
        gbs = bandwidth_gb_s(n_bytes, seconds)
        print(
            f"{label:>15}  {n_bytes / 1e6:>8.1f}  {seconds * 1e3:>8.3f}  {gbs:>8.1f}  "
            f"{fraction_of_peak(gbs):>6.1%}  {error:>9.2e}  {'yes' if checked else 'no':>7}"
        )
        del x, packed, scales, out
    print()


def precision_ladder(checkpoint) -> None:
    """Quantize three real matrices at each group size in GROUPS and print bytes with cosine."""
    print("stage 2: precision ladder on real matrices, group size against bytes and cosine")
    print(f"{'layer':>18}  {'group':>6}  {'MB':>7}  {'ms':>8}  {'cos':>9}  {'rel L2':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)

    for label, name in MATRICES[:3]:
        w = checkpoint.read(name).to(device="cuda", dtype=torch.float32)
        x = torch.randn(w.shape[1], device="cuda", generator=generator)
        exact = torch.mv(w, x)
        for group_size in GROUPS:
            padded, activations = pad_columns(w, x, group_size)
            packed, scales = pack_int4(padded, group_size=group_size)
            out = torch.empty(padded.shape[0], device="cuda")
            result = run_int4(packed, scales, activations, out, group_size)
            torch.cuda.synchronize()
            n_bytes = packed.numel() + scales.numel() * 4
            seconds = time_call(
                lambda p=packed, s=scales, a=activations, o=out, g=group_size: run_int4(p, s, a, o, g),
                WARMUP,
                ITERATIONS,
            )
            print(
                f"{label:>18}  {group_size:>6}  {n_bytes / 1e6:>7.2f}  {seconds * 1e3:>8.3f}  "
                f"{cosine(result, exact):>9.5f}  {relative_error(result, exact):>9.2e}"
            )
        print()
        del w, x, exact
    print()


def stack_scale(checkpoint) -> None:
    """Run the 16 real expert MLPs in order and report time, bytes and cosine at each depth.

    The blocks are distinct layers, so depth means the real stack rather than one layer
    repeated. Attention and norms are absent, so this measures the weight stream per token.
    """
    print("stage 3: the real expert MLP stack, int8 up and int4 down")
    print(f"{'blocks':>7}  {'MB':>8}  {'ms':>8}  {'us/block':>9}  {'cos':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)
    source = []
    blocks = []
    for index in range(EXPERT_LAYERS):
        prefix = f"model.vlm_with_expert.lm_expert.layers.{index}.mlp"
        up = checkpoint.read(f"{prefix}.up_proj.weight").to(device="cuda", dtype=torch.float32)
        down = checkpoint.read(f"{prefix}.down_proj.weight").to(device="cuda", dtype=torch.float32)
        source.append((up, down))
        blocks.append(MlpBlock(up, down, down_bits=4))

    hidden = blocks[0].hidden_width
    x = torch.randn(hidden, device="cuda", generator=generator)
    buffers = (torch.empty(hidden, device="cuda"), torch.empty(hidden, device="cuda"))

    def chain(depth: int) -> torch.Tensor:
        """Run the first `depth` blocks in order, alternating two scratch buffers."""
        buffers[0].copy_(x)
        for index in range(depth):
            blocks[index].into(buffers[index % 2], buffers[(index + 1) % 2])
        return buffers[depth % 2]

    reference = x
    for depth in range(1, EXPERT_LAYERS + 1):
        up, down = source[depth - 1]
        reference = torch.mv(down, F.silu(torch.mv(up, reference)))
        if depth not in STACKS:
            continue
        result = chain(depth)
        torch.cuda.synchronize()
        seconds = time_call(lambda d=depth: chain(d), WARMUP, ITERATIONS)
        n_bytes = sum(block.bytes for block in blocks[:depth])
        print(
            f"{depth:>7}  {n_bytes / 1e6:>8.2f}  {seconds * 1e3:>8.3f}  "
            f"{seconds * 1e6 / depth:>9.2f}  {cosine(result, reference):>9.5f}"
        )
    print()


def main() -> int:
    """Run the three scaling stages in order, smallest first, inside the memory cap."""
    enforce_memory_budget()
    warm_gpu()
    print(f"GPU {torch.cuda.get_device_name(0)} | peak {PEAK_BANDWIDTH_GB_S:.0f} GB/s | working cap {WORKING_CAP_BYTES / 1e9:.0f} GB per shape")
    print()
    shape_sweep()
    checkpoint = open_checkpoint(CHECKPOINT)
    precision_ladder(checkpoint)
    stack_scale(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

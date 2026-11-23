"""What a packed model costs on disk, and what loading it costs.

Concept: a deployment quantizes once and then loads. A bundle that is smaller than the
fp16 checkpoint is the point, and a load that validates before returning is the price.
This task builds four action blocks at the two precision mixes the project ships, saves
them, loads them, and checks that every tensor and one full GEMV are unchanged.

File time is wall clock, because a disk read is not a GPU kernel. Kernel time is not
measured here.

Task: python -m vla_cu bundle
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from kernels.triton.int8_gemv import int8_gemv_into
from vla_cu.bench_utils import enforce_memory_budget, warm_gpu
from vla_cu.blocks import ActionBlock, AttentionBlock, MlpBlock
from vla_cu.bundle import Bundle, PackedMatrix, load_bundle, save_bundle

HIDDEN = 960
INTERMEDIATE = 2560
STACK_BLOCKS = 4
RESULTS = Path(__file__).resolve().parent / "results"
MIXES = ((4, 4, "int4 attn, int4 down"), (8, 8, "int8 attn, int8 down"))
REPEATS = 5


def make_blocks(attention_bits: int, down_bits: int) -> list[ActionBlock]:
    """Build `STACK_BLOCKS` seeded action blocks at one precision mix."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    scale = 1.0 / HIDDEN**0.5
    blocks = []
    for _ in range(STACK_BLOCKS):
        matrices = [torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator) * scale for _ in range(4)]
        up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * scale
        down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * scale
        blocks.append(
            ActionBlock(
                AttentionBlock(*matrices, bits=attention_bits),
                MlpBlock(up, down, down_bits=down_bits),
            )
        )
    return blocks


def bundle_for(blocks: list[ActionBlock]) -> Bundle:
    """Describe every packed matrix of `blocks` as a named bundle."""
    matrices: dict[str, PackedMatrix] = {}
    for index, block in enumerate(blocks):
        attention = block.attention
        mlp = block.mlp
        matrices[f"block{index}.qkv"] = PackedMatrix(attention.qkv, attention.qkv_scales, attention.bits, attention.group_size, 3 * HIDDEN, HIDDEN)
        matrices[f"block{index}.o"] = PackedMatrix(attention.o, attention.o_scales, attention.bits, attention.group_size, HIDDEN, HIDDEN)
        matrices[f"block{index}.up"] = PackedMatrix(mlp.up, mlp.up_scales, 8, mlp.group_size, INTERMEDIATE, HIDDEN)
        matrices[f"block{index}.down"] = PackedMatrix(mlp.down, mlp.down_scales, mlp.down_bits, mlp.group_size, HIDDEN, INTERMEDIATE)
    return Bundle(matrices, {"model": "action", "layers": str(len(blocks))})


def load_seconds(path: Path) -> float:
    """Return the fastest of `REPEATS` wall-clock loads, in seconds."""
    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        load_bundle(path)
        timings.append(time.perf_counter() - start)
    return min(timings)


def main() -> int:
    """Save, load and verify the two precision mixes, and print size and time."""
    enforce_memory_budget()
    warm_gpu()
    fp32_bytes = STACK_BLOCKS * (4 * HIDDEN * HIDDEN + 2 * INTERMEDIATE * HIDDEN) * 4
    fp16_bytes = fp32_bytes // 2
    print(f"GPU {torch.cuda.get_device_name(0)} | fp32 weights {fp32_bytes / 1e6:.1f} MB | fp16 {fp16_bytes / 1e6:.1f} MB")
    print(f"{STACK_BLOCKS} action blocks at {HIDDEN} -> {INTERMEDIATE} -> {HIDDEN}, file time is wall clock, best of {REPEATS}")
    print()
    print(f"{'mix':>22}  {'MB':>7}  {'vs fp32':>8}  {'vs fp16':>8}  {'save ms':>8}  {'load ms':>8}  {'exact':>6}  {'gemv':>6}")

    for attention_bits, down_bits, label in MIXES:
        blocks = make_blocks(attention_bits, down_bits)
        bundle = bundle_for(blocks)
        path = RESULTS / f"action_{attention_bits}_{down_bits}.bin"
        start = time.perf_counter()
        size = save_bundle(path, bundle)
        save_ms = (time.perf_counter() - start) * 1e3
        loaded = load_bundle(path)

        exact = all(
            torch.equal(loaded.matrices[name].data, matrix.data.cpu()) and torch.equal(loaded.matrices[name].scales, matrix.scales.cpu())
            for name, matrix in bundle.matrices.items()
        )
        original = bundle.matrices["block0.up"]
        restored = loaded.matrices["block0.up"]
        x = torch.randn(HIDDEN, device="cuda")
        reference = torch.empty(INTERMEDIATE, device="cuda")
        result = torch.empty(INTERMEDIATE, device="cuda")
        int8_gemv_into(original.data, original.scales, x, reference)
        int8_gemv_into(restored.data.cuda(), restored.scales.cuda(), x, result)
        gemv_exact = torch.equal(result, reference)

        print(
            f"{label:>22}  {size / 1e6:>7.2f}  {fp32_bytes / size:>7.2f}x  {fp16_bytes / size:>7.2f}x  "
            f"{save_ms:>8.2f}  {load_seconds(path) * 1e3:>8.2f}  {str(exact):>6}  {str(gemv_exact):>6}"
        )
        del blocks, bundle, loaded, reference, result, x

    print()
    print(f"files in {RESULTS}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

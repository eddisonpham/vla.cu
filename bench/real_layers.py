"""Int4 and int8 accuracy on real SmolVLA matrices, read straight from the checkpoint.

Concept: every accuracy number so far came from Gaussian weights with planted outliers.
The checkpoint decides the format: its expert matrices are 720 wide, which group 64 does
not divide, so padding to 768 and group 16 are measured side by side on the same weights.
Activations are not stored, so x is Gaussian and the reference is the bf16 weights in fp32.

Task: python -m vla_cu real
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int8_gemv import int8_gemv_into
from vla_cu.bench_utils import enforce_memory_budget, time_call
from vla_cu.checkpoint import open_checkpoint
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8
from vla_cu.rotate import hadamard, rotate_rows, rotate_vector

CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "smolvla_base" / "model.safetensors"
ROTATION_BLOCK = 64
WARMUP = 20
ITERATIONS = 100
MATRICES = (
    ("vlm.0.mlp.down", "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.down_proj.weight"),
    ("vlm.0.mlp.gate", "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.gate_proj.weight"),
    ("expert.0.mlp.down", "model.vlm_with_expert.lm_expert.layers.0.mlp.down_proj.weight"),
    ("expert.0.mlp.up", "model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj.weight"),
    ("expert.0.attn.o", "model.vlm_with_expert.lm_expert.layers.0.self_attn.o_proj.weight"),
)


@dataclass(frozen=True)
class Case:
    """One quantization path: how to run it, and the bytes it stores for this matrix."""

    label: str
    nbytes: int
    call: Callable[[], Tensor]


def cosine(approx: Tensor, exact: Tensor) -> float:
    """Return the cosine similarity of two vectors."""
    return torch.nn.functional.cosine_similarity(approx, exact, dim=0).item()


def relative_error(approx: Tensor, exact: Tensor) -> float:
    """Return the L2 norm of the error over the L2 norm of the reference."""
    return (torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)).item()


def pad_columns(w: Tensor, x: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
    """Zero-pad K until it divides by `group_size`; a zero weight contributes nothing."""
    pad = (-w.shape[1]) % group_size
    if pad == 0:
        return w, x
    return F.pad(w, (0, pad)), F.pad(x, (0, pad))


def run_int4(packed: Tensor, scales: Tensor, x: Tensor, out: Tensor, group_size: int) -> Tensor:
    """Launch the Triton int4 GEMV and return its output buffer."""
    int4_gemv_into(packed, scales, x, out, group_size=group_size)
    return out


def run_int8(codes: Tensor, scales: Tensor, x: Tensor, out: Tensor) -> Tensor:
    """Launch the Triton int8 GEMV and return its output buffer."""
    int8_gemv_into(codes, scales, x, out)
    return out


def int8_case(w: Tensor, x: Tensor) -> Case:
    """Quantize one matrix to int8 and describe the single case it has."""
    codes, scales = quantize_int8(w)
    out = torch.empty(codes.shape[0], device=w.device)
    return Case("int8", codes.numel() + scales.numel() * 4, lambda: run_int8(codes, scales, x, out))


def int4_cases(w: Tensor, x: Tensor) -> list[Case]:
    """Build the int4 cases: group 64, group 16, and a rotation in front of group 64."""
    cases: list[Case] = []
    for group_size, tag in ((GROUP_SIZE, "int4 g64"), (16, "int4 g16")):
        padded, activations = pad_columns(w, x, group_size)
        packed, scales = pack_int4(padded, group_size=group_size)
        out = torch.empty(padded.shape[0], device=w.device)
        label = f"{tag} pad{padded.shape[1]}" if padded.shape[1] != w.shape[1] else tag
        nbytes = packed.numel() + scales.numel() * 4
        cases.append(Case(label, nbytes, lambda p=packed, s=scales, a=activations, o=out, g=group_size: run_int4(p, s, a, o, g)))
    if w.shape[1] % ROTATION_BLOCK == 0:
        matrix = hadamard(ROTATION_BLOCK, device=str(w.device))
        rotated_w = rotate_rows(w, matrix)
        rotated_x = rotate_vector(x, matrix)
        packed, scales = pack_int4(rotated_w)
        out = torch.empty(rotated_w.shape[0], device=w.device)
        nbytes = packed.numel() + scales.numel() * 4
        cases.append(Case("int4 g64 rot", nbytes, lambda p=packed, s=scales, a=rotated_x, o=out: run_int4(p, s, a, o, GROUP_SIZE)))
    return cases


def main() -> int:
    """Read five real matrices and print accuracy and bytes for each quantization path."""
    enforce_memory_budget()
    checkpoint = open_checkpoint(CHECKPOINT)
    print(f"GPU {torch.cuda.get_device_name(0)} | checkpoint {CHECKPOINT.name} | batch 1")
    print(f"warmup {WARMUP} | iterations {ITERATIONS} | rotation block {ROTATION_BLOCK}")
    print()
    print(f"{'layer':>18}  {'N x K':>11}  {'path':>15}  {'MB':>7}  {'ms':>8}  {'cos':>9}  {'rel L2':>9}")
    generator = torch.Generator(device="cuda").manual_seed(0)

    for label, name in MATRICES:
        w = checkpoint.read(name).to(device="cuda", dtype=torch.float32)
        n_out, n_in = w.shape
        x = torch.randn(n_in, device="cuda", generator=generator)
        exact = torch.mv(w, x)
        for case in [int8_case(w, x)] + int4_cases(w, x):
            result = case.call()
            torch.cuda.synchronize()
            seconds = time_call(case.call, WARMUP, ITERATIONS)
            print(
                f"{label:>18}  {f'{n_out} x {n_in}':>11}  {case.label:>15}  {case.nbytes / 1e6:>7.2f}  "
                f"{seconds * 1e3:>8.3f}  {cosine(result, exact):>9.5f}  {relative_error(result, exact):>9.2e}"
            )
        print()
        del w, x, exact

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

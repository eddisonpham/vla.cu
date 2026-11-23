"""Does a Hadamard rotation make group-64 int4 accurate enough, at no byte cost?

Concept: 4 bits give 16 levels per group, and a group's scale is set by its largest
weight, so a single outlier forces every other weight in that group onto a coarse grid.
Rotating the weight matrix by a block Hadamard transform spreads each weight across the
group, which is the mechanism behind QuaRot and SpinQuant. The rotation is orthogonal,
so an exact product is unchanged; only the quantization error changes. At batch 1 the
transform is K log K work against K / 2 bytes, well under a percent of the kernel.

Two weight distributions are reported, because a rotation has little to do on Gaussian
weights and a lot to do once outliers exist, and real transformer weights have outliers.

Task: python -m vla_cu rotate
"""
from __future__ import annotations

import math

import torch

from kernels.triton.int4_gemv import int4_gemv_into
from vla_cu.bench_utils import enforce_memory_budget
from vla_cu.quant import GROUP_SIZE, pack_int4

BLOCK = 64
SHAPES = ((2560, 960), (1920, 768))
OUTLIER_FRACTION = 1e-3
OUTLIER_SCALE = 20.0


def hadamard(size: int) -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix, which is orthogonal and symmetric."""
    matrix = torch.ones((1, 1), device="cuda")
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0
        )
    return matrix / math.sqrt(size)


def rotate_rows(weights: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Rotate [N, K] weights along K in blocks of the matrix size."""
    n_rows, n_cols = weights.shape
    blocks = weights.reshape(n_rows, n_cols // matrix.shape[0], matrix.shape[0])
    return torch.matmul(blocks, matrix).reshape(n_rows, n_cols)


def rotate_vector(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Rotate a [K] activation vector in blocks of the matrix size."""
    blocks = vector.reshape(vector.numel() // matrix.shape[0], matrix.shape[0])
    return torch.matmul(blocks, matrix).reshape(-1)


def cosine(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the cosine similarity of two vectors."""
    return torch.nn.functional.cosine_similarity(approx, exact, dim=0).item()


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the L2 norm of the error over the L2 norm of the reference."""
    return (torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)).item()


def main() -> int:
    """Measure cosine and relative L2 for int4 with and without a block rotation."""
    enforce_memory_budget()
    matrix = hadamard(BLOCK)
    print(f"GPU {torch.cuda.get_device_name(0)} | block Hadamard {BLOCK} | group {GROUP_SIZE}")
    print(f"outliers: {OUTLIER_FRACTION:.0e} of weights scaled by {OUTLIER_SCALE:.0f}")
    print()
    print(f"{'N x K':>12}  {'weights':>9}  {'path':>9}  {'cos':>9}  {'rel L2':>9}")

    for n_out, n_in in SHAPES:
        generator = torch.Generator(device="cuda").manual_seed(0)
        scale = 1.0 / n_in**0.5
        for label in ("gaussian", "outliers"):
            w = torch.randn((n_out, n_in), device="cuda", generator=generator) * scale
            if label == "outliers":
                mask = torch.rand((n_out, n_in), device="cuda", generator=generator) < OUTLIER_FRACTION
                w = torch.where(mask, w * OUTLIER_SCALE, w)
            x = torch.randn(n_in, device="cuda", generator=generator)
            exact = torch.mv(w, x)

            packed, scales = pack_int4(w)
            out = torch.empty(n_out, device="cuda")
            int4_gemv_into(packed, scales, x, out)
            torch.cuda.synchronize()
            print(f"{f'{n_out} x {n_in}':>12}  {label:>9}  {'int4':>9}  {cosine(out, exact):>9.5f}  {relative_error(out, exact):>9.2e}")

            rotated_w = rotate_rows(w, matrix)
            rotated_x = rotate_vector(x, matrix)
            packed_r, scales_r = pack_int4(rotated_w)
            out_r = torch.empty(n_out, device="cuda")
            int4_gemv_into(packed_r, scales_r, rotated_x, out_r)
            torch.cuda.synchronize()
            print(f"{'':>12}  {label:>9}  {'+rotate':>9}  {cosine(out_r, exact):>9.5f}  {relative_error(out_r, exact):>9.2e}")
            print()
            del w, x, exact, packed, scales, out, rotated_w, rotated_x, packed_r, scales_r, out_r

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

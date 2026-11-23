"""Block Hadamard rotation, the accuracy trick that costs no bytes.

Concept: int4 gives 16 levels per group of 64 weights, and a group's scale is set by its
largest weight, so one outlier pushes the other 63 onto a coarse grid. A rotation by an
orthogonal matrix spreads each weight over its whole block, which flattens the outliers
before quantization. The product is unchanged because R R transpose is the identity, so
the transform costs arithmetic and no storage.

The Sylvester matrix is symmetric, so the same matrix rotates weights on the way into
the checkpoint and activations on the way into the kernel.
"""

from __future__ import annotations

import math

import torch


def hadamard(size: int, device: str = "cuda") -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix of order `size`, a power of two.

    The order is checked because the recursive doubling below would otherwise return the
    next power of two, and a caller would rotate blocks of the wrong width.
    """
    if size < 1 or size & (size - 1):
        raise ValueError(f"expected a positive power of two, got {size}")
    matrix = torch.ones((1, 1), device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0
        )
    return matrix / math.sqrt(size)


def rotate_rows(weights: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Rotate [N, K] weights along K in whole blocks of the matrix order.

    Assumptions: K is a multiple of the matrix order, so no block straddles a boundary.
    """
    n_rows, n_cols = weights.shape
    blocks = weights.reshape(n_rows, n_cols // matrix.shape[0], matrix.shape[0])
    return torch.matmul(blocks, matrix).reshape(n_rows, n_cols)


def rotate_vector(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Rotate a [K] activation vector in whole blocks of the matrix order."""
    blocks = vector.reshape(vector.numel() // matrix.shape[0], matrix.shape[0])
    return torch.matmul(blocks, matrix).reshape(-1)

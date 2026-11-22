"""Vector add in Triton: the first real kernel of the project.

Concept: a flat grid of programs, each owning one contiguous slice of the data,
plus a mask for the final partial block. Arithmetic is one add per element, so
the kernel is memory-bound: two reads and one write per element.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr) -> None:
    """Compute out[i] = x[i] + y[i] for i < n_elements on contiguous fp32 tensors.

    Mapping: program `pid` owns [pid*BLOCK, (pid+1)*BLOCK). At BLOCK=1024 each
    program reads two 4 KB chunks and writes one 4 KB chunk, all coalesced, so the
    bottleneck is memory bandwidth. The mask removes the padded lanes that the last
    program sees when n_elements is not a multiple of BLOCK.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return x + y elementwise through the Triton kernel.

    Assumptions: both tensors are contiguous, fp32, on the CUDA device, and have
    identical shapes. Any length is allowed, including 0 and lengths that are not
    a multiple of BLOCK_SIZE.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError(f"expected fp32 CUDA tensors, got {x.dtype} on {x.device}")
    if not (x.is_contiguous() and y.is_contiguous()):
        raise ValueError("expected contiguous tensors")

    n_elements = x.numel()
    out = torch.empty_like(x)
    if n_elements == 0:
        return out

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK=BLOCK_SIZE)
    return out

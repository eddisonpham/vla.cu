"""Custom operator wrappers for the fused GEMV kernels, registered with torch.

Concept: a raw Triton launch is a Python call, which the rest of torch cannot see.
`torch.library.custom_op` turns it into a real operator with a schema and a shape
function, so it composes with `torch.compile`, functionalization and autograd even
though the body stays a kernel launch. The fake implementation is what makes tracing
work: it returns a correctly shaped empty tensor without touching a GPU.

These ops are inference only and have no backward, so autograd registration is left
out on purpose.

Task: python -m vla_cu layer
"""

from __future__ import annotations

import torch
from torch import Tensor

from kernels.triton.int4_gemv import int4_gemv
from kernels.triton.int8_gemv import int8_gemv


@torch.library.custom_op("vla_cu::int8_gemv", mutates_args=())
def int8_gemv_op(w: Tensor, scales: Tensor, x: Tensor) -> Tensor:
    """Return scales * (w @ x) for int8 weights [N, K] and per-row fp32 scales [N]."""
    return int8_gemv(w, scales, x)


@int8_gemv_op.register_fake
def _(w: Tensor, scales: Tensor, x: Tensor) -> Tensor:
    """Describe the output shape to the tracing layer without running the kernel."""
    return w.new_empty((w.shape[0],), dtype=torch.float32)


@torch.library.custom_op("vla_cu::int4_gemv", mutates_args=())
def int4_gemv_op(packed: Tensor, scales: Tensor, x: Tensor) -> Tensor:
    """Return the dequantized GEMV of uint8 packed weights [N, K // 2] with group scales."""
    return int4_gemv(packed, scales, x)


@int4_gemv_op.register_fake
def _(packed: Tensor, scales: Tensor, x: Tensor) -> Tensor:
    """Describe the output shape to the tracing layer without running the kernel."""
    return torch.empty((packed.shape[0],), dtype=torch.float32, device=packed.device)

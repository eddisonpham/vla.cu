"""A 4-bit weight matrix that replaces one Linear layer's forward pass.

Concept: the project's whole claim is that weights can stay packed and be unpacked where
they are used, so a layer object should hold the packed bytes and nothing else. One token
dispatches to the GEMV, a chunk dispatches to the GEMM. Inference only: no bias, no
backward, fp32 activations, one CUDA device.
"""

from __future__ import annotations

import torch
from torch import Tensor

from kernels.triton.int4_gemm import int4_gemm
from kernels.triton.int4_gemv import int4_gemv
from vla_cu.quant import GROUP_SIZE, pack_int4
from vla_cu.validate import require


class Int4Linear(torch.nn.Module):
    """An [out_features, in_features] fp32 weight stored at 4 bits with group scales.

    `forward` reads fp32 [in_features] or [tokens, in_features] and returns
    [out_features] or [tokens, out_features]. The group size must divide in_features,
    because a group may not straddle a row's end. Buffers move with `.to()` like any
    module state.
    """

    def __init__(self, weight: Tensor, group_size: int = GROUP_SIZE) -> None:
        super().__init__()
        require(weight.dim() == 2, f"expected a 2D weight, got {tuple(weight.shape)}")
        require(weight.dtype == torch.float32, f"expected fp32 weights, got {weight.dtype}")
        require(group_size > 0 and group_size % 2 == 0, f"expected an even positive group size, got {group_size}")
        require(
            weight.shape[1] % group_size == 0,
            f"in_features {weight.shape[1]} is not divisible by group size {group_size}",
        )
        packed, scales = pack_int4(weight.detach().contiguous(), group_size=group_size)
        self.out_features, self.in_features = weight.shape
        self.group_size = group_size
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, group_size: int = GROUP_SIZE) -> Int4Linear:
        """Quantize a Linear's weight, dropping its bias, which this module does not model."""
        return cls(linear.weight, group_size=group_size)

    def forward(self, x: Tensor) -> Tensor:
        """Return x @ weight.T for x [in_features] as [out_features], or [tokens, in_features] as [tokens, out_features]."""
        require(x.dtype == torch.float32, f"expected fp32 activations, got {x.dtype}")
        require(x.device == self.packed.device, f"expected activations on {self.packed.device}, got {x.device}")
        require(x.shape[-1] == self.in_features, f"expected {self.in_features} in_features, got {x.shape[-1]}")
        values = x.contiguous()
        if values.dim() == 1:
            return int4_gemv(self.packed, self.scales, values, group_size=self.group_size)
        require(values.dim() == 2, f"expected [in_features] or [tokens, in_features], got {tuple(values.shape)}")
        return int4_gemm(self.packed, self.scales, values, group_size=self.group_size)

    def extra_repr(self) -> str:
        """Report the shape and format a reader needs to place the layer."""
        return f"in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size}"

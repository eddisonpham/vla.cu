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

    def graphed(self, x: Tensor | None = None) -> GraphedInt4Linear:
        """Capture this layer's launch, bound to `x` when the caller owns the input buffer."""
        sample = torch.empty(self.in_features, device=self.packed.device, dtype=torch.float32) if x is None else x
        return GraphedInt4Linear(self, sample)

    def extra_repr(self) -> str:
        """Report the shape and format a reader needs to place the layer."""
        return f"in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size}"


class GraphedInt4Linear:
    """One Int4Linear call captured into a CUDA graph and replayed through a static buffer.

    `__call__` copies the input into the captured buffer unless it already is that buffer,
    replays and returns the captured output, which the next call reuses, so clone it to
    keep it. Passing a buffer to `Int4Linear.graphed` binds the capture to the caller's own
    input and skips the copy. A graph only removes Python and Triton dispatch, so it pays
    off on small layers and does nothing on a DRAM bound one.
    """

    def __init__(self, layer: Int4Linear, sample: Tensor, warmup: int = 3) -> None:
        require(sample.dtype == torch.float32, f"expected fp32 activations, got {sample.dtype}")
        require(sample.device == layer.packed.device, f"expected activations on {layer.packed.device}, got {sample.device}")
        require(sample.shape[-1] == layer.in_features, f"expected {layer.in_features} in_features, got {sample.shape[-1]}")
        require(sample.dim() in (1, 2), f"expected [in_features] or [tokens, in_features], got {tuple(sample.shape)}")
        require(sample.is_contiguous(), "expected a contiguous input buffer for capture")
        self.layer = layer
        self.static_in = sample.detach()
        for _ in range(warmup):
            self.output = layer(self.static_in)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = layer(self.static_in)

    def __call__(self, x: Tensor) -> Tensor:
        """Replay the captured call for `x`, whose shape must match the capture."""
        require(
            x.shape == self.static_in.shape,
            f"graph captured a {tuple(self.static_in.shape)} input, got {tuple(x.shape)}",
        )
        require(x.dtype == torch.float32, f"expected fp32 activations, got {x.dtype}")
        require(x.device == self.static_in.device, f"expected activations on {self.static_in.device}, got {x.device}")
        if x.data_ptr() != self.static_in.data_ptr():
            self.static_in.copy_(x)
        return self.replay()

    def replay(self) -> Tensor:
        """Replay the graph and return the captured output, which the next call overwrites."""
        self.graph.replay()
        return self.output

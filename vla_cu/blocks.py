"""Attention and MLP blocks assembled from the packed kernels, at batch 1.

Concept: a block is a fixed set of projections plus one elementwise step, and every
projection is a packed matrix with its own scale layout. Holding the packed weights and
the scratch buffers in one object is what lets a benchmark, a CUDA graph and a future
model call the same code. Norms and residuals are not modelled, so a block here means
its projections only.

Attention at one visible token reduces to the value path, so the block is v = Wv x and
out = Wo v. The three qkv matrices share x, which is what the grouped launch exploits.

Task: python -m vla_cu block
"""

from __future__ import annotations

import torch
from torch import Tensor

from kernels.triton.int4_gemv import int4_gemv_into
from kernels.triton.int4_grouped_gemv import int4_grouped_gemv_into
from kernels.triton.int8_gemv import int8_gemv_into
from kernels.triton.int8_grouped_gemv import int8_grouped_gemv_into
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8
from vla_cu.validate import require

BITS = (4, 8)


def _pack(w: Tensor, bits: int, group_size: int) -> tuple[Tensor, Tensor]:
    """Quantize one fp32 matrix at `bits`, using the format that width owns."""
    if bits == 4:
        return pack_int4(w, group_size=group_size)
    return quantize_int8(w)


def matrix_bytes(rows: int, cols: int, bits: int, group_size: int) -> int:
    """Return the stored bytes of one matrix at `bits`, scales included."""
    weights = rows * cols * bits // 8
    scales = 4 * rows * (cols // group_size) if bits == 4 else 4 * rows
    return weights + scales


class MlpBlock:
    """A two-layer MLP: int8 up, SiLU in place, down at int4 or int8.

    `up` is [intermediate, hidden], `down` is [hidden, intermediate], both fp32. The
    intermediate vector is owned by the block, so `into` is not reentrant.
    """

    def __init__(self, up: Tensor, down: Tensor, down_bits: int = 4, group_size: int = GROUP_SIZE) -> None:
        require(up.dim() == 2 and down.dim() == 2, f"expected 2D matrices, got {tuple(up.shape)} and {tuple(down.shape)}")
        require(up.shape[0] == down.shape[1] and up.shape[1] == down.shape[0], f"up {tuple(up.shape)} and down {tuple(down.shape)} do not compose")
        require(down_bits in BITS, f"expected 4 or 8 bits, got {down_bits}")
        intermediate, hidden = up.shape
        self.hidden_width = hidden
        self.intermediate = intermediate
        self.down_bits = down_bits
        self.group_size = group_size
        self.up, self.up_scales = quantize_int8(up)
        self.down, self.down_scales = _pack(down, down_bits, group_size)
        self.hidden = torch.empty(intermediate, device=up.device, dtype=torch.float32)
        self.bytes = matrix_bytes(intermediate, hidden, 8, group_size) + matrix_bytes(hidden, intermediate, down_bits, group_size)

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write down(silu(up @ x)) into `out`, reading `x` [hidden] and reusing `hidden`."""
        int8_gemv_into(self.up, self.up_scales, x, self.hidden)
        torch.nn.functional.silu(self.hidden, inplace=True)
        if self.down_bits == 4:
            int4_gemv_into(self.down, self.down_scales, self.hidden, out, group_size=self.group_size)
        else:
            int8_gemv_into(self.down, self.down_scales, self.hidden, out)


class AttentionBlock:
    """q, k, v through one grouped launch, then o, at 4 or 8 bits.

    All four matrices are square [hidden, hidden]. At one visible token the softmax is
    the identity, so this is the value path and nothing else.
    """

    def __init__(self, wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor, bits: int = 8, group_size: int = GROUP_SIZE) -> None:
        require(bits in BITS, f"expected 4 or 8 bits, got {bits}")
        require(wq.dim() == 2 and wq.shape == wk.shape == wv.shape == wo.shape, f"expected four equal square matrices, got {tuple(wq.shape)}, {tuple(wk.shape)}, {tuple(wv.shape)}, {tuple(wo.shape)}")
        require(wq.shape[0] == wq.shape[1], f"expected square attention matrices, got {tuple(wq.shape)}")
        hidden = wq.shape[0]
        self.hidden_width = hidden
        self.bits = bits
        self.group_size = group_size
        self.qkv, self.qkv_scales = _pack(torch.cat((wq, wk, wv), dim=0), bits, group_size)
        self.o, self.o_scales = _pack(wo, bits, group_size)
        self.qkv_out = torch.empty(3 * hidden, device=wq.device, dtype=torch.float32)
        self.segment = torch.arange(3, device=wq.device, dtype=torch.int32).repeat_interleave(hidden)
        self.x_base = torch.zeros(3, device=wq.device, dtype=torch.int32)
        self.bytes = matrix_bytes(3 * hidden, hidden, bits, group_size) + matrix_bytes(hidden, hidden, bits, group_size)

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write Wo(Wv x) into `out`, reading `x` [hidden] and keeping the qkv vector."""
        if self.bits == 4:
            int4_grouped_gemv_into(self.qkv, self.qkv_scales, x, self.qkv_out, self.segment, self.x_base, group_size=self.group_size)
            int4_gemv_into(self.o, self.o_scales, self.qkv_out[2 * self.hidden_width :], out, group_size=self.group_size)
        else:
            int8_grouped_gemv_into(self.qkv, self.qkv_scales, x, self.qkv_out, self.segment, self.x_base)
            int8_gemv_into(self.o, self.o_scales, self.qkv_out[2 * self.hidden_width :], out)


class ActionBlock:
    """Attention then MLP, the projection stack of one action-expert layer."""

    def __init__(self, attention: AttentionBlock, mlp: MlpBlock) -> None:
        require(attention.hidden_width == mlp.hidden_width, f"attention width {attention.hidden_width} does not match MLP width {mlp.hidden_width}")
        self.attention = attention
        self.mlp = mlp
        self.hidden_width = attention.hidden_width
        self.intermediate = mlp.intermediate
        self.bytes = attention.bytes + mlp.bytes
        self.attention_out = torch.empty(attention.hidden_width, device=attention.qkv_out.device, dtype=torch.float32)

    def into(self, x: Tensor, out: Tensor) -> None:
        """Write mlp(attention(x)) into `out`, reading `x` [hidden]."""
        self.attention.into(x, self.attention_out)
        self.mlp.into(self.attention_out, out)

"""The SigLIP tower and pixel shuffle connector, fp32 or packed, for the image prefix rows.

Concept: a 512 by 512 frame is 32 by 32 patches of 16 pixels, 12 SigLIP layers turn each
patch into a 768 vector, and the connector folds every 4 by 4 patch block into one 12288
vector before projecting to 960. One frame therefore becomes 64 prefix rows. The tower is
read from the checkpoint once per call, and at 8 or 4 bits each projection is stored as
codes, so the same bytes serve three frames at a fraction of the compute.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from kernels.triton.int4_gemm import int4_gemm_into
from kernels.triton.int8_gemm import int8_gemm_into
from vla_cu.checkpoint import Checkpoint
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8
from vla_cu.validate import require

BASE = "model.vlm_with_expert.vlm.model.vision_model"
CONNECTOR = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
IMAGE = 512
PATCH = 16
SIDE = IMAGE // PATCH
SCALE = 4
EPS = 1e-6
LAYERS = 12
FRAME_ROWS = (SIDE * SIDE) // (SCALE**2)
BITS = (4, 8, 32)


def _read(checkpoint: Checkpoint, name: str) -> Tensor:
    """Return one checkpoint tensor as fp32 on the GPU."""
    return checkpoint.read(name).to(device="cuda", dtype=torch.float32)


def pixel_shuffle(x: Tensor, scale: int) -> Tensor:
    """Fold each scale by scale patch block into one row, in the connector's channel order.

    x is [batch, patches, channels] on a square grid and returns
    [batch, patches / scale ** 2, channels * scale ** 2]. Each row is the height sub block
    major, then the width sub block, then the channels, which is the order the real
    connector's view and permute produce.
    """
    batch, patches, channels = x.shape
    side = int(patches**0.5)
    require(side * side == patches, f"{patches} patches are not a square grid")
    require(side % scale == 0, f"grid side {side} is not divisible by scale {scale}")
    x = x.reshape(batch, side, side, channels)
    x = x.reshape(batch, side, side // scale, channels * scale)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(batch, side // scale, side // scale, channels * scale**2)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(batch, patches // (scale**2), channels * scale**2)


def _gelu(x: Tensor) -> Tensor:
    """Return the tanh approximation of GELU, the activation SigLIP was trained with."""
    return F.gelu(x, approximate="tanh")


def _layer_norm(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Return the layer norm of the last axis at SigLIP's epsilon."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, EPS)


class _Projection:
    """One tower matmul, held as fp32 weights or as packed codes with their scales.

    `bits` 32 keeps the fp32 matrix, 8 stores one code per byte with a scale per output
    row, and 4 stores two codes per byte with a scale per `group_size` inputs, so K must
    be whole groups. The bias stays fp32, because it is one add and packing it buys nothing.
    `bytes` is the stored size, which is what a caller counts when it compares precisions.
    """

    def __init__(self, weight: Tensor, bias: Tensor | None, bits: int, group_size: int = GROUP_SIZE) -> None:
        require(bits in BITS, f"expected 32, 8 or 4 bits, got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.out_features, self.in_features = weight.shape
        self.bias = (
            torch.zeros(self.out_features, device=weight.device, dtype=torch.float32) if bias is None else bias
        )
        if bits == 32:
            self.weight = weight
            self.bytes = weight.numel() * 4
        elif bits == 8:
            self.codes, self.scales = quantize_int8(weight)
            self.bytes = self.codes.numel() + self.scales.numel() * 4
        else:
            require(weight.shape[1] % group_size == 0, f"K {weight.shape[1]} is not whole {group_size} weight groups")
            self.codes, self.scales = pack_int4(weight, group_size=group_size)
            self.bytes = self.codes.numel() + self.scales.numel() * 4

    def __call__(self, x: Tensor) -> Tensor:
        """Return x @ weight.T + bias for x [tokens, in_features], as [tokens, out_features]."""
        if self.bits == 32:
            return x @ self.weight.T + self.bias
        out = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=torch.float32)
        if self.bits == 8:
            int8_gemm_into(self.codes, self.scales, x.contiguous(), out)
        else:
            int4_gemm_into(self.codes, self.scales, x.contiguous(), out, group_size=self.group_size)
        return out + self.bias


class Tower:
    """The 12 SigLIP layers and the connector, read once from the checkpoint at `bits`.

    One tower serves every frame of a prefix build, which is the difference between one
    weight read and one per frame. `bits` 32 is the fp32 reference; 8 and 4 pack each
    projection with this project's GEMM kernels and leave the norms and biases fp32.
    `bytes` reports the stored size of every matrix and norm the tower holds.
    """

    def __init__(self, checkpoint: Checkpoint, bits: int = 32, group_size: int = GROUP_SIZE) -> None:
        require(bits in BITS, f"expected 32, 8 or 4 bits, got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.patch_weight = _read(checkpoint, f"{BASE}.embeddings.patch_embedding.weight")
        self.patch_bias = _read(checkpoint, f"{BASE}.embeddings.patch_embedding.bias")
        self.position = _read(checkpoint, f"{BASE}.embeddings.position_embedding.weight")
        self.layers = [self._read_layer(checkpoint, index) for index in range(LAYERS)]
        self.post_weight = _read(checkpoint, f"{BASE}.post_layernorm.weight")
        self.post_bias = _read(checkpoint, f"{BASE}.post_layernorm.bias")
        self.connector = _Projection(_read(checkpoint, CONNECTOR), None, bits, group_size)
        self.bytes = self.patch_weight.numel() * 4 + self.patch_bias.numel() * 4 + self.position.numel() * 4
        self.bytes += self.post_weight.numel() * 4 + self.post_bias.numel() * 4 + self.connector.bytes
        self.bytes += sum(layer[index].bytes for layer in self.layers for index in (2, 3, 4, 5, 8, 9))
        self.bytes += sum(tensor.numel() * 4 for layer in self.layers for tensor in (layer[0], layer[1], layer[6], layer[7]))

    def _read_layer(self, checkpoint: Checkpoint, index: int) -> tuple:
        """Return one encoder layer as two norms and four projections, in run order."""
        prefix = f"{BASE}.encoder.layers.{index}"
        return (
            _read(checkpoint, f"{prefix}.layer_norm1.weight"),
            _read(checkpoint, f"{prefix}.layer_norm1.bias"),
            _Projection(
                _read(checkpoint, f"{prefix}.self_attn.q_proj.weight"),
                _read(checkpoint, f"{prefix}.self_attn.q_proj.bias"),
                self.bits,
                self.group_size,
            ),
            _Projection(
                _read(checkpoint, f"{prefix}.self_attn.k_proj.weight"),
                _read(checkpoint, f"{prefix}.self_attn.k_proj.bias"),
                self.bits,
                self.group_size,
            ),
            _Projection(
                _read(checkpoint, f"{prefix}.self_attn.v_proj.weight"),
                _read(checkpoint, f"{prefix}.self_attn.v_proj.bias"),
                self.bits,
                self.group_size,
            ),
            _Projection(
                _read(checkpoint, f"{prefix}.self_attn.out_proj.weight"),
                _read(checkpoint, f"{prefix}.self_attn.out_proj.bias"),
                self.bits,
                self.group_size,
            ),
            _read(checkpoint, f"{prefix}.layer_norm2.weight"),
            _read(checkpoint, f"{prefix}.layer_norm2.bias"),
            _Projection(
                _read(checkpoint, f"{prefix}.mlp.fc1.weight"),
                _read(checkpoint, f"{prefix}.mlp.fc1.bias"),
                self.bits,
                self.group_size,
            ),
            _Projection(
                _read(checkpoint, f"{prefix}.mlp.fc2.weight"),
                _read(checkpoint, f"{prefix}.mlp.fc2.bias"),
                self.bits,
                self.group_size,
            ),
        )

    def hidden(self, frame: Tensor) -> Tensor:
        """Return the tower's normalized patch outputs [1024, 768] for one frame [3, 512, 512].

        The patch embedding is a stride 16 convolution, positions are the row major table
        the checkpoint stores, and attention is full bidirectional over all 1024 patches.
        """
        patches = F.conv2d(frame[None], self.patch_weight, self.patch_bias, stride=PATCH)
        x = patches.flatten(2).transpose(1, 2)[0] + self.position
        for norm1_weight, norm1_bias, q_proj, k_proj, v_proj, out_proj, norm2_weight, norm2_bias, fc1, fc2 in self.layers:
            normed = _layer_norm(x, norm1_weight, norm1_bias)
            q = q_proj(normed).view(SIDE * SIDE, HEADS, HEAD_DIM).transpose(0, 1)
            k = k_proj(normed).view(SIDE * SIDE, HEADS, HEAD_DIM).transpose(0, 1)
            v = v_proj(normed).view(SIDE * SIDE, HEADS, HEAD_DIM).transpose(0, 1)
            weights = torch.softmax((q @ k.transpose(1, 2)) * (1.0 / math.sqrt(HEAD_DIM)), dim=-1)
            context = (weights @ v).transpose(0, 1).reshape(SIDE * SIDE, HIDDEN)
            x = x + out_proj(context)
            x = x + fc2(_gelu(fc1(_layer_norm(x, norm2_weight, norm2_bias))))
        return _layer_norm(x, self.post_weight, self.post_bias)

    def rows(self, frame: Tensor) -> Tensor:
        """Return the connector's 64 prefix rows [64, 960] for one frame [3, 512, 512]."""
        return self.connector(pixel_shuffle(self.hidden(frame)[None], SCALE)[0])


def embed_frame(checkpoint: Checkpoint, frame: Tensor, bits: int = 32) -> Tensor:
    """Return the 64 image rows [64, 960] for one frame [3, 512, 512] in [-1, 1]."""
    return Tower(checkpoint, bits).rows(frame)


def image_tokens(checkpoint: Checkpoint, frames: Tensor, bits: int = 32) -> Tensor:
    """Return the image prefix rows [frames, 64, 960], scaled by sqrt(960) as SmolVLA scales them.

    One tower is built for the whole batch, so the checkpoint is read once per call
    rather than once per frame.
    """
    tower = Tower(checkpoint, bits)
    rows = torch.stack([tower.rows(frame) for frame in frames])
    return rows * math.sqrt(rows.shape[-1])

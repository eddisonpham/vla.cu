"""The SigLIP tower and pixel shuffle connector, in fp32, for the image prefix rows.

Concept: a 512 by 512 frame is 32 by 32 patches of 16 pixels, 12 SigLIP layers turn each
patch into a 768 vector, and the connector folds every 4 by 4 patch block into one 12288
vector before projecting to 960. One frame therefore becomes 64 prefix rows. The weights
and the row order are the real checkpoint's; only the frames are synthetic.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from vla_cu.checkpoint import Checkpoint
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


def embed_frame(checkpoint: Checkpoint, frame: Tensor) -> Tensor:
    """Return the 64 image rows [64, 960] for one frame [3, 512, 512] in [-1, 1]."""
    patches = F.conv2d(
        frame[None],
        _read(checkpoint, f"{BASE}.embeddings.patch_embedding.weight"),
        _read(checkpoint, f"{BASE}.embeddings.patch_embedding.bias"),
        stride=PATCH,
    )
    x = patches.flatten(2).transpose(1, 2)[0] + _read(checkpoint, f"{BASE}.embeddings.position_embedding.weight")
    for index in range(LAYERS):
        prefix = f"{BASE}.encoder.layers.{index}"
        normed = _layer_norm(
            x, _read(checkpoint, f"{prefix}.layer_norm1.weight"), _read(checkpoint, f"{prefix}.layer_norm1.bias")
        )
        q = (normed @ _read(checkpoint, f"{prefix}.self_attn.q_proj.weight").T + _read(checkpoint, f"{prefix}.self_attn.q_proj.bias"))
        k = (normed @ _read(checkpoint, f"{prefix}.self_attn.k_proj.weight").T + _read(checkpoint, f"{prefix}.self_attn.k_proj.bias"))
        v = (normed @ _read(checkpoint, f"{prefix}.self_attn.v_proj.weight").T + _read(checkpoint, f"{prefix}.self_attn.v_proj.bias"))
        q = q.view(SIDE * SIDE, HEADS, HEAD_DIM).transpose(0, 1)
        k = k.view(SIDE * SIDE, HEADS, HEAD_DIM).transpose(0, 1)
        v = v.view(SIDE * SIDE, HEADS, HEAD_DIM).transpose(0, 1)
        weights = torch.softmax((q @ k.transpose(1, 2)) * (1.0 / math.sqrt(HEAD_DIM)), dim=-1)
        context = (weights @ v).transpose(0, 1).reshape(SIDE * SIDE, HIDDEN)
        x = x + context @ _read(checkpoint, f"{prefix}.self_attn.out_proj.weight").T + _read(checkpoint, f"{prefix}.self_attn.out_proj.bias")
        normed = _layer_norm(
            x, _read(checkpoint, f"{prefix}.layer_norm2.weight"), _read(checkpoint, f"{prefix}.layer_norm2.bias")
        )
        x = x + _gelu(normed @ _read(checkpoint, f"{prefix}.mlp.fc1.weight").T + _read(checkpoint, f"{prefix}.mlp.fc1.bias")) @ _read(
            checkpoint, f"{prefix}.mlp.fc2.weight"
        ).T + _read(checkpoint, f"{prefix}.mlp.fc2.bias")
    x = _layer_norm(
        x, _read(checkpoint, f"{BASE}.post_layernorm.weight"), _read(checkpoint, f"{BASE}.post_layernorm.bias")
    )
    rows = pixel_shuffle(x[None], SCALE)[0]
    return rows @ _read(checkpoint, CONNECTOR).T


def image_tokens(checkpoint: Checkpoint, frames: Tensor) -> Tensor:
    """Return the image prefix rows [frames, 64, 960], scaled by sqrt(960) as SmolVLA scales them."""
    rows = torch.stack([embed_frame(checkpoint, frame) for frame in frames])
    return rows * math.sqrt(rows.shape[-1])

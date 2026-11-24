"""The real 48 token VLM prefix as the key and value cache the expert cross attends to.

Concept: the expert never reads VLM hidden states. Each cross layer projects the cached
keys and values of the same backbone layer, which are post-rope and 320 wide. This runs
the 16 real backbone layers over 48 language rows, bidirectionally, and returns that
cache. No tokenizer or camera frame ships with the repo, so the rows are a fixed
vocabulary slice and the 192 image rows of a real inference prefix are absent.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from kernels.triton.rmsnorm import EPS
from vla_cu.blocks import gqa_mix, rotary
from vla_cu.checkpoint import Checkpoint
from vla_cu.vision import FRAME_ROWS, image_tokens

LAYERS = 16
TOKENS = 48
FRAMES = 3
FULL_TOKENS = FRAMES * FRAME_ROWS + TOKENS + 1
BASE = "model.vlm_with_expert.vlm.model.text_model"
STATE = "model.state_proj"
WEIGHTS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _read(checkpoint: Checkpoint, name: str) -> Tensor:
    """Return one checkpoint tensor as fp32 on the GPU."""
    return checkpoint.read(name).to(device="cuda", dtype=torch.float32)


def _norm(x: Tensor, weight: Tensor) -> Tensor:
    """Return the row wise RMSNorm of `x` scaled by `weight`."""
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS) * weight


def prefix_inputs(checkpoint: Checkpoint, tokens: int = TOKENS) -> Tensor:
    """Return the prefix rows [tokens, 960]: real embedding rows scaled as the prefill scales them."""
    embed = checkpoint.read(f"{BASE}.embed_tokens.weight")
    ids = torch.arange(tokens, dtype=torch.long)
    return embed.index_select(0, ids).to(device="cuda", dtype=torch.float32) * math.sqrt(embed.shape[1])


def cache_from_rows(
    checkpoint: Checkpoint, rows: Tensor, visible: Tensor | None = None, layers: int = LAYERS
) -> tuple[list[Tensor], list[Tensor]]:
    """Return post-rope keys and values for `rows` [tokens, 960] through every backbone layer.

    Row i is position i. Without `visible` the rows attend to each other in both directions,
    which is the language prefix. `rotary` runs at this repo's theta, not the 10000 in
    LeRobot's own `apply_rope`, so the cache matches this repo's expert queries.
    """
    x = rows
    keys: list[Tensor] = []
    values: list[Tensor] = []
    for index in range(layers):
        prefix = f"{BASE}.layers.{index}"
        weights = [_read(checkpoint, f"{prefix}.{name}") for name in WEIGHTS]
        input_norm, post_norm, wq, wk, wv, wo, gate, up, down = weights
        normed = _norm(x, input_norm)
        q = normed @ wq.T
        k = normed @ wk.T
        v = normed @ wv.T
        keys.append(rotary(k))
        values.append(v)
        x = x + gqa_mix(q, k, v, causal=False, visible=visible) @ wo.T
        normed = _norm(x, post_norm)
        x = x + (F.silu(normed @ gate.T) * (normed @ up.T)) @ down.T
    return keys, values


def prefix_cache(checkpoint: Checkpoint, tokens: int = TOKENS, layers: int = LAYERS) -> tuple[list[Tensor], list[Tensor]]:
    """Return post-rope keys and values [tokens, 320] for the 48 language rows at every layer."""
    return cache_from_rows(checkpoint, prefix_inputs(checkpoint, tokens), layers=layers)


def full_prefix_inputs(checkpoint: Checkpoint, frames: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Return the 241 prefix rows and their visibility mask: 3 frames, 48 language, 1 state.

    The state row sees every row and no other row sees it, which is the block mask the
    prefill's att_masks produce. `frames` is [3, 3, 512, 512] in [-1, 1] and `state` is a
    32 wide padded robot state, so both are inputs the caller supplies.
    """
    rows = torch.cat(
        (
            image_tokens(checkpoint, frames).reshape(-1, image_tokens(checkpoint, frames).shape[-1]),
            prefix_inputs(checkpoint, TOKENS),
            (state @ _read(checkpoint, f"{STATE}.weight").T + _read(checkpoint, f"{STATE}.bias"))[None],
        ),
        dim=0,
    )
    visible = torch.ones((rows.shape[0], rows.shape[0]), dtype=torch.bool, device=rows.device)
    visible[:-1, -1] = False
    return rows, visible

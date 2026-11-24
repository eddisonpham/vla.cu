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
from vla_cu.packfile import PACKED_LAYERS, load_pack, pack_path
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


_WEIGHTS: dict[tuple[str, int, int], list[list[Tensor]]] = {}


def backbone_weights(checkpoint: Checkpoint, layers: int = LAYERS) -> list[list[Tensor]]:
    """Return the fp32 tensors of every backbone layer, from the pack file when it exists.

    A prefill re-reads about 315 MB of bf16 to rebuild the key and value cache, so the
    layer weights are held the way `vision.tower_for` holds the tower. One entry, keyed by
    path, header length and layer count; a different key drops the previous set, which
    bounds the GPU memory kept between calls.
    """
    key = (str(checkpoint.path), checkpoint.header_bytes, layers)
    held = _WEIGHTS.get(key)
    if held is None:
        _WEIGHTS.clear()
        path = pack_path(checkpoint)
        if layers == PACKED_LAYERS and path.exists():
            state = load_pack(path)
            held = _WEIGHTS[key] = [
                [state[f"text.{index}.{name}"].to(device="cuda").to(torch.float32) for name in WEIGHTS]
                for index in range(layers)
            ]
        else:
            held = _WEIGHTS[key] = [
                [_read(checkpoint, f"{BASE}.layers.{index}.{name}") for name in WEIGHTS] for index in range(layers)
            ]
    return held


def weights_state(held: list[list[Tensor]]) -> dict[str, Tensor]:
    """Return held layer tensors named the way a pack file stores them, as bf16.

    The checkpoint stores these weights in bf16, so casting back is exact and the pack file
    carries half the bytes to the GPU that the fp32 cache holds.
    """
    return {
        f"text.{index}.{name}": tensor.to(torch.bfloat16)
        for index, layer in enumerate(held)
        for name, tensor in zip(WEIGHTS, layer)
    }


def forget_backbone() -> None:
    """Drop the held backbone layers, so the next call pays the read again."""
    _WEIGHTS.clear()


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
    held = backbone_weights(checkpoint, layers)
    for index in range(layers):
        input_norm, post_norm, wq, wk, wv, wo, gate, up, down = held[index]
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


def full_prefix_inputs(
    checkpoint: Checkpoint, frames: Tensor, state: Tensor, tower_bits: int = 32
) -> tuple[Tensor, Tensor]:
    """Return the 241 prefix rows and their visibility mask: 3 frames, 48 language, 1 state.

    The state row sees every row and no other row sees it, which is the block mask the
    prefill's att_masks produce. `frames` is [3, 3, 512, 512] in [-1, 1] and `state` is a
    32 wide padded robot state, so both are inputs the caller supplies. The tower runs
    once for all three frames, at `tower_bits`.
    """
    image_rows = image_tokens(checkpoint, frames, tower_bits)
    rows = torch.cat(
        (
            image_rows.reshape(-1, image_rows.shape[-1]),
            prefix_inputs(checkpoint, TOKENS),
            (state @ _read(checkpoint, f"{STATE}.weight").T + _read(checkpoint, f"{STATE}.bias"))[None],
        ),
        dim=0,
    )
    visible = torch.ones((rows.shape[0], rows.shape[0]), dtype=torch.bool, device=rows.device)
    visible[:-1, -1] = False
    return rows, visible

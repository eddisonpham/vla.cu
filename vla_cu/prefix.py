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

LAYERS = 16
TOKENS = 48
BASE = "model.vlm_with_expert.vlm.model.text_model"
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


def prefix_cache(checkpoint: Checkpoint, tokens: int = TOKENS, layers: int = LAYERS) -> tuple[list[Tensor], list[Tensor]]:
    """Return post-rope keys and values [tokens, 320] for each backbone layer.

    The prefix attends to itself in both directions, which is the mask the prefill pass
    uses, and row i is position i. `rotary` runs at this repo's theta, not the 10000 in
    LeRobot's own `apply_rope`, so the cache matches this repo's expert queries.
    """
    x = prefix_inputs(checkpoint, tokens)
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
        x = x + gqa_mix(q, k, v, causal=False) @ wo.T
        normed = _norm(x, post_norm)
        x = x + (F.silu(normed @ gate.T) * (normed @ up.T)) @ down.T
    return keys, values

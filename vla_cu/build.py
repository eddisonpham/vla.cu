"""Build a runnable block from one layer of a safetensors checkpoint.

Concept: a checkpoint stores projection matrices and a block is those matrices plus the
elementwise step between them, so the mapping is a name lookup and a cast to fp32. A
three-matrix MLP builds the gated SwiGLU block, which is the shape SmolVLA actually has.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from vla_cu.blocks import GatedMlpBlock, GqaTokenBlock, MlpBlock, MlpTokenBlock
from vla_cu.checkpoint import Checkpoint, open_checkpoint
from vla_cu.quant import GROUP_SIZE


def _open(checkpoint: Checkpoint | str | Path) -> Checkpoint:
    """Return the parsed checkpoint, reading only the header when given a path."""
    return checkpoint if isinstance(checkpoint, Checkpoint) else open_checkpoint(checkpoint)


def _matrix(checkpoint: Checkpoint, name: str, device: torch.device | str) -> Tensor:
    """Read one matrix from the checkpoint as contiguous fp32 on `device`."""
    return checkpoint.read(name).to(device=device, dtype=torch.float32).contiguous()


def mlp_from_checkpoint(
    checkpoint: Checkpoint | str | Path,
    layer: str,
    tokens: int = 1,
    down_bits: int = 4,
    group_size: int = GROUP_SIZE,
    device: torch.device | str = "cuda",
) -> MlpBlock | MlpTokenBlock | GatedMlpBlock:
    """Build the MLP at `layer`, the prefix that ends at the block.

    Reads up_proj and down_proj, plus gate_proj when the checkpoint has one, which selects
    the gated SwiGLU block. `tokens` 1 reads and writes [hidden], above that [tokens, hidden].
    """
    source = _open(checkpoint)
    up = _matrix(source, f"{layer}.up_proj.weight", device)
    down = _matrix(source, f"{layer}.down_proj.weight", device)
    if f"{layer}.gate_proj.weight" in source.tensors:
        gate = _matrix(source, f"{layer}.gate_proj.weight", device)
        return GatedMlpBlock(gate, up, down, down_bits=down_bits, group_size=group_size, tokens=tokens)
    if tokens == 1:
        return MlpBlock(up, down, down_bits=down_bits, group_size=group_size)
    return MlpTokenBlock(up, down, down_bits=down_bits, group_size=group_size, tokens=tokens)


def attention_from_checkpoint(
    checkpoint: Checkpoint | str | Path,
    layer: str,
    tokens: int = 1,
    causal: bool = True,
    head_dim: int = 64,
    device: torch.device | str = "cuda",
) -> GqaTokenBlock:
    """Build the attention block at `layer`, the prefix that ends at the attention module.

    Reads q_proj, k_proj, v_proj and o_proj as int8, with k and v allowed fewer heads than
    q. `tokens` fixes the buffer shapes, so a captured graph replays that token count.
    """
    source = _open(checkpoint)
    return GqaTokenBlock(
        _matrix(source, f"{layer}.q_proj.weight", device),
        _matrix(source, f"{layer}.k_proj.weight", device),
        _matrix(source, f"{layer}.v_proj.weight", device),
        _matrix(source, f"{layer}.o_proj.weight", device),
        head_dim=head_dim,
        tokens=tokens,
        causal=causal,
    )

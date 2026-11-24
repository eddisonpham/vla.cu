"""Build a runnable block from one layer of a safetensors checkpoint.

Concept: a checkpoint stores projection matrices and a block is those matrices plus the
elementwise step between them, so the mapping is a name lookup and a cast to fp32. A
layer index resolves against the two stacks a SmolVLA checkpoint carries: the SmolVLM2
backbone and the action expert. A three-matrix MLP builds the gated SwiGLU block, which
is the shape SmolVLA actually has.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from vla_cu.blocks import GatedMlpBlock, GqaTokenBlock, MlpBlock, MlpTokenBlock
from vla_cu.checkpoint import Checkpoint, open_checkpoint
from vla_cu.quant import GROUP_SIZE
from vla_cu.validate import require

PREFIXES: dict[str, str] = {
    "vlm": "model.vlm_with_expert.vlm.model.text_model.layers.{layer}",
    "expert": "model.vlm_with_expert.lm_expert.layers.{layer}",
}


def _open(checkpoint: Checkpoint | str | Path) -> Checkpoint:
    """Return the parsed checkpoint, reading only the header when given a path."""
    return checkpoint if isinstance(checkpoint, Checkpoint) else open_checkpoint(checkpoint)


def _matrix(checkpoint: Checkpoint, name: str, device: torch.device | str) -> Tensor:
    """Read one matrix from the checkpoint as contiguous fp32 on `device`."""
    return checkpoint.read(name).to(device=device, dtype=torch.float32).contiguous()


def _held_layers(source: Checkpoint, template: str) -> list[int]:
    """Return the layer indices the checkpoint stores under one naming scheme."""
    marker = template.split("{layer}")[0]
    indices = set()
    for name in source.tensors:
        if name.startswith(marker):
            head = name[len(marker) :].split(".", 1)[0]
            if head.isdigit():
                indices.add(int(head))
    return sorted(indices)


def layer_prefix(checkpoint: Checkpoint | str | Path, layer: int, part: str = "vlm") -> str:
    """Return the prefix of one backbone or action expert layer.

    `part` is \"vlm\" for the SmolVLM2 backbone or \"expert\" for the action expert, which
    share `model.vlm_with_expert` in a real checkpoint. A missing layer is refused here,
    with the range the checkpoint holds.
    """
    source = _open(checkpoint)
    require(part in PREFIXES, f"expected part 'vlm' or 'expert', got {part!r}")
    require(layer >= 0, f"expected a non-negative layer index, got {layer}")
    template = PREFIXES[part]
    held = _held_layers(source, template)
    require(layer in held, f"layer {layer} is not in the {part} stack, which holds layers {held}")
    return template.format(layer=layer)


def mlp_from_checkpoint(
    checkpoint: Checkpoint | str | Path,
    layer: str | int,
    tokens: int = 1,
    down_bits: int = 4,
    group_size: int = GROUP_SIZE,
    device: torch.device | str = "cuda",
    part: str = "vlm",
) -> MlpBlock | MlpTokenBlock | GatedMlpBlock:
    """Build the MLP at `layer`, a full prefix or an index under `part`.

    An index resolves through `layer_prefix`, so `part` selects the backbone or the action
    expert. Reads up_proj and down_proj, plus gate_proj when the checkpoint has one, which
    selects the gated SwiGLU block.
    """
    source = _open(checkpoint)
    layer = f"{layer_prefix(source, layer, part)}.mlp" if isinstance(layer, int) else layer
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
    layer: str | int,
    tokens: int = 1,
    causal: bool = True,
    head_dim: int = 64,
    device: torch.device | str = "cuda",
    part: str = "vlm",
) -> GqaTokenBlock:
    """Build the attention block at `layer`, a full prefix or an index under `part`.

    Reads q_proj, k_proj, v_proj and o_proj as int8, with k and v allowed fewer heads than
    q. `tokens` fixes the buffer shapes, so a captured graph replays that token count.
    """
    source = _open(checkpoint)
    if isinstance(layer, int):
        layer = f"{layer_prefix(source, layer, part)}.self_attn"
    return GqaTokenBlock(
        _matrix(source, f"{layer}.q_proj.weight", device),
        _matrix(source, f"{layer}.k_proj.weight", device),
        _matrix(source, f"{layer}.v_proj.weight", device),
        _matrix(source, f"{layer}.o_proj.weight", device),
        head_dim=head_dim,
        tokens=tokens,
        causal=causal,
    )

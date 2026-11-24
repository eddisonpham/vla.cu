"""A checkpoint layer becomes a runnable block, gated and plain, at one token and a chunk."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from vla_cu import attention_from_checkpoint, gqa_mix, layer_prefix, mlp_from_checkpoint, open_checkpoint

HIDDEN = 64
INTERMEDIATE = 128


def write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    """Write fp32 tensors to `path` in the safetensors layout, padding the header to 8 bytes."""
    header = {}
    offset = 0
    payload = bytearray()
    for name, tensor in tensors.items():
        data = tensor.contiguous().float().numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        payload += data
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)
    return path


def matrices(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gate, up and down fp32 matrices at the scale the benchmarks use."""
    generator = torch.Generator().manual_seed(seed)
    gate = torch.randn((INTERMEDIATE, HIDDEN), generator=generator) * (1.0 / HIDDEN**0.5)
    up = torch.randn((INTERMEDIATE, HIDDEN), generator=generator) * (1.0 / HIDDEN**0.5)
    down = torch.randn((HIDDEN, INTERMEDIATE), generator=generator) * (1.0 / INTERMEDIATE**0.5)
    return gate, up, down


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the cosine similarity of two flattened tensors."""
    return F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()


def test_a_gated_layer_builds_and_matches_the_swiglu_chain(tmp_path: Path) -> None:
    """gate_proj, up_proj and down_proj are the three matrices a real SmolVLA MLP stores."""
    gate, up, down = matrices(0)
    path = write_checkpoint(tmp_path / "model.safetensors", {
        "layers.0.mlp.gate_proj.weight": gate,
        "layers.0.mlp.up_proj.weight": up,
        "layers.0.mlp.down_proj.weight": down,
    })
    checkpoint = open_checkpoint(path)
    x = torch.randn(HIDDEN, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    out = torch.empty(HIDDEN, device="cuda")

    mlp_from_checkpoint(checkpoint, "layers.0.mlp").into(x, out)

    exact = down.cuda() @ (F.silu(gate.cuda() @ x) * (up.cuda() @ x))
    assert cosine(out, exact) > 0.99


def test_a_gated_layer_builds_a_chunk(tmp_path: Path) -> None:
    """A token count above one must switch to the GEMM path without a caller change."""
    gate, up, down = matrices(2)
    path = write_checkpoint(tmp_path / "model.safetensors", {
        "layers.1.mlp.gate_proj.weight": gate,
        "layers.1.mlp.up_proj.weight": up,
        "layers.1.mlp.down_proj.weight": down,
    })
    block = mlp_from_checkpoint(open_checkpoint(path), "layers.1.mlp", tokens=3)
    x = torch.randn((3, HIDDEN), device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
    out = torch.empty((3, HIDDEN), device="cuda")

    block.into(x, out)

    exact = (F.silu(x @ gate.cuda().T) * (x @ up.cuda().T)) @ down.cuda().T
    assert out.shape == (3, HIDDEN)
    assert cosine(out, exact) > 0.99


def test_a_two_matrix_layer_still_builds(tmp_path: Path) -> None:
    """A checkpoint without gate_proj must build the plain block, not fail."""
    up = torch.randn((INTERMEDIATE, HIDDEN), generator=torch.Generator().manual_seed(4)) * 0.1
    down = torch.randn((HIDDEN, INTERMEDIATE), generator=torch.Generator().manual_seed(5)) * 0.1
    path = write_checkpoint(tmp_path / "plain.safetensors", {
        "blk.mlp.up_proj.weight": up,
        "blk.mlp.down_proj.weight": down,
    })

    block = mlp_from_checkpoint(open_checkpoint(path), "blk.mlp")

    assert block.hidden_width == HIDDEN
    assert block.intermediate == INTERMEDIATE


def test_attention_builds_from_four_projections(tmp_path: Path) -> None:
    """The four attention matrices go through the same reader, grouped heads included."""
    generator = torch.Generator().manual_seed(6)
    wq = torch.randn((HIDDEN, HIDDEN), generator=generator) * 0.1
    wk = torch.randn((32, HIDDEN), generator=generator) * 0.1
    wv = torch.randn((32, HIDDEN), generator=generator) * 0.1
    wo = torch.randn((HIDDEN, HIDDEN), generator=generator) * 0.1
    path = write_checkpoint(tmp_path / "attn.safetensors", {
        "layers.2.self_attn.q_proj.weight": wq,
        "layers.2.self_attn.k_proj.weight": wk,
        "layers.2.self_attn.v_proj.weight": wv,
        "layers.2.self_attn.o_proj.weight": wo,
    })
    block = attention_from_checkpoint(open_checkpoint(path), "layers.2.self_attn", tokens=3, head_dim=32)
    x = torch.randn((3, HIDDEN), device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
    out = torch.empty((3, HIDDEN), device="cuda")

    block.into(x, None, out)

    exact = gqa_mix(x @ wq.cuda().T, x @ wk.cuda().T, x @ wv.cuda().T, head_dim=32) @ wo.cuda().T
    assert out.shape == (3, HIDDEN)
    assert cosine(out, exact) > 0.995


def test_an_index_resolves_the_backbone_and_the_expert_stacks(tmp_path: Path) -> None:
    """Both stacks live under model.vlm_with_expert, so `part` decides which one an index names."""
    small_gate, small_up, small_down = matrices(8)
    expert_gate = torch.randn((256, 128), generator=torch.Generator().manual_seed(9)) * 0.05
    expert_up = torch.randn((256, 128), generator=torch.Generator().manual_seed(10)) * 0.05
    expert_down = torch.randn((128, 256), generator=torch.Generator().manual_seed(11)) * 0.05
    path = write_checkpoint(tmp_path / "smolvla.safetensors", {
        "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.gate_proj.weight": small_gate,
        "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.weight": small_up,
        "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.down_proj.weight": small_down,
        "model.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj.weight": expert_gate,
        "model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj.weight": expert_up,
        "model.vlm_with_expert.lm_expert.layers.0.mlp.down_proj.weight": expert_down,
    })
    checkpoint = open_checkpoint(path)

    backbone = mlp_from_checkpoint(checkpoint, 0)
    expert = mlp_from_checkpoint(checkpoint, 0, part="expert")

    assert layer_prefix(checkpoint, 0) == "model.vlm_with_expert.vlm.model.text_model.layers.0"
    assert layer_prefix(checkpoint, 0, part="expert") == "model.vlm_with_expert.lm_expert.layers.0"
    assert (backbone.hidden_width, backbone.intermediate) == (HIDDEN, INTERMEDIATE)
    assert (expert.hidden_width, expert.intermediate) == (128, 256)


def test_attention_resolves_by_index(tmp_path: Path) -> None:
    """An index names the layer and the builder appends self_attn itself."""
    generator = torch.Generator().manual_seed(12)
    wq = torch.randn((HIDDEN, HIDDEN), generator=generator) * 0.1
    wk = torch.randn((32, HIDDEN), generator=generator) * 0.1
    wv = torch.randn((32, HIDDEN), generator=generator) * 0.1
    wo = torch.randn((HIDDEN, HIDDEN), generator=generator) * 0.1
    path = write_checkpoint(tmp_path / "expert.safetensors", {
        "model.vlm_with_expert.lm_expert.layers.2.self_attn.q_proj.weight": wq,
        "model.vlm_with_expert.lm_expert.layers.2.self_attn.k_proj.weight": wk,
        "model.vlm_with_expert.lm_expert.layers.2.self_attn.v_proj.weight": wv,
        "model.vlm_with_expert.lm_expert.layers.2.self_attn.o_proj.weight": wo,
    })

    block = attention_from_checkpoint(open_checkpoint(path), 2, part="expert", tokens=3, head_dim=32)

    assert block.tokens == 3
    assert block.hidden_width == HIDDEN


@pytest.mark.cpu
def test_an_index_outside_the_stack_names_the_range(tmp_path: Path) -> None:
    """A wrong index must name the layers that exist, before any matrix is read."""
    up = torch.zeros((INTERMEDIATE, HIDDEN))
    down = torch.zeros((HIDDEN, INTERMEDIATE))
    path = write_checkpoint(tmp_path / "one.safetensors", {
        "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.weight": up,
        "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.down_proj.weight": down,
    })

    with pytest.raises(ValueError, match="layer 7 is not in the vlm stack"):
        mlp_from_checkpoint(open_checkpoint(path), 7)


@pytest.mark.cpu
def test_an_unknown_stack_is_refused(tmp_path: Path) -> None:
    """An unknown `part` must fail with the two names it accepts, not a KeyError later."""
    up = torch.zeros((INTERMEDIATE, HIDDEN))
    down = torch.zeros((HIDDEN, INTERMEDIATE))
    path = write_checkpoint(tmp_path / "named.safetensors", {
        "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj.weight": up,
        "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.down_proj.weight": down,
    })

    with pytest.raises(ValueError, match="part 'vlm' or 'expert'"):
        mlp_from_checkpoint(open_checkpoint(path), 0, part="vision")


def test_a_prefix_without_the_block_raises_a_key_error(tmp_path: Path) -> None:
    """A wrong prefix must name the missing tensor, not build a block of zeros."""
    up = torch.zeros((INTERMEDIATE, HIDDEN))
    down = torch.zeros((HIDDEN, INTERMEDIATE))
    path = write_checkpoint(tmp_path / "named.safetensors", {
        "blk.mlp.up_proj.weight": up,
        "blk.mlp.down_proj.weight": down,
    })

    with pytest.raises(KeyError, match="up_proj"):
        mlp_from_checkpoint(open_checkpoint(path), "other.mlp")

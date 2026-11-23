"""The block objects against an fp32 chain on the same weights."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vla_cu.blocks import ActionBlock, AttentionBlock, MlpBlock


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the cosine similarity of two vectors."""
    return F.cosine_similarity(a, b, dim=0).item()


def mlp_matrices(hidden: int, intermediate: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return up and down fp32 matrices at the same scale the benchmarks use."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scale = 1.0 / hidden**0.5
    up = torch.randn((intermediate, hidden), device="cuda", generator=generator) * scale
    down = torch.randn((hidden, intermediate), device="cuda", generator=generator) * scale
    return up, down


def attention_matrices(hidden: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return q, k, v and o fp32 matrices at the benchmark scale."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scale = 1.0 / hidden**0.5
    return tuple(torch.randn((hidden, hidden), device="cuda", generator=generator) * scale for _ in range(4))


def test_mlp_matches_the_fp32_chain() -> None:
    """int8 up, SiLU and int4 down must stay close to the dense product."""
    up, down = mlp_matrices(64, 128, 0)
    x = torch.randn(64, device="cuda")
    out = torch.empty(64, device="cuda")
    MlpBlock(up, down).into(x, out)

    exact = down @ F.silu(up @ x)
    assert cosine(out, exact) > 0.99


def test_int8_down_is_closer_than_int4_down() -> None:
    """The extra byte must buy accuracy on the same weights, or the format choice is noise."""
    up, down = mlp_matrices(64, 128, 1)
    x = torch.randn(64, device="cuda")
    out4 = torch.empty(64, device="cuda")
    out8 = torch.empty(64, device="cuda")
    MlpBlock(up, down, down_bits=4).into(x, out4)
    MlpBlock(up, down, down_bits=8).into(x, out8)

    exact = down @ F.silu(up @ x)
    assert cosine(out8, exact) > cosine(out4, exact)


@pytest.mark.parametrize("bits", (4, 8))
def test_attention_matches_the_value_path(bits: int) -> None:
    """At one token the softmax is the identity, so only Wv and Wo may affect the result."""
    wq, wk, wv, wo = attention_matrices(64, 2)
    x = torch.randn(64, device="cuda")
    out = torch.empty(64, device="cuda")
    AttentionBlock(wq, wk, wv, wo, bits=bits).into(x, out)

    assert cosine(out, wo @ (wv @ x)) > (0.98 if bits == 4 else 0.999)


def test_action_block_composes_attention_and_mlp() -> None:
    """The two stages must pass one hidden vector between them without a shape slip."""
    wq, wk, wv, wo = attention_matrices(64, 3)
    up, down = mlp_matrices(64, 128, 4)
    x = torch.randn(64, device="cuda")
    out = torch.empty(64, device="cuda")
    ActionBlock(AttentionBlock(wq, wk, wv, wo, bits=8), MlpBlock(up, down, down_bits=8)).into(x, out)

    exact = down @ F.silu(up @ (wo @ (wv @ x)))
    assert cosine(out, exact) > 0.995


def test_rejects_matrices_that_do_not_compose() -> None:
    """A mismatched intermediate width must fail at construction, not at the first launch."""
    up, down = mlp_matrices(64, 128, 5)
    with pytest.raises(ValueError):
        MlpBlock(up, down[:, :-1])

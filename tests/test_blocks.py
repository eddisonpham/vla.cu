"""The block objects against an fp32 chain on the same weights."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernels.triton.gqa_attention import gqa_attention_into
from kernels.triton.rmsnorm import add_into, add_norm_into, rmsnorm_into
from vla_cu.blocks import (
    ActionBlock,
    AttentionBlock,
    GqaTokenBlock,
    MlpBlock,
    MlpTokenBlock,
    gqa_mix,
    rotary,
    rotary_tables,
)


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


def test_mlp_token_block_matches_the_fp32_chain() -> None:
    """The chunk MLP must hold the same chain as the one-token block, GEMMs or not."""
    up, down = mlp_matrices(64, 128, 12)
    x = torch.randn((3, 64), device="cuda")
    out = torch.empty((3, 64), device="cuda")
    MlpTokenBlock(up, down, tokens=3).into(x, out)

    exact = F.silu(x @ up.T) @ down.T
    assert out.shape == (3, 64)
    assert cosine(out.reshape(-1), exact.reshape(-1)) > 0.99


def test_mlp_token_block_agrees_with_the_token_loop() -> None:
    """The GEMM rounds activations to fp16 at the dot, so it is close, not identical, to the GEMV."""
    up, down = mlp_matrices(64, 128, 13)
    x = torch.randn((2, 64), device="cuda")
    chunk = torch.empty((2, 64), device="cuda")
    loop = torch.empty((2, 64), device="cuda")
    MlpTokenBlock(up, down, tokens=2).into(x, chunk)
    block = MlpBlock(up, down)
    for token in range(2):
        block.into(x[token], loop[token])

    assert cosine(chunk.reshape(-1), loop.reshape(-1)) > 0.9999


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


def test_add_norm_matches_the_two_pass_chain() -> None:
    """One launch must equal the residual add then the norm, or the fusion is a bug."""
    generator = torch.Generator(device="cuda").manual_seed(14)
    a = torch.randn(128, device="cuda", generator=generator)
    b = torch.randn(128, device="cuda", generator=generator)
    weight = torch.randn(128, device="cuda", generator=generator)
    added = torch.empty(128, device="cuda")
    normed = torch.empty(128, device="cuda")
    expected_added = torch.empty(128, device="cuda")
    expected_normed = torch.empty(128, device="cuda")
    add_into(a, b, expected_added)
    rmsnorm_into(expected_added, weight, expected_normed)

    add_norm_into(a, b, weight, added, normed)

    torch.testing.assert_close(added, expected_added, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(normed, expected_normed, rtol=1e-6, atol=1e-6)


def test_rotary_turns_row_one_and_leaves_row_zero() -> None:
    """Position 0 is the identity rotation, which is what lets a one token block skip it."""
    generator = torch.Generator(device="cuda").manual_seed(6)
    x = torch.randn((2, 15 * 64), device="cuda", generator=generator)

    turned = rotary(x)

    torch.testing.assert_close(turned[0], x[0])
    assert not torch.allclose(turned[1], x[1])
    torch.testing.assert_close(turned.norm(dim=1), x.norm(dim=1), rtol=1e-5, atol=1e-5)


def test_rotary_moves_a_score_only_when_the_positions_differ() -> None:
    """A rotation preserves q.k for one position and moves it for two, which is relative position."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn((2, 64), device="cuda", generator=generator)
    k = torch.randn((2, 64), device="cuda", generator=generator)

    plain = q @ k.T
    turned = rotary(q) @ rotary(k).T

    torch.testing.assert_close(turned.diagonal(), plain.diagonal(), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(turned[0, 1], plain[0, 1])


def test_gqa_token_block_mixes_three_keys() -> None:
    """Three tokens put three keys in every softmax, so the value path alone cannot match."""
    generator = torch.Generator(device="cuda").manual_seed(8)
    scale = 1.0 / 16**0.5
    wq = torch.randn((16, 16), device="cuda", generator=generator) * scale
    wk = torch.randn((8, 16), device="cuda", generator=generator) * scale
    wv = torch.randn((8, 16), device="cuda", generator=generator) * scale
    wo = torch.randn((32, 16), device="cuda", generator=generator) * scale
    x = torch.randn((3, 16), device="cuda", generator=generator)
    out = torch.empty((3, 32), device="cuda")

    GqaTokenBlock(wq, wk, wv, wo, head_dim=8, tokens=3).into(x, None, out)

    exact = gqa_mix(x @ wq.T, x @ wk.T, x @ wv.T, head_dim=8) @ wo.T
    assert out.shape == (3, 32)
    assert cosine(out.reshape(-1), exact.reshape(-1)) > 0.995


def test_gqa_mix_masks_future_keys() -> None:
    """A causal block lets token 0 see one key, so its context is that key's value alone."""
    generator = torch.Generator(device="cuda").manual_seed(9)
    q = torch.randn((3, 16), device="cuda", generator=generator)
    k = torch.randn((3, 8), device="cuda", generator=generator)
    v = torch.randn((3, 8), device="cuda", generator=generator)

    causal = gqa_mix(q, k, v, head_dim=8)
    full = gqa_mix(q, k, v, head_dim=8, causal=False)

    torch.testing.assert_close(causal[0], v[0].repeat(2))
    assert not torch.allclose(full[0], v[0].repeat(2))


@pytest.mark.parametrize("tokens", (3, 5, 8, 17))
def test_attention_kernel_matches_the_einsum_reference(tokens: int) -> None:
    """The kernel must reproduce rope and the masked softmax over positions, not heads."""
    generator = torch.Generator(device="cuda").manual_seed(10)
    heads, kv_heads, head_dim = 4, 2, 8
    q = torch.randn((tokens, heads * head_dim), device="cuda", generator=generator)
    k = torch.randn((tokens, kv_heads * head_dim), device="cuda", generator=generator)
    v = torch.randn((tokens, kv_heads * head_dim), device="cuda", generator=generator)
    cosine_table, sine_table = rotary_tables(tokens, head_dim)

    out = torch.empty_like(q)
    gqa_attention_into(q, k, v, cosine_table, sine_table, out, head_dim=head_dim)

    torch.testing.assert_close(out, gqa_mix(q, k, v, head_dim=head_dim), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tokens", (8, 65))
def test_gqa_token_block_above_the_loop_matches_the_reference(tokens: int) -> None:
    """A chunk past the projection loop must cross the same chain, fused mix or not."""
    generator = torch.Generator(device="cuda").manual_seed(11)
    scale = 1.0 / 16**0.5
    wq = torch.randn((16, 16), device="cuda", generator=generator) * scale
    wk = torch.randn((8, 16), device="cuda", generator=generator) * scale
    wv = torch.randn((8, 16), device="cuda", generator=generator) * scale
    wo = torch.randn((32, 16), device="cuda", generator=generator) * scale
    x = torch.randn((tokens, 16), device="cuda", generator=generator)
    out = torch.empty((tokens, 32), device="cuda")

    GqaTokenBlock(wq, wk, wv, wo, head_dim=8, tokens=tokens).into(x, None, out)

    exact = gqa_mix(x @ wq.T, x @ wk.T, x @ wv.T, head_dim=8) @ wo.T
    assert out.shape == (tokens, 32)
    assert cosine(out.reshape(-1), exact.reshape(-1)) > 0.995

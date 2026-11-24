"""The cooperative layer tails against the eager launch chains, across widths and tokens.

The fused kernels have their own thread mapping and a grid barrier, and the gated one
computes both projections in one pass, so each needs a correctness gate of its own.
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from kernels.triton.rmsnorm import add_into, add_norm_into
from vla_cu.blocks import GatedMlpBlock, MlpTokenBlock
from vla_cu.cuda_coop import GROUP, HIDDEN, INTERMEDIATE, MAX_TOKENS, coop_gated_into, coop_layer_into
from vla_cu.quant import pack_int4, quantize_int8

HOST_COMPILER = (
    shutil.which("cl")
    if os.name == "nt"
    else shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
)
pytestmark = pytest.mark.skipif(
    shutil.which("nvcc") is None or HOST_COMPILER is None,
    reason="no CUDA build toolchain on PATH; on Windows, run inside vcvars64.bat",
)

TOKENS = (1, 2, 3, 8, 16, 17, 50, 64)
WIDTHS = ((256, 640, 1), (256, 640, 3), (256, 640, 17), (384, 768, 5))


def compare(tokens: int, hidden: int, intermediate: int) -> None:
    """Run the five launch path and the fused path on the same weights and compare."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(hidden, device="cuda", generator=generator) + 1.0
    up = torch.randn((intermediate, hidden), device="cuda", generator=generator) * (1.0 / hidden**0.5)
    down = torch.randn((hidden, intermediate), device="cuda", generator=generator) * (1.0 / intermediate**0.5)
    x = torch.randn((tokens, hidden), device="cuda", generator=generator)
    other = torch.randn((tokens, hidden), device="cuda", generator=generator)
    added = torch.empty_like(x)
    normed = torch.empty_like(x)
    tail = torch.empty_like(x)
    expected = torch.empty_like(x)
    add_norm_into(x, other, weight, added, normed)
    MlpTokenBlock(up, down, down_bits=4, group_size=GROUP, tokens=tokens).into(normed, tail)
    add_into(added, tail, expected)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP)
    fused_added = torch.empty_like(x)
    fused_normed = torch.empty_like(x)
    fused_out = torch.empty_like(x)
    act = torch.empty((tokens, intermediate), device="cuda")

    coop_layer_into(
        x, other, weight, up_codes, up_scales, down_packed, down_scales,
        fused_added, fused_normed, act, fused_out,
    )

    torch.testing.assert_close(fused_added, added, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fused_normed, normed, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tokens", TOKENS)
def test_matches_the_five_launch_path(tokens: int) -> None:
    """One launch must produce the same residual, norm and output as the five."""
    compare(tokens, HIDDEN, INTERMEDIATE)


@pytest.mark.parametrize(("hidden", "intermediate", "tokens"), WIDTHS)
def test_serves_another_layer_width(hidden: int, intermediate: int, tokens: int) -> None:
    """Widths arrive as launch parameters, so a second layer shape must work unchanged."""
    compare(tokens, hidden, intermediate)


def compare_gated(tokens: int, hidden: int, intermediate: int) -> None:
    """Run the six launch gated chain and the fused gated tail on the same weights."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(hidden, device="cuda", generator=generator) + 1.0
    gate = torch.randn((intermediate, hidden), device="cuda", generator=generator) * (1.0 / hidden**0.5)
    up = torch.randn((intermediate, hidden), device="cuda", generator=generator) * (1.0 / hidden**0.5)
    down = torch.randn((hidden, intermediate), device="cuda", generator=generator) * (1.0 / intermediate**0.5)
    x = torch.randn((tokens, hidden), device="cuda", generator=generator)
    other = torch.randn((tokens, hidden), device="cuda", generator=generator)
    added = torch.empty_like(x)
    normed = torch.empty_like(x)
    tail = torch.empty_like(x)
    expected = torch.empty_like(x)
    add_norm_into(x, other, weight, added, normed)
    block = GatedMlpBlock(gate, up, down, down_bits=4, group_size=GROUP, tokens=tokens)
    if tokens == 1:
        block.into(normed[0], tail[0])
    else:
        block.into(normed, tail)
    add_into(added, tail, expected)
    gate_codes, gate_scales = quantize_int8(gate)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP)
    fused_added = torch.empty_like(x)
    fused_normed = torch.empty_like(x)
    fused_out = torch.empty_like(x)
    act = torch.empty((tokens, intermediate), device="cuda")

    coop_gated_into(
        x, other, weight, gate_codes, gate_scales, up_codes, up_scales,
        down_packed, down_scales, fused_added, fused_normed, act, fused_out,
    )

    torch.testing.assert_close(fused_added, added, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fused_normed, normed, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(("tokens", "hidden", "intermediate"), ((1, 256, 640), (3, 256, 640), (17, 384, 768)))
def test_gated_matches_the_six_launch_path(tokens: int, hidden: int, intermediate: int) -> None:
    """The gated stage computes two projections in one pass, so it needs its own gate."""
    compare_gated(tokens, hidden, intermediate)


def test_gated_rejects_a_gate_that_does_not_match_up() -> None:
    """A gate and an up of different heights would write a mismatched act, so the host refuses."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    tokens, hidden, intermediate = 2, 256, 640
    x = torch.randn((tokens, hidden), device="cuda", generator=generator)
    other = torch.randn((tokens, hidden), device="cuda", generator=generator)
    weight = torch.randn(hidden, device="cuda", generator=generator)
    gate = torch.empty((intermediate, hidden), device="cuda", dtype=torch.int8)
    gate_scales = torch.empty(intermediate, device="cuda")
    up = torch.empty((intermediate // 2, hidden), device="cuda", dtype=torch.int8)
    up_scales = torch.empty(intermediate // 2, device="cuda")
    down = torch.empty((hidden, intermediate // 2), device="cuda", dtype=torch.uint8)
    down_scales = torch.empty((hidden, intermediate // GROUP), device="cuda")

    with pytest.raises(ValueError):
        coop_gated_into(
            x, other, weight, gate, gate_scales, up, up_scales, down, down_scales,
            torch.empty_like(x), torch.empty_like(x), torch.empty((tokens, intermediate), device="cuda"),
            torch.empty_like(x),
        )


def test_rejects_a_width_the_tiles_cannot_serve() -> None:
    """The K loop steps by 16, so a hidden width that does not divide by 16 must be refused."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    tokens, hidden, intermediate = 2, 200, 256
    x = torch.randn((tokens, hidden), device="cuda", generator=generator)
    other = torch.empty_like(x)
    weight = torch.randn(hidden, device="cuda", generator=generator)
    up = torch.randn((intermediate, hidden), device="cuda", generator=generator)
    down = torch.randn((hidden, intermediate), device="cuda", generator=generator)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP)

    with pytest.raises(ValueError):
        coop_layer_into(
            x, other, weight, up_codes, up_scales, down_packed, down_scales,
            torch.empty_like(x), torch.empty_like(x), torch.empty((tokens, intermediate), device="cuda"),
            torch.empty_like(x),
        )


def test_rejects_a_non_contiguous_input() -> None:
    """The kernel reads float4 words, so a transposed view must be refused, not misread."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((HIDDEN, 4), device="cuda", generator=generator).t()
    other = torch.randn((4, HIDDEN), device="cuda", generator=generator)
    weight = torch.randn(HIDDEN, device="cuda", generator=generator)
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator)
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP)

    with pytest.raises(RuntimeError):
        coop_layer_into(
            x, other, weight, up_codes, up_scales, down_packed, down_scales,
            torch.empty((4, HIDDEN), device="cuda"), torch.empty((4, HIDDEN), device="cuda"),
            torch.empty((4, INTERMEDIATE), device="cuda"), torch.empty((4, HIDDEN), device="cuda"),
        )


def test_rejects_more_tokens_than_the_kernel_holds() -> None:
    """The token loops are bounded by the compile-time maximum, so the host must refuse more."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    tokens = MAX_TOKENS + 1
    x = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
    other = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
    weights = torch.randn(HIDDEN, device="cuda", generator=generator)
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator)
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP)

    with pytest.raises(ValueError):
        coop_layer_into(
            x, other, weights, up_codes, up_scales, down_packed, down_scales,
            torch.empty_like(x), torch.empty_like(x), torch.empty((tokens, INTERMEDIATE), device="cuda"),
            torch.empty_like(x),
        )

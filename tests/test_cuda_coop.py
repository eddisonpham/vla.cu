"""The cooperative layer tail against the five launch path, including awkward token counts.

The fused kernel has its own thread mapping and a grid barrier, so it needs a correctness
gate of its own rather than a comparison against the Triton kernel alone.
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from kernels.triton.rmsnorm import add_into, add_norm_into
from vla_cu.blocks import MlpTokenBlock
from vla_cu.cuda_coop import HIDDEN, INTERMEDIATE, MAX_TOKENS, coop_layer_into
from vla_cu.quant import GROUP_SIZE, pack_int4, quantize_int8

HOST_COMPILER = (
    shutil.which("cl")
    if os.name == "nt"
    else shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
)
pytestmark = pytest.mark.skipif(
    shutil.which("nvcc") is None or HOST_COMPILER is None,
    reason="no CUDA build toolchain on PATH; on Windows, run inside vcvars64.bat",
)

TOKENS = (1, 2, 3, 8, 50)


def reference(
    x: torch.Tensor, other: torch.Tensor, weight: torch.Tensor, up: torch.Tensor, down: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the five launch answer: added, normed and the layer tail."""
    added = torch.empty_like(x)
    normed = torch.empty_like(x)
    tail = torch.empty_like(x)
    out = torch.empty_like(x)
    add_norm_into(x, other, weight, added, normed)
    MlpTokenBlock(up, down, down_bits=4, group_size=GROUP_SIZE, tokens=x.shape[0]).into(normed, tail)
    add_into(added, tail, out)
    return added, normed, out


@pytest.mark.parametrize("tokens", TOKENS)
def test_matches_the_five_launch_path(tokens: int) -> None:
    """One launch must produce the same residual, norm and output as the five."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(HIDDEN, device="cuda", generator=generator) + 1.0
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator) * (1.0 / HIDDEN**0.5)
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator) * (1.0 / INTERMEDIATE**0.5)
    x = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
    other = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
    added, normed, expected = reference(x, other, weight, up, down)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP_SIZE)
    fused_added = torch.empty_like(x)
    fused_normed = torch.empty_like(x)
    fused_out = torch.empty_like(x)
    act = torch.empty((tokens, INTERMEDIATE), device="cuda")

    coop_layer_into(
        x, other, weight, up_codes, up_scales, down_packed, down_scales,
        fused_added, fused_normed, act, fused_out,
    )

    torch.testing.assert_close(fused_added, added, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(fused_normed, normed, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_out, expected, rtol=2e-2, atol=2e-2)


def test_rejects_more_tokens_than_the_kernel_holds() -> None:
    """The token loops are bounded by the compile-time maximum, so the host must refuse more."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    tokens = MAX_TOKENS + 1
    x = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
    other = torch.randn((tokens, HIDDEN), device="cuda", generator=generator)
    weights = torch.randn((HIDDEN, HIDDEN), device="cuda", generator=generator)
    up = torch.randn((INTERMEDIATE, HIDDEN), device="cuda", generator=generator)
    down = torch.randn((HIDDEN, INTERMEDIATE), device="cuda", generator=generator)
    up_codes, up_scales = quantize_int8(up)
    down_packed, down_scales = pack_int4(down, group_size=GROUP_SIZE)

    with pytest.raises(ValueError):
        coop_layer_into(
            x, other, weights, up_codes, up_scales, down_packed, down_scales,
            torch.empty_like(x), torch.empty_like(x), torch.empty((tokens, INTERMEDIATE), device="cuda"),
            torch.empty_like(x),
        )

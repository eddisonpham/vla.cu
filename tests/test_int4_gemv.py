"""The fused int4 GEMV against dequantize-then-matmul in float64."""

from __future__ import annotations

import pytest
import torch

from kernels.triton.int4_gemv import int4_gemv
from vla_cu.quant import GROUP_SIZE, dequantize_int4, pack_int4

SHAPES = ((1, 64), (3, 128), (17, 320), (64, 960), (128, 2048))


def reference(packed: torch.Tensor, scales: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Dequantize to float64, then matmul, so the reference carries no fp32 error."""
    weights = dequantize_int4(packed, scales).to(torch.float64)
    return (weights @ x.to(torch.float64)).to(torch.float32)


@pytest.mark.parametrize(("n_out", "n_in"), SHAPES)
def test_matches_dequantize_then_matmul(n_out: int, n_in: int) -> None:
    """Nibble order, group scale placement, and masking all have to agree with the reference."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn((n_out, n_in), device="cuda", generator=generator) * 0.05
    x = torch.randn(n_in, device="cuda", generator=generator)

    packed, scales = pack_int4(w)
    out = int4_gemv(packed, scales, x)

    assert out.shape == (n_out,)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, reference(packed, scales, x), rtol=1e-4, atol=1e-3)


def test_block_k_matches_reference() -> None:
    """A smaller block only changes the loop chunking, not the arithmetic."""
    generator = torch.Generator(device="cuda").manual_seed(3)
    w = torch.randn((11, 512), device="cuda", generator=generator) * 0.05
    x = torch.randn(512, device="cuda", generator=generator)

    packed, scales = pack_int4(w)
    out = int4_gemv(packed, scales, x, block_k=256)

    torch.testing.assert_close(out, reference(packed, scales, x), rtol=1e-4, atol=1e-3)


def test_rejects_scale_table_of_wrong_length() -> None:
    """A scale table sized for a different group size must fail before the launch."""
    w = torch.randn((4, 128), device="cuda") * 0.05
    packed, scales = pack_int4(w)
    x = torch.randn(128, device="cuda")

    with pytest.raises(ValueError):
        int4_gemv(packed, scales[:, :1], x)


def test_fp16_scales_match_fp32_scales() -> None:
    """Narrowing only the stored scale table must stay inside the quantization error."""
    generator = torch.Generator(device="cuda").manual_seed(4)
    w = torch.randn((64, 960), device="cuda", generator=generator) * 0.05
    x = torch.randn(960, device="cuda", generator=generator)

    packed, scales32 = pack_int4(w)
    _, scales16 = pack_int4(w, scale_dtype=torch.float16)
    out16 = int4_gemv(packed, scales16, x)

    torch.testing.assert_close(out16, reference(packed, scales16, x), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(out16, int4_gemv(packed, scales32, x), rtol=1e-3, atol=1e-3)


def test_cache_hints_do_not_change_results() -> None:
    """An eviction hint changes where a line lands, never what a load returns."""
    generator = torch.Generator(device="cuda").manual_seed(5)
    w = torch.randn((32, 512), device="cuda", generator=generator) * 0.05
    x = torch.randn(512, device="cuda", generator=generator)

    packed, scales = pack_int4(w)
    assert torch.equal(int4_gemv(packed, scales, x, hints=True), int4_gemv(packed, scales, x, hints=False))


def test_rejects_float_weights() -> None:
    """Handing fp32 weights to the int4 path would reinterpret bytes, not complain later."""
    packed = torch.zeros((4, 64), device="cuda", dtype=torch.float32)
    scales = torch.ones((4, 2), device="cuda")
    x = torch.zeros(128, device="cuda")

    with pytest.raises(ValueError):
        int4_gemv(packed, scales, x)

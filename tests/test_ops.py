"""The custom op wrappers behave like real torch ops, not like hidden launches."""

from __future__ import annotations

import pytest
import torch

from vla_cu.ops import int4_gemv_op, int8_gemv_op
from vla_cu.quant import pack_int4


def test_int8_op_matches_the_kernel() -> None:
    """The op is a wrapper, so its values must be bit identical to a direct launch."""
    generator = torch.Generator(device="cuda").manual_seed(5)
    w = torch.randint(-127, 128, (23, 320), device="cuda", dtype=torch.int8, generator=generator)
    scales = torch.rand(23, device="cuda", generator=generator) * 0.01
    x = torch.randn(320, device="cuda", generator=generator)

    from kernels.triton.int8_gemv import int8_gemv

    torch.testing.assert_close(int8_gemv_op(w, scales, x), int8_gemv(w, scales, x))


def test_int4_op_matches_the_kernel() -> None:
    """Same check for the fused 4-bit path, whose output shape comes from the fake."""
    generator = torch.Generator(device="cuda").manual_seed(6)
    w = torch.randn((23, 320), device="cuda", generator=generator) * 0.05
    x = torch.randn(320, device="cuda", generator=generator)
    packed, scales = pack_int4(w)

    from kernels.triton.int4_gemv import int4_gemv

    out = int4_gemv_op(packed, scales, x)
    assert out.shape == (23,)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, int4_gemv(packed, scales, x))


def test_opcheck_passes_for_both_ops() -> None:
    """opcheck runs the schema, fake tensor and autograd registration checks."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    w8 = torch.randint(-127, 128, (9, 128), device="cuda", dtype=torch.int8, generator=generator)
    s8 = torch.rand(9, device="cuda", generator=generator) * 0.01
    w = torch.randn((9, 128), device="cuda", generator=generator) * 0.05
    x = torch.randn(128, device="cuda", generator=generator)
    packed, scales = pack_int4(w)

    torch.library.opcheck(int8_gemv_op, (w8, s8, x))
    torch.library.opcheck(int4_gemv_op, (packed, scales, x))


def test_fake_implementation_needs_no_gpu_memory() -> None:
    """Tracing must produce a correctly shaped meta tensor without a launch."""
    w = torch.empty((64, 256), device="cuda", dtype=torch.uint8)
    scales = torch.empty((64, 4), device="cuda")

    with torch._subclasses.FakeTensorMode() as mode:
        meta_w = mode.from_tensor(w)
        meta_scales = mode.from_tensor(scales)
        meta_x = mode.from_tensor(torch.empty(512, device="cuda"))
        out = int4_gemv_op(meta_w, meta_scales, meta_x)

    assert out.shape == (64,)
    assert out.device == meta_w.device

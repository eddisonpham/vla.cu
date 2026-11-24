"""The public Int4Linear matches the packed reference at one token and at a chunk."""

from __future__ import annotations

import pytest
import torch

from vla_cu import Int4Linear, dequantize_int4, pack_int4


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return the L2 error over the L2 reference, the shape-free accuracy measure."""
    return (torch.linalg.vector_norm(approx - exact) / torch.linalg.vector_norm(exact)).item()


def test_one_token_matches_the_dequantized_reference() -> None:
    """One token takes the GEMV, and both paths read the same codes."""
    generator = torch.Generator(device="cuda").manual_seed(11)
    weight = torch.randn((384, 960), device="cuda", generator=generator) * (1.0 / 960**0.5)
    x = torch.randn(960, device="cuda", generator=generator)
    layer = Int4Linear(weight)
    packed, scales = pack_int4(weight)
    exact = torch.mv(dequantize_int4(packed, scales), x)

    out = layer(x)
    assert out.shape == (384,)
    assert out.dtype == torch.float32
    assert relative_error(out, exact) < 1e-5


def test_a_chunk_takes_the_gemm_and_agrees_rowwise() -> None:
    """At two or more tokens each row is the same GEMV, so the two paths must agree."""
    generator = torch.Generator(device="cuda").manual_seed(12)
    weight = torch.randn((192, 640), device="cuda", generator=generator) * (1.0 / 640**0.5)
    layer = Int4Linear(weight)
    packed, scales = pack_int4(weight)
    values = dequantize_int4(packed, scales)

    x = torch.randn((3, 640), device="cuda", generator=generator)
    torch.testing.assert_close(layer(x), x @ values.T, rtol=2e-3, atol=2e-3)

    chunk = torch.randn((21, 640), device="cuda", generator=generator)
    assert relative_error(layer(chunk), chunk @ values.T) < 1e-3


def test_from_linear_reuses_the_weight() -> None:
    """Wrapping an existing Linear is the migration path a caller actually has."""
    generator = torch.Generator(device="cuda").manual_seed(13)
    native = torch.nn.Linear(256, 64, bias=True).cuda()
    layer = Int4Linear.from_linear(native)
    x = torch.randn(256, device="cuda", generator=generator)

    assert relative_error(layer(x), Int4Linear(native.weight)(x)) == 0.0


def test_a_width_that_does_not_divide_by_the_group_is_refused() -> None:
    """A group may not straddle the row end, so the constructor has to say no."""
    with pytest.raises(ValueError, match="not divisible"):
        Int4Linear(torch.empty((8, 100), device="cuda"))


def test_a_wrong_activation_shape_is_refused() -> None:
    """A kernel handed the wrong width reads plausible garbage, so the module checks."""
    layer = Int4Linear(torch.empty((8, 128), device="cuda"))
    with pytest.raises(ValueError, match="in_features"):
        layer(torch.empty(96, device="cuda"))


def test_fp64_activations_are_refused() -> None:
    """Every kernel here is fp32, and a silent cast would hide a caller's mistake."""
    layer = Int4Linear(torch.empty((8, 128), device="cuda"))
    with pytest.raises(ValueError, match="fp32"):
        layer(torch.empty(128, device="cuda", dtype=torch.float64))


def test_a_graph_replays_and_picks_up_a_new_input() -> None:
    """The captured call must read the copied input, not the sample it was captured with."""
    generator = torch.Generator(device="cuda").manual_seed(14)
    weight = torch.randn((192, 640), device="cuda", generator=generator) * (1.0 / 640**0.5)
    layer = Int4Linear(weight)
    first = torch.randn(640, device="cuda", generator=generator)
    second = torch.randn(640, device="cuda", generator=generator)

    replay = layer.graphed()

    torch.testing.assert_close(replay(first), layer(first), rtol=0, atol=0)
    torch.testing.assert_close(replay(second), layer(second), rtol=0, atol=0)


def test_a_graph_refuses_a_different_shape() -> None:
    """`copy_` would broadcast a short input, so the wrapper checks the capture shape."""
    layer = Int4Linear(torch.empty((8, 128), device="cuda"))
    replay = layer.graphed()

    with pytest.raises(ValueError, match="graph captured"):
        replay(torch.empty(96, device="cuda"))


def test_a_bound_graph_reads_the_caller_buffer_in_place() -> None:
    """Binding the capture to an input buffer removes the copy, so an in place rewrite shows."""
    generator = torch.Generator(device="cuda").manual_seed(15)
    weight = torch.randn((192, 640), device="cuda", generator=generator) * (1.0 / 640**0.5)
    layer = Int4Linear(weight)
    x = torch.randn(640, device="cuda", generator=generator)

    bound = layer.graphed(x)

    assert bound.static_in.data_ptr() == x.data_ptr()
    torch.testing.assert_close(bound(x), layer(x), rtol=0, atol=0)
    x.copy_(torch.randn(640, device="cuda", generator=generator))
    torch.testing.assert_close(bound.replay(), layer(x), rtol=0, atol=0)

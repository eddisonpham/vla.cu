"""Loader and Python wrappers for the cooperative tail kernels.

Concept: a grid barrier is only legal when every block of the grid is resident at the
same time, and `cudaLaunchCooperativeKernel` is what makes the driver promise that.
Inside the kernel `cooperative_groups::this_grid().sync()` separates the stages, so the
residual add, the norm, the projections and the second residual add run as one launch
instead of five or six. The plain tail runs the int8 up projection, then SiLU, then the
int4 down projection. The gated tail stages the norm once and computes silu(gate x) and
up x in the same pass, because a SmolVLA MLP carries both matrices.

Widths come from the tensors, not from a constant, so any layer whose hidden width is a
multiple of 16 and whose intermediate width is a multiple of 64 can use either kernel.
Shared memory is sized from those widths on the host.

Task: python -m vla_cu coop, python -m vla_cu coopgated
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

from vla_cu.validate import require

CSRC = Path(__file__).resolve().parent.parent / "csrc" / "coop_layer.cu"
HIDDEN = 720
INTERMEDIATE = 1920
GROUP = 64
MAX_TOKENS = 64
_EXTENSION = None


def extension():
    """Compile the extension on first use and return the cached module.

    A build needs nvcc and a host compiler, and on Windows the host compiler is only on
    PATH inside a shell where vcvars64.bat has been called, so a missing toolchain is
    reported as a RuntimeError with that hint.
    """
    global _EXTENSION
    if _EXTENSION is None:
        try:
            _EXTENSION = load(
                name="vla_cu_coop_layer",
                sources=[str(CSRC)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
            )
        except (RuntimeError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"could not build {CSRC.name}: {error}. On Windows, call vcvars64.bat in this shell first (see agents/HARDWARE.md)."
            ) from error
    return _EXTENSION


def _geometry(x: Tensor, other: Tensor, weight: Tensor, added: Tensor, normed: Tensor, out: Tensor) -> tuple[int, int]:
    """Check the fp32 buffers both tails share and return the token and hidden counts."""
    tokens = x.shape[0]
    require(1 <= tokens <= MAX_TOKENS, f"the cooperative kernel holds 1 to {MAX_TOKENS} tokens, got {tokens}")
    require(x.dim() == 2 and weight.dim() == 1, f"expected x [tokens, hidden] and weight [hidden], got {tuple(x.shape)} and {tuple(weight.shape)}")
    hidden = weight.shape[0]
    for name, tensor in (("x", x), ("other", other), ("added", added), ("normed", normed), ("out", out)):
        require(tuple(tensor.shape) == (tokens, hidden), f"expected {name} {(tokens, hidden)}, got {tuple(tensor.shape)}")
    require(hidden % 16 == 0, f"hidden must divide by 16 for the tensor core K step, got {hidden}")
    return tokens, hidden


def _int8_matrix(codes: Tensor, scales: Tensor, rows: int, cols: int, name: str) -> None:
    """Check one int8 projection and its row scales."""
    require(codes.dtype == torch.int8, f"expected int8 codes for {name}, got {codes.dtype}")
    require(tuple(codes.shape) == (rows, cols), f"expected {name} [{(rows, cols)}], got {tuple(codes.shape)}")
    require(tuple(scales.shape) == (rows,), f"expected {name} scales [{rows}], got {tuple(scales.shape)}")


def _int4_down(down: Tensor, down_scales: Tensor, hidden: int, intermediate: int) -> None:
    """Check the packed int4 down projection and its group scales."""
    require(down.dtype == torch.uint8, f"expected uint8 packed down weights, got {down.dtype}")
    require(tuple(down.shape) == (hidden, intermediate // 2), f"expected down [{(hidden, intermediate // 2)}], got {tuple(down.shape)}")
    require(tuple(down_scales.shape) == (hidden, intermediate // GROUP), f"expected down scales [{(hidden, intermediate // GROUP)}], got {tuple(down_scales.shape)}")


def coop_layer_into(
    x: Tensor,
    other: Tensor,
    weight: Tensor,
    up: Tensor,
    up_scales: Tensor,
    down: Tensor,
    down_scales: Tensor,
    added: Tensor,
    normed: Tensor,
    act: Tensor,
    out: Tensor,
    blocks: int = 0,
) -> None:
    """Write the two-matrix layer tail for every token with one cooperative launch.

    Computes added = x + other, normed = rmsnorm(added), act = silu(up @ normed) and
    out = added + down @ act. up is int8 [intermediate, hidden] with fp32 row scales,
    down is uint8 [hidden, intermediate // 2] with fp32 group scales, act is fp32
    [tokens, intermediate]. `blocks` defaults to the co-resident capacity, at most one
    block per SM.
    """
    tokens, hidden = _geometry(x, other, weight, added, normed, out)
    require(up.dim() == 2 and up.shape[1] == hidden, f"expected up [intermediate, {hidden}], got {tuple(up.shape)}")
    intermediate = up.shape[0]
    require(intermediate % GROUP == 0, f"intermediate must divide by {GROUP} for the int4 groups, got {intermediate}")
    _int8_matrix(up, up_scales, intermediate, hidden, "up")
    _int4_down(down, down_scales, hidden, intermediate)
    require(tuple(act.shape) == (tokens, intermediate), f"expected act [(tokens, intermediate)], got {tuple(act.shape)}")
    extension().coop_layer_launch(
        x, other, weight, up, up_scales, down, down_scales, added, normed, act, out, blocks
    )


def coop_gated_into(
    x: Tensor,
    other: Tensor,
    weight: Tensor,
    gate: Tensor,
    gate_scales: Tensor,
    up: Tensor,
    up_scales: Tensor,
    down: Tensor,
    down_scales: Tensor,
    added: Tensor,
    normed: Tensor,
    act: Tensor,
    out: Tensor,
    blocks: int = 0,
) -> None:
    """Write the gated SwiGLU layer tail for every token with one cooperative launch.

    Computes added = x + other, normed = rmsnorm(added), act = silu(gate @ normed) times
    (up @ normed) and out = added + down @ act. gate and up are int8 [intermediate, hidden]
    with fp32 row scales, and the remaining buffers are the same as `coop_layer_into`.
    `blocks` defaults to the co-resident capacity, at most one block per SM.
    """
    tokens, hidden = _geometry(x, other, weight, added, normed, out)
    require(gate.dim() == 2 and gate.shape[1] == hidden, f"expected gate [intermediate, {hidden}], got {tuple(gate.shape)}")
    require(tuple(up.shape) == tuple(gate.shape), f"expected up {tuple(gate.shape)}, got {tuple(up.shape)}")
    intermediate = gate.shape[0]
    require(intermediate % GROUP == 0, f"intermediate must divide by {GROUP} for the int4 groups, got {intermediate}")
    _int8_matrix(gate, gate_scales, intermediate, hidden, "gate")
    _int8_matrix(up, up_scales, intermediate, hidden, "up")
    _int4_down(down, down_scales, hidden, intermediate)
    require(tuple(act.shape) == (tokens, intermediate), f"expected act [{(tokens, intermediate)}], got {tuple(act.shape)}")
    extension().coop_gated_launch(
        x, other, weight, gate, gate_scales, up, up_scales, down, down_scales, added, normed, act, out, blocks
    )

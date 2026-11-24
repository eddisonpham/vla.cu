"""Loader and Python wrapper for the cooperative whole-tail kernel.

Concept: a grid barrier is only legal when every block of the grid is resident at the
same time, and `cudaLaunchCooperativeKernel` is what makes the driver promise that.
Inside the kernel `cooperative_groups::this_grid().sync()` then separates the stages,
so the residual add, the norm, the int8 up projection, SiLU, the int4 down projection
and the second residual add run as one launch instead of five. Both projections are
16 x 16 WMMA tiles: the block stages one 16 token activation tile in shared memory,
each warp owns a 16 column output tile, and the K loop dequantizes the weight tile into
shared memory and issues one tensor core dot per step.

Widths come from the tensors, not from a constant, so any layer whose hidden width is a
multiple of 16 and whose intermediate width is a multiple of 64 can use it. Shared memory
is sized from those widths on the host.

Task: python -m vla_cu coop
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
    """Write the layer tail for every token with one cooperative launch.

    Computes added = x + other, normed = rmsnorm(added), act = silu(up @ normed) and
    out = added + down @ act. Shapes come from the tensors: x, other, added, normed and
    out are fp32 [tokens, hidden]; weight fp32 [hidden]; up int8 [intermediate, hidden]
    with fp32 row scales [intermediate]; down uint8 [hidden, intermediate // 2] with fp32
    group scales [hidden, intermediate // 64]; act fp32 [tokens, intermediate]. The hidden
    width must divide by 16 and the intermediate width by 64. `blocks` defaults to the
    co-resident block capacity, at most one per SM.
    """
    tokens = x.shape[0]
    require(1 <= tokens <= MAX_TOKENS, f"the cooperative kernel holds 1 to {MAX_TOKENS} tokens, got {tokens}")
    require(x.dim() == 2 and weight.dim() == 1, f"expected x [tokens, hidden] and weight [hidden], got {tuple(x.shape)} and {tuple(weight.shape)}")
    hidden = weight.shape[0]
    require(x.shape == (tokens, hidden), f"expected x [{tokens}, {hidden}], got {tuple(x.shape)}")
    require(hidden % 16 == 0, f"hidden must divide by 16 for the tensor core K step, got {hidden}")
    require(up.dim() == 2 and up.shape[1] == hidden, f"expected up [intermediate, {hidden}], got {tuple(up.shape)}")
    intermediate = up.shape[0]
    require(intermediate % GROUP == 0, f"intermediate must divide by {GROUP} for the int4 groups, got {intermediate}")
    require(up.dtype == torch.int8, f"expected int8 codes for up, got {up.dtype}")
    require(up_scales.shape == (intermediate,), f"expected up scales [{intermediate}], got {tuple(up_scales.shape)}")
    require(tuple(down.shape) == (hidden, intermediate // 2), f"expected down [{hidden}, {intermediate // 2}], got {tuple(down.shape)}")
    require(down.dtype == torch.uint8, f"expected uint8 packed down weights, got {down.dtype}")
    require(tuple(down_scales.shape) == (hidden, intermediate // GROUP), f"expected down scales [{(hidden, intermediate // GROUP)}], got {tuple(down_scales.shape)}")
    require(tuple(act.shape) == (tokens, intermediate), f"expected act [{tokens}, {intermediate}], got {tuple(act.shape)}")
    extension().coop_layer_launch(
        x, other, weight, up, up_scales, down, down_scales, added, normed, act, out, blocks
    )

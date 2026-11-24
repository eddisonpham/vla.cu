"""Loader and Python wrapper for the cooperative whole-tail kernel.

Concept: a grid barrier is only legal when every block of the grid is resident at the
same time, and `cudaLaunchCooperativeKernel` is what makes the driver promise that.
Inside the kernel `cooperative_groups::this_grid().sync()` then separates the stages,
so the residual add, the norm, the int8 up projection, SiLU, the int4 down projection
and the second residual add run as one launch instead of five.

The kernel is written for one expert layer: hidden 720, intermediate 1920, group 64 for
the int4 weights, and at most 64 tokens. It launches one block per SM, so the barrier
cannot wait on a block the scheduler has not started.

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
    out = added + down @ act. Shapes: x, other, added, normed and out are fp32
    [tokens, 720]; weight fp32 [720]; up int8 [1920, 720] with fp32 rows scales [1920];
    down uint8 [720, 960] with fp32 group scales [720, 30]; act fp32 [tokens, 1920].
    `blocks` defaults to one per SM, which is the most a cooperative launch may ask for
    unless occupancy says otherwise.
    """
    tokens = x.shape[0]
    require(1 <= tokens <= MAX_TOKENS, f"the cooperative kernel holds 1 to {MAX_TOKENS} tokens, got {tokens}")
    require(tuple(x.shape) == (tokens, HIDDEN), f"expected x [{tokens}, {HIDDEN}], got {tuple(x.shape)}")
    require(weight.shape == (HIDDEN,), f"expected weight [{HIDDEN}], got {tuple(weight.shape)}")
    require(tuple(up.shape) == (INTERMEDIATE, HIDDEN), f"expected up [{INTERMEDIATE}, {HIDDEN}], got {tuple(up.shape)}")
    require(up.dtype == torch.int8, f"expected int8 codes for up, got {up.dtype}")
    require(up_scales.shape == (INTERMEDIATE,), f"expected up scales [{INTERMEDIATE}], got {tuple(up_scales.shape)}")
    require(tuple(down.shape) == (HIDDEN, INTERMEDIATE // 2), f"expected down [{HIDDEN}, {INTERMEDIATE // 2}], got {tuple(down.shape)}")
    require(down.dtype == torch.uint8, f"expected uint8 packed down weights, got {down.dtype}")
    require(tuple(down_scales.shape) == (HIDDEN, INTERMEDIATE // GROUP), f"expected down scales [{(HIDDEN, INTERMEDIATE // GROUP)}], got {tuple(down_scales.shape)}")
    require(tuple(act.shape) == (tokens, INTERMEDIATE), f"expected act [{tokens}, {INTERMEDIATE}], got {tuple(act.shape)}")
    extension().coop_layer_launch(
        x, other, weight, up, up_scales, down, down_scales, added, normed, act, out, blocks
    )

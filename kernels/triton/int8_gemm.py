"""Fused int8 GEMM over a chunk of tokens, one launch per projection.

Concept: int8 codes need no per-weight scale, only one per output row, so the tensor
core consumes them directly and the row scale applies once to the finished tile. A chunk
therefore reads each weight once instead of once per token, which is the byte argument
the int4 GEMM makes (EXP-063), in the format the attention block already stores.

Task: python -m vla_cu int8tiles
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.validate import check_int8_gemm, check_power_of_two, require

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 128
MAX_BLOCK_M = 128
SHARED_LIMIT_BYTES = 101376
LONG_TOKENS = 33
LONG_TILE: tuple[int, int, int, int] = (16, 128, 128, 2)
Tile = tuple[int, int, int, int]
TILES: tuple[Tile, ...] = (
    (16, 64, 128, 3),
    (32, 64, 128, 3),
    (64, 64, 128, 2),
    (128, 64, 64, 2),
)


@triton.jit
def int8_gemm_kernel(
    w_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Dot a block of rows with int8 weight tiles and scale each output column once.

    w: int8 [N, K], scales: fp32 [N], x: fp32 [M, K], out: fp32 [M, N].
    Mapping: program (m, n) owns tokens [m*BLOCK_M, +BLOCK_M) and outputs
    [n*BLOCK_N, +BLOCK_N) and walks K in BLOCK_K steps. Each step rounds both operands to
    fp16, which holds every int8 code exactly, and issues one dot. The row scale multiplies
    the finished accumulator in fp32, outside the loop.
    Expected bottleneck: memory at chunk sizes, since the weights are read once per chunk.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        codes = tl.load(
            w_ptr + n_offsets[:, None] * K + k_offsets[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0,
        )
        activations = tl.load(
            x_ptr + m_offsets[:, None] * K + k_offsets[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(activations.to(tl.float16), tl.trans(codes.to(tl.float16)))

    scale = tl.load(scale_ptr + n_offsets, mask=n_mask, other=0.0)
    tl.store(
        out_ptr + m_offsets[:, None] * N + n_offsets[None, :],
        acc * scale[None, :],
        mask=m_mask[:, None] & n_mask[None, :],
    )


def shared_memory_bytes(tile: Tile) -> int:
    """Return the fp16 operand bytes one tile stages through its pipeline."""
    block_m, block_n, block_k, stages = tile
    return 2 * (block_m * block_k + block_k * block_n) * stages


def tile_for(tokens: int) -> Tile:
    """Return the preset for `tokens`, using a narrow row tile once the chunk is long.

    A 50 token chunk on a 64 row tile is one row block, so 720 outputs means six programs
    on 26 SMs, and a 16 row tile makes four. That tile won every repeat of three paired
    runs at 720 x 1920 and 50 tokens, by 1.07x to 1.53x (EXP-080). The direction repeated
    and the size did not, so quote the range and not the best reading.
    """
    if tokens >= LONG_TOKENS:
        return LONG_TILE
    for tile in TILES:
        if tokens <= tile[0] and shared_memory_bytes(tile) <= SHARED_LIMIT_BYTES:
            return tile
    return TILES[-1]


def int8_gemm_into(
    w: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    tile: Tile | None = None,
) -> None:
    """Write scales * (w @ x.T) into `out` with one launch per output tile.

    Assumptions: w int8 [N, K] contiguous, scales fp32 [N] contiguous, x fp32 [M, K]
    contiguous, out fp32 [M, N] contiguous, all on one CUDA device. K and N are masked,
    so neither needs to divide the tile. M is padded to the tile's row count.
    """
    n_tokens, n_rows, n_cols = check_int8_gemm(w, scales, x, out)
    block_m, block_n, block_k, stages = tile_for(n_tokens) if tile is None else tile
    check_power_of_two("block size", block_k)
    check_power_of_two("output tile", block_n)
    require(block_n >= 16, f"expected an output tile of at least 16, got {block_n}")
    staged = shared_memory_bytes((block_m, block_n, block_k, stages))
    require(staged <= SHARED_LIMIT_BYTES, f"tile stages {staged} bytes against a {SHARED_LIMIT_BYTES} byte limit")
    if n_tokens == 0 or n_rows == 0:
        return

    grid = (triton.cdiv(n_tokens, block_m), triton.cdiv(n_rows, block_n))
    int8_gemm_kernel[grid](
        w,
        scales,
        x,
        out,
        n_tokens,
        n_rows,
        n_cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=stages,
    )


def int8_gemm(w: torch.Tensor, scales: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Allocate the output and return the int8 GEMM of `x`."""
    out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=torch.float32)
    int8_gemm_into(w, scales, x, out)
    return out

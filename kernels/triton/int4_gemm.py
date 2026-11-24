"""Fused 4-bit dequantize and GEMM for the token counts a policy actually has.

Concept: a GEMV reads every weight once per call, so M calls read the weights M times.
A GEMM reads them once and reuses each byte M times, and `tl.dot` turns the inner
product into a tensor-core operation. The dequantized weight tile is therefore built in
registers and handed straight to the matrix unit, which is the same trick the GEMV
cannot use because a tensor core needs a tile, not a vector.

Nibble order is element 2i low, element 2i+1 high, so a step splits into two dots: the
even elements of the activation vector times the low nibbles, plus the odd elements
times the high ones. That avoids an interleave and keeps both operands contiguous along
the reduction.

The group scale is expanded to one value per weight before the dot, because the tensor
core sums over k and would otherwise mix groups. Dequantization happens in fp32 and is
rounded to fp16 once, at the dot, so the weight carries one rounding instead of two.

The tile is chosen from a menu, not from taste: shared memory decides which tile and
stage count fit, and `m50` sweeps the menu at the expert shapes. Long K steps amortize
the loop, and deeper stages hide the load latency that a 12 to 30 step loop cannot.

A K split combines inside the same launch: every program writes its partial to a
workspace, then the last one to arrive sums the partials in split order and resets the
arrival counter. The separate reduce launch this replaces was measured as the whole cost
of the split (EXP-068, EXP-071).

Task: python -m vla_cu m50
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.quant import GROUP_SIZE
from vla_cu.validate import check_int4_gemm, check_power_of_two, require

BLOCK_N = 64
BLOCK_K = 128
NUM_STAGES = 2
MAX_BLOCK_M = 128
SHARED_LIMIT_BYTES = 101376
WIDE_ROWS = 1024
Tile = tuple[int, int, int, int]
TILES: tuple[Tile, ...] = (
    (16, 128, 128, 2),
    (32, 128, 128, 2),
    (64, 128, 128, 2),
    (16, 64, 128, 3),
    (32, 64, 128, 3),
)


@triton.jit
def int4_gemm_kernel(
    packed_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    workspace_ptr,
    counter_ptr,
    M,
    N,
    K,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLITS: tl.constexpr,
) -> None:
    """Dequantize int4 weight tiles in registers and dot them with a block of rows.

    packed: uint8 [N, K // 2], scales: fp32 or fp16 [N, K // GROUP], x: fp32 [M, K],
    out: fp32 [M, N], workspace: fp32 [SPLITS, M, N], counters: int32 [tiles], the last two
    used only when SPLITS is above 1.
    Mapping: program (m, n, split) owns tokens [m*BLOCK_M, +BLOCK_M), outputs
    [n*BLOCK_N, +BLOCK_N) and walks every SPLITS'th K step, starting at its own. Each step
    loads a [BLOCK_N, BLOCK_K // 2] byte tile sitting on whole groups, splits it into low
    and high nibbles, applies the group scale, and issues two dots against the even and odd
    halves of the activation tile.
    A split launch then writes its partial, raises its tile's arrival counter with release
    semantics, and the last arrival loads every partial with the L1 bypassed, sums them in
    split order and resets the counter, which makes the sum deterministic and keeps the
    kernel to one launch. Expected bottleneck: compute once M grows, memory at M = 1.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    split = tl.program_id(axis=2)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    bytes_per_step: tl.constexpr = BLOCK_K // 2
    byte_lanes = tl.arange(0, bytes_per_step)
    n_groups = K // GROUP
    n_bytes_total = K // 2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(split * BLOCK_K, K, BLOCK_K * SPLITS):
        byte_base = k // 2
        byte_index = byte_base + byte_lanes
        weight_mask = n_mask[:, None] & (byte_index < n_bytes_total)[None, :]
        packed_offsets = n_offsets[:, None] * n_bytes_total + byte_index[None, :]
        packed = tl.load(packed_ptr + packed_offsets, mask=weight_mask, other=0).to(tl.int32)

        group_index = byte_index // (GROUP // 2)
        scale = tl.load(
            scale_ptr + n_offsets[:, None] * n_groups + group_index[None, :],
            mask=weight_mask & (group_index < n_groups)[None, :],
            other=0.0,
        ).to(tl.float32)

        low = ((packed & 0x0F).to(tl.float32) - 8.0) * scale
        high = (((packed >> 4) & 0x0F).to(tl.float32) - 8.0) * scale

        k_element = k + 2 * byte_lanes
        activation_offsets = m_offsets[:, None] * K + k_element[None, :]
        activation_mask = m_mask[:, None] & (k_element < K)[None, :]
        activations_even = tl.load(x_ptr + activation_offsets, mask=activation_mask, other=0.0)
        activations_odd = tl.load(
            x_ptr + activation_offsets + 1,
            mask=activation_mask & (k_element + 1 < K)[None, :],
            other=0.0,
        )

        acc += tl.dot(activations_even.to(tl.float16), tl.trans(low.to(tl.float16)))
        acc += tl.dot(activations_odd.to(tl.float16), tl.trans(high.to(tl.float16)))

    output = m_offsets[:, None] * N + n_offsets[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    if SPLITS == 1:
        tl.store(out_ptr + output, acc, mask=out_mask)
    else:
        tl.store(workspace_ptr + split * M * N + output, acc, mask=out_mask)
        tile = pid_m * tl.cdiv(N, BLOCK_N) + pid_n
        arrived = tl.atomic_add(counter_ptr + tile, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLITS - 1:
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for other in tl.static_range(SPLITS):
                total += tl.load(
                    workspace_ptr + other * M * N + output,
                    mask=out_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.store(out_ptr + output, total, mask=out_mask)
            tl.store(counter_ptr + tile, 0)


def shared_memory_bytes(tile: Tile) -> int:
    """Return the fp16 operand bytes one tile stages through its pipeline."""
    block_m, block_n, block_k, stages = tile
    return (2 * block_m * block_k + 2 * block_k * block_n) * stages


def tile_for(tokens: int, n_rows: int) -> Tile:
    """Return the measured preset for a shape, and the smallest row tile that covers `tokens`.

    The `m50` sweep splits the menu by output width: 1920 outputs want a 128 wide output
    tile at two stages, while 720 outputs want 64 wide at three stages, because at M = 50
    the grid has to keep 26 SMs busy. Row tile is the smallest preset covering the tokens,
    so a 50 token chunk pads to 64 rows and not to 128. The narrow family stops at 32 rows:
    a 64 row, 3 stage tile asks for 106496 bytes against this GPU's 101376 limit.
    """
    family = [tile for tile in TILES if (tile[1] == 128) == (n_rows >= WIDE_ROWS)]
    for tile in family:
        if tokens <= tile[0] and shared_memory_bytes(tile) <= SHARED_LIMIT_BYTES:
            return tile
    return family[-1]


def block_m_for(m_rows: int) -> int:
    """Return the smallest power-of-two token tile at least as tall as `m_rows`.

    A tensor core needs 16 rows, so the tile floor is 16; rows past `m_rows` are masked
    and cost compute but no memory.
    """
    if m_rows < 1:
        raise ValueError(f"expected at least one token, got {m_rows}")
    tile = 16
    while tile < m_rows:
        tile *= 2
    return min(tile, MAX_BLOCK_M)


def int4_gemm_into(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    group_size: int = GROUP_SIZE,
    tile: Tile | None = None,
    splits: int = 1,
    partial: torch.Tensor | None = None,
    counters: torch.Tensor | None = None,
) -> None:
    """Write (packed weights dequantized) @ x transpose into `out`, no fp32 weights.

    Assumptions: packed uint8 [N, K // 2] contiguous, scales fp32 or fp16
    [N, K // group_size] contiguous, x fp32 [M, K] contiguous, out fp32 [M, N]
    contiguous, all on one CUDA device. K must be divisible by group_size, the tile's K
    step must be a multiple of group_size, and its row count pads M. `tile` overrides the
    menu, which is what the `m50` sweep varies. `splits` divides the K loop across
    programs in one launch, and then `partial` must be a contiguous fp32 buffer
    [splits, M, N] and `counters` a zeroed int32 buffer of one counter per output tile.
    """
    n_tokens, n_rows, n_cols, _ = check_int4_gemm(packed, scales, x, out, group_size)
    block_m, block_n, block_k, stages = tile_for(n_tokens, n_rows) if tile is None else tile
    check_power_of_two("block size", block_k)
    require(block_k % group_size == 0, f"block size {block_k} must be a multiple of group size {group_size}")
    check_power_of_two("output tile", block_n)
    require(block_n >= 16, f"expected an output tile of at least 16, got {block_n}")
    staged = shared_memory_bytes((block_m, block_n, block_k, stages))
    require(staged <= SHARED_LIMIT_BYTES, f"tile stages {staged} bytes against a {SHARED_LIMIT_BYTES} byte limit")
    require(splits >= 1, f"expected at least one split, got {splits}")
    require(
        splits <= triton.cdiv(n_cols, block_k),
        f"{splits} splits leave a slice of {block_k} empty for K {n_cols}",
    )
    if splits > 1:
        shaped = None if partial is None else tuple(partial.shape)
        require(
            partial is not None
            and partial.dtype == torch.float32
            and partial.shape == (splits, n_tokens, n_rows)
            and partial.is_contiguous(),
            f"expected a contiguous fp32 partial buffer {(splits, n_tokens, n_rows)}, got {shaped}",
        )
        n_tiles = triton.cdiv(n_tokens, block_m) * triton.cdiv(n_rows, block_n)
        held = None if counters is None else tuple(counters.shape)
        require(
            counters is not None
            and counters.dtype == torch.int32
            and counters.shape == (n_tiles,)
            and counters.is_contiguous(),
            f"expected int32 counters for {n_tiles} tiles, got {counters.dtype if counters is not None else None} {held}",
        )
    if n_tokens == 0 or n_rows == 0:
        return

    grid = (triton.cdiv(n_tokens, block_m), triton.cdiv(n_rows, block_n), splits)
    int4_gemm_kernel[grid](
        packed,
        scales,
        x,
        out,
        partial if splits > 1 else out,
        counters if splits > 1 else out,
        n_tokens,
        n_rows,
        n_cols,
        GROUP=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLITS=splits,
        num_stages=stages,
    )


def int4_gemm(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
    tile: Tile | None = None,
) -> torch.Tensor:
    """Allocate the output and return the dequantized GEMM."""
    out = torch.empty((x.shape[0], packed.shape[0]), device=x.device, dtype=torch.float32)
    int4_gemm_into(packed, scales, x, out, group_size=group_size, tile=tile)
    return out

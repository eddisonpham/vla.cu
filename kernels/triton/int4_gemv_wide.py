"""int4 GEMV over int32 words: wide weight loads and a dot reduction.

Concept: reading the packed buffer as int32, so a lane owns four consecutive words, is
the cheapest way to ask Triton for 16 byte weight loads, and a group sum can be a dot
against a constant matrix instead of a reshape and two nested sums. At 4096 x 8192 the
loads still come out 4 and 8 bytes wide (EXP-052) and the dot reduction is slower than
the staged mapping (EXP-056), so this mapping is a measured alternative, not the default.

Task: python -m vla_cu plateau
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.quant import GROUP_SIZE
from vla_cu.validate import check_int4_gemv, check_power_of_two, require

BLOCK_K = 1024
WARPS = 1
BYTES_PER_WORD = 4
NIBBLES_PER_WORD = 8
DOT_LANES = 16
PRECISION = "ieee"


@triton.jit
def int4_gemv_wide_kernel(
    words_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    N,
    K,
    GROUP: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_LANES: tl.constexpr,
    PRECISION: tl.constexpr,
    W_EVICT: tl.constexpr,
    X_EVICT: tl.constexpr,
) -> None:
    """Dot one packed int4 row with one fp32 vector, reading the row as int32 words.

    words: int32 [N, K // 8], scales: fp32 or fp16 [N, K // GROUP], x: fp32 [K], out: fp32 [N].
    Mapping: program r owns output r and walks K in BLOCK_K element steps. The weights of
    a step are one flat run of BLOCK_K // 8 words whose nibbles are already in element
    order, so unpacking gives a [groups, GROUP] product tile with no data movement. The
    group sum is that tile dotted with a [GROUP, DOT_LANES] matrix of 1 / DOT_LANES, which
    is exact in binary and leaves every column holding the group sum, so the twenty extra
    columns are free in accuracy and not in arithmetic. `PRECISION` decides where the dot
    runs: "ieee" keeps fp32 and "tf32" trades mantissa bits for tensor cores.
    Expected bottleneck: memory at "ieee", because the dot does 16 times the multiplies a
    sum needs and that becomes the floor once the weights are streaming well.
    """
    row = tl.program_id(axis=0)
    n_groups = K // GROUP
    groups_per_block: tl.constexpr = BLOCK_K // GROUP
    block_words: tl.constexpr = BLOCK_K // 8
    flat_lanes = tl.arange(0, block_words)
    nibble_lanes = tl.arange(0, 8)
    group_lanes = tl.arange(0, groups_per_block)
    element_lanes = tl.arange(0, GROUP)
    shifts = nibble_lanes * 4
    ones = tl.full((GROUP, DOT_LANES), 1.0 / DOT_LANES, tl.float32)
    acc = tl.zeros((groups_per_block, DOT_LANES), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        first_group = k // GROUP
        group_mask = first_group + group_lanes < n_groups
        flat = tl.load(
            words_ptr + row * (K // 8) + k // 8 + flat_lanes,
            mask=k // 8 + flat_lanes < K // 8,
            other=0,
            eviction_policy=W_EVICT,
        )
        codes = ((flat[:, None] >> shifts[None, :]) & 0x0F).to(tl.float32) - 8.0
        codes = tl.reshape(codes, (groups_per_block, GROUP))
        activations = tl.load(
            x_ptr + k + group_lanes[:, None] * GROUP + element_lanes[None, :],
            mask=group_mask[:, None],
            other=0.0,
            eviction_policy=X_EVICT,
        )
        scale = tl.load(
            scale_ptr + row * n_groups + first_group + group_lanes,
            mask=group_mask,
            other=0.0,
            eviction_policy=W_EVICT,
        ).to(tl.float32)
        acc += tl.dot(codes * activations, ones, input_precision=PRECISION) * scale[:, None]

    tl.store(out_ptr + row, tl.sum(tl.sum(acc, axis=1), axis=0))


def int4_gemv_wide_into(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    out: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    warps: int = WARPS,
    hints: bool = True,
    precision: str = PRECISION,
) -> None:
    """Write (packed weights dequantized) @ x into `out`, reading the row as int32.

    Assumptions: as `int4_gemv_into`, plus K divisible by 32 so every row start is 16
    byte aligned, group_size a multiple of 8, and group_size at least DOT_LANES because
    the dot reduction has no shape below that.
    """
    n_rows, n_cols, _ = check_int4_gemv(packed, scales, x, out, group_size)
    check_power_of_two("block size", block_k)
    require(block_k % group_size == 0, f"block size {block_k} must be a multiple of group size {group_size}")
    require(group_size % NIBBLES_PER_WORD == 0, f"group size {group_size} must be a multiple of {NIBBLES_PER_WORD}")
    require(group_size >= DOT_LANES, f"group size {group_size} must be at least {DOT_LANES}")
    require(n_cols % (2 * BYTES_PER_WORD * NIBBLES_PER_WORD) == 0, f"K {n_cols} must be a multiple of 32 for 16 byte row starts")
    if n_rows == 0:
        return

    int4_gemv_wide_kernel[(n_rows,)](
        packed.view(torch.int32),
        scales,
        x,
        out,
        n_rows,
        n_cols,
        GROUP=group_size,
        BLOCK_K=block_k,
        DOT_LANES=DOT_LANES,
        PRECISION=precision,
        num_warps=warps,
        W_EVICT="evict_first" if hints else "",
        X_EVICT="evict_last" if hints else "",
    )


def int4_gemv_wide(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
    block_k: int = BLOCK_K,
    warps: int = WARPS,
    hints: bool = True,
    precision: str = PRECISION,
) -> torch.Tensor:
    """Allocate the output and return the wide-load dequantized GEMV."""
    out = torch.empty(packed.shape[0], device=packed.device, dtype=torch.float32)
    int4_gemv_wide_into(
        packed, scales, x, out, group_size=group_size, block_k=block_k, warps=warps, hints=hints, precision=precision
    )
    return out

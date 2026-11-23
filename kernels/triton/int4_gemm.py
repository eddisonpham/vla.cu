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

The step width is set by shared memory, not by taste: a 512-wide step needs two 64-wide
fp16 tiles per stage and 3 stages of that is 224 KB against this GPU's 101 KB limit. A
64-wide step with 2 stages fits, which is what the launch asks for.

Task: python -m vla_cu crossover
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vla_cu.quant import GROUP_SIZE
from vla_cu.validate import check_int4_gemm, check_power_of_two, require

BLOCK_N = 64
BLOCK_K = 64
NUM_STAGES = 2
MAX_BLOCK_M = 128


@triton.jit
def int4_gemm_kernel(
    packed_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    M,
    N,
    K,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Dequantize int4 weight tiles in registers and dot them with a block of rows.

    packed: uint8 [N, K // 2], scales: fp32 or fp16 [N, K // GROUP], x: fp32 [M, K],
    out: fp32 [M, N].
    Mapping: program (m, n) owns tokens [m*BLOCK_M, +BLOCK_M) and outputs [n*BLOCK_N, +BLOCK_N)
    and walks K in BLOCK_K element steps. Each step loads a [BLOCK_N, BLOCK_K // 2] byte
    tile sitting on whole groups, splits it into low and high nibbles, applies the group
    scale, and issues two dots against the even and odd halves of the activation tile.
    Expected bottleneck: compute once M grows, memory at M = 1. The crossing point is the
    answer this kernel exists to measure.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    bytes_per_step: tl.constexpr = BLOCK_K // 2
    byte_lanes = tl.arange(0, bytes_per_step)
    n_groups = K // GROUP
    n_bytes_total = K // 2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
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

    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets[None, :], acc, mask=m_mask[:, None] & n_mask[None, :])


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
    block_k: int = BLOCK_K,
    block_n: int = BLOCK_N,
) -> None:
    """Write (packed weights dequantized) @ x transpose into `out`, no fp32 weights.

    Assumptions: packed uint8 [N, K // 2] contiguous, scales fp32 or fp16
    [N, K // group_size] contiguous, x fp32 [M, K] contiguous, out fp32 [M, N]
    contiguous, all on one CUDA device. K must be divisible by group_size and block_k
    must be a multiple of group_size, which is a power of two. block_n is a power of two
    of at least 16. M is padded up to the next power of two.
    """
    n_tokens, n_rows, n_cols, _ = check_int4_gemm(packed, scales, x, out, group_size)
    check_power_of_two("block size", block_k)
    require(block_k % group_size == 0, f"block size {block_k} must be a multiple of group size {group_size}")
    check_power_of_two("output tile", block_n)
    require(block_n >= 16, f"expected an output tile of at least 16, got {block_n}")
    if n_tokens == 0 or n_rows == 0:
        return

    block_m = block_m_for(n_tokens)
    grid = (triton.cdiv(n_tokens, block_m), triton.cdiv(n_rows, block_n))
    int4_gemm_kernel[grid](
        packed,
        scales,
        x,
        out,
        n_tokens,
        n_rows,
        n_cols,
        GROUP=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=NUM_STAGES,
    )


def int4_gemm(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Allocate the output and return the dequantized GEMM."""
    out = torch.empty((x.shape[0], packed.shape[0]), device=x.device, dtype=torch.float32)
    int4_gemm_into(packed, scales, x, out, group_size=group_size)
    return out

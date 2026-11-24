#pragma once

#include <cuda_runtime.h>

namespace vla_cu_int4 {

constexpr int GROUP = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARPS_PER_BLOCK * 32;
constexpr int WEIGHTS_PER_LANE = 8;
constexpr int WEIGHTS_PER_STEP = 32 * WEIGHTS_PER_LANE;

template <bool STREAM>
__device__ __forceinline__ unsigned int load_word(const unsigned int* address) {
    return STREAM ? __ldcs(address) : *address;
}

__device__ __forceinline__ float dot_eight(unsigned int word, float4 lo, float4 hi) {
    float local = 0.0f;
    local = fmaf(static_cast<float>(word & 0x0F) - 8.0f, lo.x, local);
    local = fmaf(static_cast<float>((word >> 4) & 0x0F) - 8.0f, lo.y, local);
    local = fmaf(static_cast<float>((word >> 8) & 0x0F) - 8.0f, lo.z, local);
    local = fmaf(static_cast<float>((word >> 12) & 0x0F) - 8.0f, lo.w, local);
    local = fmaf(static_cast<float>((word >> 16) & 0x0F) - 8.0f, hi.x, local);
    local = fmaf(static_cast<float>((word >> 20) & 0x0F) - 8.0f, hi.y, local);
    local = fmaf(static_cast<float>((word >> 24) & 0x0F) - 8.0f, hi.z, local);
    local = fmaf(static_cast<float>((word >> 28) & 0x0F) - 8.0f, hi.w, local);
    return local;
}

template <bool UNROLL, bool STREAM>
__global__ void int4_gemv_kernel(const unsigned char* __restrict__ packed,
                                 const float* __restrict__ scales,
                                 const float* __restrict__ x,
                                 float* __restrict__ out,
                                 int n_rows,
                                 int n_cols) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= n_rows) {
        return;
    }

    const int n_groups = n_cols / GROUP;
    const unsigned char* row_bytes = packed + static_cast<long>(row) * (n_cols / 2);
    const float* row_scales = scales + static_cast<long>(row) * n_groups;

    float acc = 0.0f;

    if constexpr (UNROLL) {
        const int steps = n_cols / WEIGHTS_PER_STEP;
#pragma unroll 4
        for (int step = 0; step < steps; ++step) {
            const int weight = step * WEIGHTS_PER_STEP + lane * WEIGHTS_PER_LANE;
            const unsigned int word = load_word<STREAM>(reinterpret_cast<const unsigned int*>(row_bytes + weight / 2));
            const float4 act_lo = *reinterpret_cast<const float4*>(x + weight);
            const float4 act_hi = *reinterpret_cast<const float4*>(x + weight + 4);
            acc += dot_eight(word, act_lo, act_hi) * row_scales[weight / GROUP];
        }
        for (int base = steps * WEIGHTS_PER_STEP; base < n_cols; base += WEIGHTS_PER_STEP) {
            const int weight = base + lane * WEIGHTS_PER_LANE;
            if (weight >= n_cols) {
                continue;
            }
            const unsigned int word = load_word<STREAM>(reinterpret_cast<const unsigned int*>(row_bytes + weight / 2));
            const float4 act_lo = *reinterpret_cast<const float4*>(x + weight);
            const float4 act_hi = *reinterpret_cast<const float4*>(x + weight + 4);
            acc += dot_eight(word, act_lo, act_hi) * row_scales[weight / GROUP];
        }
    } else {
        for (int base = 0; base < n_cols; base += WEIGHTS_PER_STEP) {
            const int weight = base + lane * WEIGHTS_PER_LANE;
            if (weight >= n_cols) {
                continue;
            }
            const unsigned int word = *reinterpret_cast<const unsigned int*>(row_bytes + weight / 2);
            const float4 act_lo = *reinterpret_cast<const float4*>(x + weight);
            const float4 act_hi = *reinterpret_cast<const float4*>(x + weight + 4);
            acc += dot_eight(word, act_lo, act_hi) * row_scales[weight / GROUP];
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0) {
        out[row] = acc;
    }
}

}

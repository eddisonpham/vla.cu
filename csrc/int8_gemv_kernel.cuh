#pragma once

#include <cuda_runtime.h>

namespace vla_cu_int8 {

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARPS_PER_BLOCK * 32;
constexpr int WEIGHTS_PER_LANE = 16;
constexpr int WEIGHTS_PER_STEP = 32 * WEIGHTS_PER_LANE;

__device__ __forceinline__ float signed_byte(unsigned int word, int index) {
    return static_cast<float>(static_cast<signed char>((word >> (8 * index)) & 0xFFu));
}

__device__ __forceinline__ float dot_sixteen(const uint4& weight, const float4& a0, const float4& a1,
                                             const float4& a2, const float4& a3) {
    float local = 0.0f;
    local = fmaf(signed_byte(weight.x, 0), a0.x, local);
    local = fmaf(signed_byte(weight.x, 1), a0.y, local);
    local = fmaf(signed_byte(weight.x, 2), a0.z, local);
    local = fmaf(signed_byte(weight.x, 3), a0.w, local);
    local = fmaf(signed_byte(weight.y, 0), a1.x, local);
    local = fmaf(signed_byte(weight.y, 1), a1.y, local);
    local = fmaf(signed_byte(weight.y, 2), a1.z, local);
    local = fmaf(signed_byte(weight.y, 3), a1.w, local);
    local = fmaf(signed_byte(weight.z, 0), a2.x, local);
    local = fmaf(signed_byte(weight.z, 1), a2.y, local);
    local = fmaf(signed_byte(weight.z, 2), a2.z, local);
    local = fmaf(signed_byte(weight.z, 3), a2.w, local);
    local = fmaf(signed_byte(weight.w, 0), a3.x, local);
    local = fmaf(signed_byte(weight.w, 1), a3.y, local);
    local = fmaf(signed_byte(weight.w, 2), a3.z, local);
    local = fmaf(signed_byte(weight.w, 3), a3.w, local);
    return local;
}

__global__ void int8_gemv_kernel(const signed char* __restrict__ weights,
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

    const signed char* row_weights = weights + static_cast<long>(row) * n_cols;
    float acc = 0.0f;
    const int steps = n_cols / WEIGHTS_PER_STEP;

#pragma unroll 2
    for (int step = 0; step < steps; ++step) {
        const int offset = step * WEIGHTS_PER_STEP + lane * WEIGHTS_PER_LANE;
        const uint4 word = __ldcs(reinterpret_cast<const uint4*>(row_weights + offset));
        const float4 a0 = *reinterpret_cast<const float4*>(x + offset);
        const float4 a1 = *reinterpret_cast<const float4*>(x + offset + 4);
        const float4 a2 = *reinterpret_cast<const float4*>(x + offset + 8);
        const float4 a3 = *reinterpret_cast<const float4*>(x + offset + 12);
        acc += dot_sixteen(word, a0, a1, a2, a3);
    }

    for (int col = steps * WEIGHTS_PER_STEP + lane; col < n_cols; col += 32) {
        acc += static_cast<float>(row_weights[col]) * x[col];
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0) {
        out[row] = acc * scales[row];
    }
}

}

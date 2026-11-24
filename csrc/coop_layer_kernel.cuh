#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "int4_gemv_kernel.cuh"

namespace vla_cu_coop {

namespace cg = cooperative_groups;

constexpr int HIDDEN = 720;
constexpr int INTERMEDIATE = 1920;
constexpr int GROUP = 64;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAX_TOKENS = 64;
constexpr int UP_TOKENS = 16;
constexpr int DOWN_TOKENS = 8;
constexpr int SHARED_FLOATS = INTERMEDIATE * DOWN_TOKENS;
constexpr float EPS = 1e-5f;

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + __expf(-value));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float dot_four(int word, float4 a) {
    float local = 0.0f;
    local = fmaf(static_cast<float>(static_cast<signed char>(word & 0xFF)), a.x, local);
    local = fmaf(static_cast<float>(static_cast<signed char>((word >> 8) & 0xFF)), a.y, local);
    local = fmaf(static_cast<float>(static_cast<signed char>((word >> 16) & 0xFF)), a.z, local);
    local = fmaf(static_cast<float>(static_cast<signed char>((word >> 24) & 0xFF)), a.w, local);
    return local;
}

__global__ void coop_layer_kernel(const float* __restrict__ x,
                                  const float* __restrict__ other,
                                  const float* __restrict__ weight,
                                  const signed char* __restrict__ up,
                                  const float* __restrict__ up_scales,
                                  const unsigned char* __restrict__ down,
                                  const float* __restrict__ down_scales,
                                  float* __restrict__ added,
                                  float* __restrict__ normed,
                                  float* __restrict__ act,
                                  float* __restrict__ out,
                                  int tokens) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float shared[];
    float* row = shared;
    float* partial = shared + HIDDEN;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
        float squares = 0.0f;
        for (int k = threadIdx.x; k < HIDDEN; k += THREADS) {
            const float value = x[token * HIDDEN + k] + other[token * HIDDEN + k];
            row[k] = value;
            added[token * HIDDEN + k] = value;
            squares += value * value;
        }
        partial[threadIdx.x] = squares;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        const float inverse = rsqrtf(partial[0] / HIDDEN + EPS);
        for (int k = threadIdx.x; k < HIDDEN; k += THREADS) {
            normed[token * HIDDEN + k] = row[k] * inverse * weight[k];
        }
        __syncthreads();
    }

    grid.sync();

    for (int first = 0; first < tokens; first += UP_TOKENS) {
        const int held = tokens - first < UP_TOKENS ? tokens - first : UP_TOKENS;
        for (int i = threadIdx.x; i < held * HIDDEN; i += THREADS) {
            shared[i] = normed[(first + i / HIDDEN) * HIDDEN + i % HIDDEN];
        }
        __syncthreads();
        for (int out_row = blockIdx.x + gridDim.x * warp; out_row < INTERMEDIATE; out_row += gridDim.x * WARPS) {
            const signed char* weights = up + static_cast<long>(out_row) * HIDDEN;
            const float scale = up_scales[out_row];
            float acc[UP_TOKENS];
            for (int t = 0; t < held; ++t) {
                acc[t] = 0.0f;
            }
            for (int base = 0; base < HIDDEN; base += 128) {
                const int col = base + lane * 4;
                if (col + 4 <= HIDDEN) {
                    const int word = *reinterpret_cast<const int*>(weights + col);
                    for (int t = 0; t < held; ++t) {
                        acc[t] += dot_four(word, *reinterpret_cast<const float4*>(shared + t * HIDDEN + col));
                    }
                } else {
                    for (int c = col; c < HIDDEN; ++c) {
                        const float value = static_cast<float>(weights[c]);
                        for (int t = 0; t < held; ++t) {
                            acc[t] += value * shared[t * HIDDEN + c];
                        }
                    }
                }
            }
            for (int t = 0; t < held; ++t) {
                const float total = warp_sum(acc[t]);
                if (lane == 0) {
                    act[(first + t) * INTERMEDIATE + out_row] = silu(total * scale);
                }
            }
        }
        __syncthreads();
    }

    grid.sync();

    for (int first = 0; first < tokens; first += DOWN_TOKENS) {
        const int held = tokens - first < DOWN_TOKENS ? tokens - first : DOWN_TOKENS;
        for (int i = threadIdx.x; i < held * INTERMEDIATE; i += THREADS) {
            shared[i] = act[(first + i / INTERMEDIATE) * INTERMEDIATE + i % INTERMEDIATE];
        }
        __syncthreads();
        for (int out_row = blockIdx.x + gridDim.x * warp; out_row < HIDDEN; out_row += gridDim.x * WARPS) {
            const unsigned char* bytes = down + static_cast<long>(out_row) * (INTERMEDIATE / 2);
            const float* scales = down_scales + static_cast<long>(out_row) * (INTERMEDIATE / GROUP);
            float acc[DOWN_TOKENS];
            for (int t = 0; t < held; ++t) {
                acc[t] = 0.0f;
            }
            for (int base = 0; base < INTERMEDIATE; base += 256) {
                const int weight_index = base + lane * 8;
                if (weight_index < INTERMEDIATE) {
                    const unsigned int word = *reinterpret_cast<const unsigned int*>(bytes + weight_index / 2);
                    const float group_scale = scales[weight_index / GROUP];
                    for (int t = 0; t < held; ++t) {
                        const float4 lo = *reinterpret_cast<const float4*>(shared + t * INTERMEDIATE + weight_index);
                        const float4 hi = *reinterpret_cast<const float4*>(shared + t * INTERMEDIATE + weight_index + 4);
                        acc[t] += vla_cu_int4::dot_eight(word, lo, hi) * group_scale;
                    }
                }
            }
            for (int t = 0; t < held; ++t) {
                const float total = warp_sum(acc[t]);
                if (lane == 0) {
                    const int token = first + t;
                    out[token * HIDDEN + out_row] = added[token * HIDDEN + out_row] + total;
                }
            }
        }
        __syncthreads();
    }
}

}

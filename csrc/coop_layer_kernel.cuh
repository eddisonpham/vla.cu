#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "int4_gemv_kernel.cuh"

namespace vla_cu_coop {

constexpr int GROUP = 64;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAX_TOKENS = 64;
constexpr int UP_TOKENS = 16;
constexpr int DOWN_TOKENS = 8;
constexpr float EPS = 1e-5f;

inline int shared_bytes_for(int hidden, int intermediate) {
    const int up = UP_TOKENS * hidden;
    const int down = DOWN_TOKENS * intermediate;
    const int largest = up > down ? up : down;
    const int staged = largest > hidden + THREADS ? largest : hidden + THREADS;
    return (staged + 16) * static_cast<int>(sizeof(float));
}

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

__device__ __forceinline__ float dot_row(const signed char* weights, const float* row, int hidden, int lane) {
    float sum = 0.0f;
    for (int base = 0; base < hidden; base += 128) {
        const int col = base + lane * 4;
        if (col + 4 <= hidden) {
            const int word = *reinterpret_cast<const int*>(weights + col);
            sum += dot_four(word, *reinterpret_cast<const float4*>(row + col));
        } else {
            for (int c = col; c < hidden; ++c) {
                sum += static_cast<float>(weights[c]) * row[c];
            }
        }
    }
    return warp_sum(sum);
}

__device__ __forceinline__ void add_norm_stage(const float* __restrict__ x,
                                               const float* __restrict__ other,
                                               const float* __restrict__ weight,
                                               float* __restrict__ added,
                                               float* __restrict__ normed,
                                               float* row,
                                               float* partial,
                                               int tokens,
                                               int hidden) {
    for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
        float squares = 0.0f;
        for (int k = threadIdx.x; k < hidden; k += THREADS) {
            const float value = x[token * hidden + k] + other[token * hidden + k];
            row[k] = value;
            added[token * hidden + k] = value;
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
        const float inverse = rsqrtf(partial[0] / hidden + EPS);
        for (int k = threadIdx.x; k < hidden; k += THREADS) {
            normed[token * hidden + k] = row[k] * inverse * weight[k];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void up_stage(const float* __restrict__ normed,
                                         const signed char* __restrict__ up,
                                         const float* __restrict__ up_scales,
                                         float* __restrict__ act,
                                         float* shared,
                                         int tokens,
                                         int hidden,
                                         int intermediate) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int first = 0; first < tokens; first += UP_TOKENS) {
        const int held = tokens - first < UP_TOKENS ? tokens - first : UP_TOKENS;
        for (int i = threadIdx.x; i < held * hidden; i += THREADS) {
            shared[i] = normed[(first + i / hidden) * hidden + i % hidden];
        }
        __syncthreads();
        for (int out_row = blockIdx.x + gridDim.x * warp; out_row < intermediate; out_row += gridDim.x * WARPS) {
            const signed char* weights = up + static_cast<long>(out_row) * hidden;
            const float scale = up_scales[out_row];
            float acc[UP_TOKENS];
            for (int t = 0; t < held; ++t) {
                acc[t] = 0.0f;
            }
            for (int base = 0; base < hidden; base += 128) {
                const int col = base + lane * 4;
                if (col + 4 <= hidden) {
                    const int word = *reinterpret_cast<const int*>(weights + col);
                    for (int t = 0; t < held; ++t) {
                        acc[t] += dot_four(word, *reinterpret_cast<const float4*>(shared + t * hidden + col));
                    }
                } else {
                    for (int c = col; c < hidden; ++c) {
                        const float value = static_cast<float>(weights[c]);
                        for (int t = 0; t < held; ++t) {
                            acc[t] += value * shared[t * hidden + c];
                        }
                    }
                }
            }
            for (int t = 0; t < held; ++t) {
                const float total = warp_sum(acc[t]);
                if (lane == 0) {
                    act[(first + t) * intermediate + out_row] = silu(total * scale);
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void gated_stage(const float* __restrict__ normed,
                                            const signed char* __restrict__ gate,
                                            const float* __restrict__ gate_scales,
                                            const signed char* __restrict__ up,
                                            const float* __restrict__ up_scales,
                                            float* __restrict__ act,
                                            float* shared,
                                            int tokens,
                                            int hidden,
                                            int intermediate) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int first = 0; first < tokens; first += UP_TOKENS) {
        const int held = tokens - first < UP_TOKENS ? tokens - first : UP_TOKENS;
        for (int i = threadIdx.x; i < held * hidden; i += THREADS) {
            shared[i] = normed[(first + i / hidden) * hidden + i % hidden];
        }
        __syncthreads();
        for (int out_row = blockIdx.x + gridDim.x * warp; out_row < intermediate; out_row += gridDim.x * WARPS) {
            const signed char* gate_weights = gate + static_cast<long>(out_row) * hidden;
            const signed char* up_weights = up + static_cast<long>(out_row) * hidden;
            const float gate_scale = gate_scales[out_row];
            const float up_scale = up_scales[out_row];
            float gate_acc[UP_TOKENS];
            float up_acc[UP_TOKENS];
            for (int t = 0; t < held; ++t) {
                gate_acc[t] = 0.0f;
                up_acc[t] = 0.0f;
            }
            for (int base = 0; base < hidden; base += 128) {
                const int col = base + lane * 4;
                if (col + 4 <= hidden) {
                    const int gate_word = *reinterpret_cast<const int*>(gate_weights + col);
                    const int up_word = *reinterpret_cast<const int*>(up_weights + col);
                    for (int t = 0; t < held; ++t) {
                        const float4 values = *reinterpret_cast<const float4*>(shared + t * hidden + col);
                        gate_acc[t] += dot_four(gate_word, values);
                        up_acc[t] += dot_four(up_word, values);
                    }
                } else {
                    for (int c = col; c < hidden; ++c) {
                        const float gate_value = static_cast<float>(gate_weights[c]);
                        const float up_value = static_cast<float>(up_weights[c]);
                        for (int t = 0; t < held; ++t) {
                            gate_acc[t] += gate_value * shared[t * hidden + c];
                            up_acc[t] += up_value * shared[t * hidden + c];
                        }
                    }
                }
            }
            for (int t = 0; t < held; ++t) {
                const float gate_total = warp_sum(gate_acc[t]);
                const float up_total = warp_sum(up_acc[t]);
                if (lane == 0) {
                    act[(first + t) * intermediate + out_row] =
                        silu(gate_total * gate_scale) * (up_total * up_scale);
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void down_stage(const unsigned char* __restrict__ down,
                                           const float* __restrict__ down_scales,
                                           const float* __restrict__ act,
                                           const float* __restrict__ added,
                                           float* __restrict__ out,
                                           float* shared,
                                           int tokens,
                                           int hidden,
                                           int intermediate) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int first = 0; first < tokens; first += DOWN_TOKENS) {
        const int held = tokens - first < DOWN_TOKENS ? tokens - first : DOWN_TOKENS;
        for (int i = threadIdx.x; i < held * intermediate; i += THREADS) {
            shared[i] = act[(first + i / intermediate) * intermediate + i % intermediate];
        }
        __syncthreads();
        for (int out_row = blockIdx.x + gridDim.x * warp; out_row < hidden; out_row += gridDim.x * WARPS) {
            const unsigned char* bytes = down + static_cast<long>(out_row) * (intermediate / 2);
            const float* scales = down_scales + static_cast<long>(out_row) * (intermediate / GROUP);
            float acc[DOWN_TOKENS];
            for (int t = 0; t < held; ++t) {
                acc[t] = 0.0f;
            }
            for (int base = 0; base < intermediate; base += 256) {
                const int weight_index = base + lane * 8;
                if (weight_index < intermediate) {
                    const unsigned int word = *reinterpret_cast<const unsigned int*>(bytes + weight_index / 2);
                    const float group_scale = scales[weight_index / GROUP];
                    for (int t = 0; t < held; ++t) {
                        const float4 lo = *reinterpret_cast<const float4*>(shared + t * intermediate + weight_index);
                        const float4 hi = *reinterpret_cast<const float4*>(shared + t * intermediate + weight_index + 4);
                        acc[t] += vla_cu_int4::dot_eight(word, lo, hi) * group_scale;
                    }
                }
            }
            for (int t = 0; t < held; ++t) {
                const float total = warp_sum(acc[t]);
                if (lane == 0) {
                    const int token = first + t;
                    out[token * hidden + out_row] = added[token * hidden + out_row] + total;
                }
            }
        }
        __syncthreads();
    }
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
                                  int tokens,
                                  int hidden,
                                  int intermediate) {
    extern __shared__ float shared[];
    add_norm_stage(x, other, weight, added, normed, shared, shared + hidden, tokens, hidden);
    cooperative_groups::this_grid().sync();
    up_stage(normed, up, up_scales, act, shared, tokens, hidden, intermediate);
    cooperative_groups::this_grid().sync();
    down_stage(down, down_scales, act, added, out, shared, tokens, hidden, intermediate);
}

__global__ void coop_gated_kernel(const float* __restrict__ x,
                                  const float* __restrict__ other,
                                  const float* __restrict__ weight,
                                  const signed char* __restrict__ gate,
                                  const float* __restrict__ gate_scales,
                                  const signed char* __restrict__ up,
                                  const float* __restrict__ up_scales,
                                  const unsigned char* __restrict__ down,
                                  const float* __restrict__ down_scales,
                                  float* __restrict__ added,
                                  float* __restrict__ normed,
                                  float* __restrict__ act,
                                  float* __restrict__ out,
                                  int tokens,
                                  int hidden,
                                  int intermediate) {
    extern __shared__ float shared[];
    add_norm_stage(x, other, weight, added, normed, shared, shared + hidden, tokens, hidden);
    cooperative_groups::this_grid().sync();
    gated_stage(normed, gate, gate_scales, up, up_scales, act, shared, tokens, hidden, intermediate);
    cooperative_groups::this_grid().sync();
    down_stage(down, down_scales, act, added, out, shared, tokens, hidden, intermediate);
}

}

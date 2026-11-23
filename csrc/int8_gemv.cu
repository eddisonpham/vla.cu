// Fused int8 dequantize and GEMV for batch 1, one warp per output row, streaming loads.
//
// Thread to data mapping: a warp owns one output row. Each lane loads 16 int8 weights
// as one 16-byte vector and the matching 16 activations as four float4 loads, so a warp
// step covers 512 contiguous weights with 512 bytes of weight traffic. The warp reduces
// with shuffles and lane 0 applies the row scale and writes the output, so there is no
// shared memory and no barrier.
//
// Memory access: one 16-byte load per lane per step, fully coalesced, marked streaming
// with __ldcs because each weight is read exactly once and must not evict the activation
// vector, which every row reuses. A guarded scalar tail handles K not divisible by 512,
// which is the common case for real widths such as 960. K must be a multiple of 16 so
// that every row base stays 16-byte aligned for the vector loads.
//
// The launch uses the current torch stream, not the legacy default stream. That is what
// makes the kernel capturable: a kernel on the legacy default stream is invisible to
// stream capture, so a graph would record everything except this launch.
//
// Expected bottleneck: memory, weights only, N * K bytes.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

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

}  // namespace

void int8_gemv_launch(torch::Tensor weights, torch::Tensor scales, torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(weights.is_cuda() && scales.is_cuda() && x.is_cuda() && out.is_cuda(), "expected CUDA tensors");
    TORCH_CHECK(weights.scalar_type() == torch::kInt8, "expected int8 weights");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32 && x.scalar_type() == torch::kFloat32 && out.scalar_type() == torch::kFloat32,
                "expected fp32 scales, activations and output");
    TORCH_CHECK(weights.is_contiguous() && scales.is_contiguous() && x.is_contiguous() && out.is_contiguous(),
                "expected contiguous tensors");
    TORCH_CHECK(weights.dim() == 2 && scales.dim() == 1 && x.dim() == 1 && out.dim() == 1,
                "expected w [N, K], scales [N], x [K], out [N]");

    const int n_rows = static_cast<int>(weights.size(0));
    const int n_cols = static_cast<int>(weights.size(1));
    TORCH_CHECK(x.size(0) == n_cols && out.size(0) == n_rows && scales.size(0) == n_rows,
                "shape mismatch between w, scales, x and out");
    TORCH_CHECK(n_cols % 16 == 0, "K must be a multiple of 16 so every row stays 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "activations must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(weights.data_ptr()) % 16 == 0, "weights must be 16-byte aligned");
    if (n_rows == 0 || n_cols == 0) {
        return;
    }

    const int blocks = (n_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int8_gemv_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const signed char*>(weights.data_ptr()),
        scales.data_ptr<float>(),
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        n_rows,
        n_cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemv_launch", &int8_gemv_launch, "fused int8 to fp32 GEMV, warp per row, streaming loads");
}

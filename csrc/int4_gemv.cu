// Fused 4-bit dequantize and GEMV for batch 1, one warp per output row.
//
// Thread to data mapping: a warp owns one output row. Each lane loads four packed
// bytes (a uint32) per step, which is eight weights, and the eight lanes of a warp
// segment share one 64-weight group, so the group scale is a broadcast load instead
// of a per-byte gather. The warp reduces with shuffles and lane 0 writes the output,
// so no shared memory and no block-wide barrier are needed.
//
// Memory access: packed bytes are read four at a time per lane, 128 contiguous bytes
// per warp instruction, and each lane reads its eight activations with two float4
// loads. Weights are read once, activations once per row that is resident in cache.
// Expected bottleneck: memory, weights only, N * K / 2 bytes.
//
// Three variants share this file so one run can attribute cost to one change at a
// time. Variant 0 keeps the original single guarded loop. Variant 1 splits the loop
// into a full-trip-count main loop with no bounds check, which lets the compiler
// unroll four steps and keep four weight loads in flight, plus a guarded tail.
// Variant 2 is variant 1 with the weight load marked streaming (__ldcs), because
// each packed word is read exactly once and should not evict the activation vector.
//
// Every variant launches on the current torch stream rather than the legacy default
// stream, because a launch on the legacy stream is invisible to stream capture and a
// CUDA graph would then record everything except the kernel.
//
// Preconditions enforced by the host wrapper: group size 64, K divisible by 64, all
// tensors fp32 or uint8, contiguous, and 16-byte aligned activations.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

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

void launch_impl(torch::Tensor packed, torch::Tensor scales, torch::Tensor x, torch::Tensor out, int variant) {
    TORCH_CHECK(packed.is_cuda() && scales.is_cuda() && x.is_cuda() && out.is_cuda(), "expected CUDA tensors");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "expected uint8 packed weights");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32 && x.scalar_type() == torch::kFloat32 && out.scalar_type() == torch::kFloat32,
                "expected fp32 scales, activations and output");
    TORCH_CHECK(packed.is_contiguous() && scales.is_contiguous() && x.is_contiguous() && out.is_contiguous(),
                "expected contiguous tensors");
    TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2 && x.dim() == 1 && out.dim() == 1,
                "expected packed [N, K / 2], scales [N, K / 64], x [K], out [N]");

    const int n_rows = static_cast<int>(packed.size(0));
    const int n_cols = static_cast<int>(packed.size(1) * 2);
    TORCH_CHECK(n_cols % GROUP == 0, "K must be divisible by 64");
    TORCH_CHECK(x.size(0) == n_cols && out.size(0) == n_rows, "shape mismatch between packed, x and out");
    TORCH_CHECK(scales.size(0) == n_rows && scales.size(1) == n_cols / GROUP, "scale table shape mismatch");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "activations must be 16-byte aligned");
    if (n_rows == 0) {
        return;
    }

    const int blocks = (n_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned char* packed_ptr = reinterpret_cast<const unsigned char*>(packed.data_ptr());
    const float* scale_ptr = scales.data_ptr<float>();
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (variant == 0) {
        int4_gemv_kernel<false, false><<<blocks, THREADS, 0, stream>>>(packed_ptr, scale_ptr, x_ptr, out_ptr, n_rows, n_cols);
    } else if (variant == 1) {
        int4_gemv_kernel<true, false><<<blocks, THREADS, 0, stream>>>(packed_ptr, scale_ptr, x_ptr, out_ptr, n_rows, n_cols);
    } else {
        int4_gemv_kernel<true, true><<<blocks, THREADS, 0, stream>>>(packed_ptr, scale_ptr, x_ptr, out_ptr, n_rows, n_cols);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void int4_gemv_launch(torch::Tensor packed, torch::Tensor scales, torch::Tensor x, torch::Tensor out) {
    launch_impl(packed, scales, x, out, 0);
}

void int4_gemv_unroll_launch(torch::Tensor packed, torch::Tensor scales, torch::Tensor x, torch::Tensor out) {
    launch_impl(packed, scales, x, out, 1);
}

void int4_gemv_stream_launch(torch::Tensor packed, torch::Tensor scales, torch::Tensor x, torch::Tensor out) {
    launch_impl(packed, scales, x, out, 2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_gemv_launch", &int4_gemv_launch, "fused 4-bit dequantize and GEMV, guarded loop");
    m.def("int4_gemv_unroll_launch", &int4_gemv_unroll_launch, "same kernel, split loop, unroll 4");
    m.def("int4_gemv_stream_launch", &int4_gemv_stream_launch, "same kernel, unrolled, streaming weight loads");
}

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "int4_gemv_kernel.cuh"

using namespace vla_cu_int4;

namespace {

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

}

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

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "int8_gemv_kernel.cuh"

using namespace vla_cu_int8;

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

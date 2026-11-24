#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "coop_layer_kernel.cuh"

using namespace vla_cu_coop;

void coop_layer_launch(torch::Tensor x,
                       torch::Tensor other,
                       torch::Tensor weight,
                       torch::Tensor up,
                       torch::Tensor up_scales,
                       torch::Tensor down,
                       torch::Tensor down_scales,
                       torch::Tensor added,
                       torch::Tensor normed,
                       torch::Tensor act,
                       torch::Tensor out,
                       int64_t blocks) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32, "x must be contiguous fp32 CUDA");
    TORCH_CHECK(other.is_cuda() && other.is_contiguous() && other.scalar_type() == torch::kFloat32, "other must be contiguous fp32 CUDA");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32, "weight must be contiguous fp32 CUDA");
    TORCH_CHECK(up.is_cuda() && up.is_contiguous() && up.scalar_type() == torch::kInt8, "up must be contiguous int8 CUDA");
    TORCH_CHECK(up_scales.is_cuda() && up_scales.is_contiguous() && up_scales.scalar_type() == torch::kFloat32, "up_scales must be contiguous fp32 CUDA");
    TORCH_CHECK(down.is_cuda() && down.is_contiguous() && down.scalar_type() == torch::kUInt8, "down must be contiguous uint8 CUDA");
    TORCH_CHECK(down_scales.is_cuda() && down_scales.is_contiguous() && down_scales.scalar_type() == torch::kFloat32, "down_scales must be contiguous fp32 CUDA");
    TORCH_CHECK(added.is_cuda() && added.is_contiguous() && added.scalar_type() == torch::kFloat32, "added must be contiguous fp32 CUDA");
    TORCH_CHECK(normed.is_cuda() && normed.is_contiguous() && normed.scalar_type() == torch::kFloat32, "normed must be contiguous fp32 CUDA");
    TORCH_CHECK(act.is_cuda() && act.is_contiguous() && act.scalar_type() == torch::kFloat32, "act must be contiguous fp32 CUDA");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kFloat32, "out must be contiguous fp32 CUDA");

    const int tokens = static_cast<int>(x.size(0));
    TORCH_CHECK(tokens >= 1 && tokens <= MAX_TOKENS, "tokens must be in [1, ", MAX_TOKENS, "], got ", tokens);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == HIDDEN && other.sizes() == x.sizes(), "expected x and other [tokens, ", HIDDEN, "]");
    TORCH_CHECK(added.sizes() == x.sizes() && normed.sizes() == x.sizes() && out.sizes() == x.sizes(), "expected added, normed and out [tokens, ", HIDDEN, "]");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == HIDDEN, "expected weight [", HIDDEN, "]");
    TORCH_CHECK(up.dim() == 2 && up.size(0) == INTERMEDIATE && up.size(1) == HIDDEN, "expected up [", INTERMEDIATE, ", ", HIDDEN, "]");
    TORCH_CHECK(up_scales.dim() == 1 && up_scales.size(0) == INTERMEDIATE, "expected up_scales [", INTERMEDIATE, "]");
    TORCH_CHECK(down.dim() == 2 && down.size(0) == HIDDEN && down.size(1) == INTERMEDIATE / 2, "expected down [", HIDDEN, ", ", INTERMEDIATE / 2, "]");
    TORCH_CHECK(down_scales.dim() == 2 && down_scales.size(0) == HIDDEN && down_scales.size(1) == INTERMEDIATE / GROUP, "expected down_scales [", HIDDEN, ", ", INTERMEDIATE / GROUP, "]");
    TORCH_CHECK(act.dim() == 2 && act.size(0) == tokens && act.size(1) == INTERMEDIATE, "expected act [tokens, ", INTERMEDIATE, "]");

    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int supported = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
    TORCH_CHECK(supported != 0, "this device does not support cooperative launch");
    int multiprocessors = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const int residency = multiprocessors;
    const int grid_blocks = blocks > 0 ? static_cast<int>(blocks) : residency;
    TORCH_CHECK(grid_blocks >= 1 && grid_blocks <= residency, "a cooperative launch here allows 1 to ", residency, " blocks, got ", grid_blocks);

    const size_t shared_bytes = SHARED_FLOATS * sizeof(float);
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(coop_layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
        configured = true;
    }

    const float* x_ptr = x.data_ptr<float>();
    const float* other_ptr = other.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const signed char* up_ptr = reinterpret_cast<const signed char*>(up.data_ptr());
    const float* up_scales_ptr = up_scales.data_ptr<float>();
    const unsigned char* down_ptr = reinterpret_cast<const unsigned char*>(down.data_ptr());
    const float* down_scales_ptr = down_scales.data_ptr<float>();
    float* added_ptr = added.data_ptr<float>();
    float* normed_ptr = normed.data_ptr<float>();
    float* act_ptr = act.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int token_count = tokens;

    void* arguments[] = {&x_ptr, &other_ptr, &weight_ptr, &up_ptr, &up_scales_ptr, &down_ptr,
                         &down_scales_ptr, &added_ptr, &normed_ptr, &act_ptr, &out_ptr, &token_count};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(coop_layer_kernel),
                                               dim3(grid_blocks),
                                               dim3(THREADS),
                                               arguments,
                                               shared_bytes,
                                               at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("coop_layer_launch", &coop_layer_launch, "one cooperative launch for the residual tail and the MLP");
}

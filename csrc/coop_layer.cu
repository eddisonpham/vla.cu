#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <mutex>

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

    const auto aligned = [](const torch::Tensor& tensor, int bytes) {
        return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % bytes == 0;
    };
    TORCH_CHECK(aligned(x, 16) && aligned(other, 16) && aligned(added, 16) && aligned(normed, 16) &&
                    aligned(act, 16) && aligned(out, 16),
                "float4 loads need 16-byte aligned x, other, added, normed, act and out");
    TORCH_CHECK(aligned(up, 4) && aligned(down, 4) && aligned(weight, 4) && aligned(up_scales, 4) &&
                    aligned(down_scales, 4),
                "word loads need 4-byte aligned weights and scales");

    const int tokens = static_cast<int>(x.size(0));
    TORCH_CHECK(tokens >= 1 && tokens <= MAX_TOKENS, "tokens must be in [1, ", MAX_TOKENS, "], got ", tokens);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 1, "expected x [tokens, hidden] and weight [hidden]");
    const int hidden = static_cast<int>(weight.size(0));
    const int intermediate = up.dim() == 2 ? static_cast<int>(up.size(0)) : 0;
    TORCH_CHECK(x.size(1) == hidden && other.sizes() == x.sizes(), "expected x and other [tokens, ", hidden, "]");
    TORCH_CHECK(added.sizes() == x.sizes() && normed.sizes() == x.sizes() && out.sizes() == x.sizes(),
                "expected added, normed and out [tokens, ", hidden, "]");
    TORCH_CHECK(hidden % 16 == 0, "hidden must be a multiple of 16 for the tensor core K step, got ", hidden);
    TORCH_CHECK(up.dim() == 2 && up.size(1) == hidden && intermediate >= 16, "expected up [intermediate, ", hidden, "]");
    TORCH_CHECK(intermediate % GROUP == 0, "intermediate must be a multiple of ", GROUP, " for the int4 groups, got ", intermediate);
    TORCH_CHECK(up_scales.dim() == 1 && up_scales.size(0) == intermediate, "expected up_scales [", intermediate, "]");
    TORCH_CHECK(down.dim() == 2 && down.size(0) == hidden && down.size(1) == intermediate / 2,
                "expected down [", hidden, ", ", intermediate / 2, "]");
    TORCH_CHECK(down_scales.dim() == 2 && down_scales.size(0) == hidden && down_scales.size(1) == intermediate / GROUP,
                "expected down_scales [", hidden, ", ", intermediate / GROUP, "]");
    TORCH_CHECK(act.dim() == 2 && act.size(0) == tokens && act.size(1) == intermediate,
                "expected act [tokens, ", intermediate, "]");

    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int supported = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
    TORCH_CHECK(supported != 0, "this device does not support cooperative launch");
    int multiprocessors = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    int shared_limit = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    const size_t shared_bytes = static_cast<size_t>(shared_bytes_for(hidden, intermediate));
    TORCH_CHECK(static_cast<long>(shared_bytes) <= shared_limit,
                "this layer needs ", shared_bytes, " bytes of shared memory against a limit of ", shared_limit);

    static std::mutex setup_mutex;
    {
        const std::lock_guard<std::mutex> guard(setup_mutex);
        static int configured_device = -1;
        static size_t configured_bytes = 0;
        if (configured_device != device || configured_bytes < shared_bytes) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(coop_layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(shared_bytes)));
            configured_device = device;
            configured_bytes = shared_bytes;
        }
    }

    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, coop_layer_kernel, THREADS, shared_bytes));
    TORCH_CHECK(per_sm >= 1, "the kernel does not fit one block on an SM with ", shared_bytes, " bytes of shared memory");
    const int capacity = per_sm * multiprocessors;
    const int default_blocks = capacity < multiprocessors ? capacity : multiprocessors;
    const int grid_blocks = blocks > 0 ? static_cast<int>(blocks) : default_blocks;
    TORCH_CHECK(grid_blocks >= 1 && grid_blocks <= capacity, "a cooperative launch allows 1 to ", capacity, " co-resident blocks, got ", grid_blocks);

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
    int hidden_width = hidden;
    int intermediate_width = intermediate;

    void* arguments[] = {&x_ptr, &other_ptr, &weight_ptr, &up_ptr, &up_scales_ptr, &down_ptr,
                         &down_scales_ptr, &added_ptr, &normed_ptr, &act_ptr, &out_ptr,
                         &token_count, &hidden_width, &intermediate_width};
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

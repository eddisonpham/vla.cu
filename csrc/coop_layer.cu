#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <mutex>
#include <utility>

#include "coop_layer_kernel.cuh"

using namespace vla_cu_coop;

namespace {

struct Plan {
    int blocks;
    size_t shared_bytes;
};

bool aligned(const torch::Tensor& tensor, int bytes) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % bytes == 0;
}

void check_float(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.scalar_type() == torch::kFloat32,
                name, " must be contiguous fp32 CUDA");
}

void check_float4(const torch::Tensor& tensor, const char* name) {
    check_float(tensor, name);
    TORCH_CHECK(aligned(tensor, 16), "float4 loads need 16-byte aligned ", name);
}

void check_int8_matrix(const torch::Tensor& codes, const torch::Tensor& scales, int rows, int cols, const char* name) {
    TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.scalar_type() == torch::kInt8,
                name, " must be contiguous int8 CUDA");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type() == torch::kFloat32,
                name, "_scales must be contiguous fp32 CUDA");
    TORCH_CHECK(codes.dim() == 2 && codes.size(0) == rows && codes.size(1) == cols,
                "expected ", name, " [", rows, ", ", cols, "]");
    TORCH_CHECK(scales.dim() == 1 && scales.size(0) == rows, "expected ", name, "_scales [", rows, "]");
    TORCH_CHECK(aligned(codes, 4) && aligned(scales, 4),
                "word loads need 4-byte aligned ", name, " codes and scales");
}

void check_down(const torch::Tensor& down, const torch::Tensor& down_scales, int hidden, int intermediate) {
    TORCH_CHECK(down.is_cuda() && down.is_contiguous() && down.scalar_type() == torch::kUInt8,
                "down must be contiguous uint8 CUDA");
    TORCH_CHECK(down_scales.is_cuda() && down_scales.is_contiguous() && down_scales.scalar_type() == torch::kFloat32,
                "down_scales must be contiguous fp32 CUDA");
    TORCH_CHECK(down.dim() == 2 && down.size(0) == hidden && down.size(1) == intermediate / 2,
                "expected down [", hidden, ", ", intermediate / 2, "]");
    TORCH_CHECK(down_scales.dim() == 2 && down_scales.size(0) == hidden && down_scales.size(1) == intermediate / GROUP,
                "expected down_scales [", hidden, ", ", intermediate / GROUP, "]");
    TORCH_CHECK(aligned(down, 4) && aligned(down_scales, 4),
                "word loads need 4-byte aligned down weights and scales");
}

std::pair<int, int> check_tail(const torch::Tensor& x,
                               const torch::Tensor& other,
                               const torch::Tensor& weight,
                               const torch::Tensor& added,
                               const torch::Tensor& normed,
                               const torch::Tensor& out) {
    check_float4(x, "x");
    check_float4(other, "other");
    check_float4(added, "added");
    check_float4(normed, "normed");
    check_float4(out, "out");
    check_float(weight, "weight");

    const int tokens = static_cast<int>(x.size(0));
    TORCH_CHECK(tokens >= 1 && tokens <= MAX_TOKENS, "tokens must be in [1, ", MAX_TOKENS, "], got ", tokens);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 1, "expected x [tokens, hidden] and weight [hidden]");
    const int hidden = static_cast<int>(weight.size(0));
    TORCH_CHECK(x.size(1) == hidden && other.sizes() == x.sizes(),
                "expected x and other [tokens, ", hidden, "]");
    TORCH_CHECK(added.sizes() == x.sizes() && normed.sizes() == x.sizes() && out.sizes() == x.sizes(),
                "expected added, normed and out [tokens, ", hidden, "]");
    TORCH_CHECK(hidden % 16 == 0, "hidden must be a multiple of 16 for the tensor core K step, got ", hidden);
    TORCH_CHECK(aligned(weight, 4), "word loads need 4-byte aligned weight");
    return {tokens, hidden};
}

void check_act(const torch::Tensor& act, int tokens, int intermediate) {
    check_float(act, "act");
    TORCH_CHECK(act.dim() == 2 && act.size(0) == tokens && act.size(1) == intermediate,
                "expected act [", tokens, ", ", intermediate, "]");
}

template <typename Kernel>
Plan plan_launch(Kernel kernel, int hidden, int intermediate, int64_t blocks) {
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
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(shared_bytes)));
            configured_device = device;
            configured_bytes = shared_bytes;
        }
    }

    int per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, THREADS, shared_bytes));
    TORCH_CHECK(per_sm >= 1, "the kernel does not fit one block on an SM with ", shared_bytes, " bytes of shared memory");
    const int capacity = per_sm * multiprocessors;
    const int default_blocks = capacity < multiprocessors ? capacity : multiprocessors;
    const int grid_blocks = blocks > 0 ? static_cast<int>(blocks) : default_blocks;
    TORCH_CHECK(grid_blocks >= 1 && grid_blocks <= capacity,
                "a cooperative launch allows 1 to ", capacity, " co-resident blocks, got ", grid_blocks);
    return {grid_blocks, shared_bytes};
}

}

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
    const std::pair<int, int> tail = check_tail(x, other, weight, added, normed, out);
    const int tokens = tail.first;
    const int hidden = tail.second;
    TORCH_CHECK(up.dim() == 2 && up.size(1) == hidden, "expected up [intermediate, ", hidden, "]");
    const int intermediate = static_cast<int>(up.size(0));
    TORCH_CHECK(intermediate % GROUP == 0,
                "intermediate must be a multiple of ", GROUP, " for the int4 groups, got ", intermediate);
    check_int8_matrix(up, up_scales, intermediate, hidden, "up");
    check_down(down, down_scales, hidden, intermediate);
    check_act(act, tokens, intermediate);

    const Plan plan = plan_launch(coop_layer_kernel, hidden, intermediate, blocks);
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
                                               dim3(plan.blocks),
                                               dim3(THREADS),
                                               arguments,
                                               plan.shared_bytes,
                                               at::cuda::getCurrentCUDAStream()));
}

void coop_gated_launch(torch::Tensor x,
                       torch::Tensor other,
                       torch::Tensor weight,
                       torch::Tensor gate,
                       torch::Tensor gate_scales,
                       torch::Tensor up,
                       torch::Tensor up_scales,
                       torch::Tensor down,
                       torch::Tensor down_scales,
                       torch::Tensor added,
                       torch::Tensor normed,
                       torch::Tensor act,
                       torch::Tensor out,
                       int64_t blocks) {
    const std::pair<int, int> tail = check_tail(x, other, weight, added, normed, out);
    const int tokens = tail.first;
    const int hidden = tail.second;
    TORCH_CHECK(gate.dim() == 2 && gate.size(1) == hidden, "expected gate [intermediate, ", hidden, "]");
    TORCH_CHECK(up.sizes() == gate.sizes(), "expected gate and up to have the same shape");
    const int intermediate = static_cast<int>(gate.size(0));
    TORCH_CHECK(intermediate % GROUP == 0,
                "intermediate must be a multiple of ", GROUP, " for the int4 groups, got ", intermediate);
    check_int8_matrix(gate, gate_scales, intermediate, hidden, "gate");
    check_int8_matrix(up, up_scales, intermediate, hidden, "up");
    check_down(down, down_scales, hidden, intermediate);
    check_act(act, tokens, intermediate);

    const Plan plan = plan_launch(coop_gated_kernel, hidden, intermediate, blocks);
    const float* x_ptr = x.data_ptr<float>();
    const float* other_ptr = other.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const signed char* gate_ptr = reinterpret_cast<const signed char*>(gate.data_ptr());
    const float* gate_scales_ptr = gate_scales.data_ptr<float>();
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

    void* arguments[] = {&x_ptr, &other_ptr, &weight_ptr, &gate_ptr, &gate_scales_ptr, &up_ptr,
                         &up_scales_ptr, &down_ptr, &down_scales_ptr, &added_ptr, &normed_ptr,
                         &act_ptr, &out_ptr, &token_count, &hidden_width, &intermediate_width};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(coop_gated_kernel),
                                               dim3(plan.blocks),
                                               dim3(THREADS),
                                               arguments,
                                               plan.shared_bytes,
                                               at::cuda::getCurrentCUDAStream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("coop_layer_launch", &coop_layer_launch, "one cooperative launch for the residual tail and a two-matrix MLP");
    m.def("coop_gated_launch", &coop_gated_launch, "one cooperative launch for the residual tail and a gated MLP");
}

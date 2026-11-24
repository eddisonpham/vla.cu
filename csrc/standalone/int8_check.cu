#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../int8_gemv_kernel.cuh"

using namespace vla_cu_int8;

namespace {

float host_reference(const std::vector<signed char>& weights,
                     const std::vector<float>& scales,
                     const std::vector<float>& x,
                     int row,
                     int n_cols) {
    double total = 0.0;
    for (int col = 0; col < n_cols; ++col) {
        total += static_cast<double>(weights[row * n_cols + col]) * static_cast<double>(x[col]);
    }
    return static_cast<float>(total * static_cast<double>(scales[row]));
}

}

int main() {
    const int n_rows = 511;
    const int n_cols = 1040;

    std::vector<signed char> weights(static_cast<size_t>(n_rows) * n_cols);
    std::vector<float> scales(n_rows);
    std::vector<float> x(n_cols);
    std::srand(11);
    for (auto& weight : weights) {
        weight = static_cast<signed char>(std::rand() % 254 - 127);
    }
    for (auto& scale : scales) {
        scale = 0.001f * static_cast<float>(1 + std::rand() % 10);
    }
    for (auto& value : x) {
        value = static_cast<float>(std::rand() % 1000 - 500) / 500.0f;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::printf("device free %.1f MB of %.1f MB, %d x %d, %d steps plus a tail\n",
                free_bytes / 1e6, total_bytes / 1e6, n_rows, n_cols, n_cols / WEIGHTS_PER_STEP);

    signed char* device_weights = nullptr;
    float* device_scales = nullptr;
    float* device_x = nullptr;
    float* device_out = nullptr;
    cudaMalloc(&device_weights, weights.size());
    cudaMalloc(&device_scales, scales.size() * sizeof(float));
    cudaMalloc(&device_x, x.size() * sizeof(float));
    cudaMalloc(&device_out, n_rows * sizeof(float));
    cudaMemcpy(device_weights, weights.data(), weights.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(device_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> host_out(n_rows, 0.0f);
    std::vector<float> reference(n_rows);
    for (int row = 0; row < n_rows; ++row) {
        reference[row] = host_reference(weights, scales, x, row, n_cols);
    }

    const int blocks = (n_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int8_gemv_kernel<<<blocks, THREADS>>>(device_weights, device_scales, device_x, device_out, n_rows, n_cols);
    const cudaError_t status = cudaDeviceSynchronize();
    std::printf("kernel: %s\n", cudaGetErrorString(status));
    if (status != cudaSuccess) {
        return 1;
    }
    cudaMemcpy(host_out.data(), device_out, n_rows * sizeof(float), cudaMemcpyDeviceToHost);

    float worst = 0.0f;
    float worst_scale = 0.0f;
    for (int row = 0; row < n_rows; ++row) {
        const float error = std::fabs(host_out[row] - reference[row]);
        if (error > worst) {
            worst = error;
            worst_scale = std::fabs(reference[row]);
        }
    }
    const bool ok = worst <= 1e-3f + 1e-4f * worst_scale;
    std::printf("  worst abs error %.3e at reference %.3e, %s\n", worst, worst_scale, ok ? "ok" : "FAIL");

    cudaFree(device_weights);
    cudaFree(device_scales);
    cudaFree(device_x);
    cudaFree(device_out);
    return ok ? 0 : 1;
}

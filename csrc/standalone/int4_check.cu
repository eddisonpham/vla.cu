#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../int4_gemv_kernel.cuh"

using namespace vla_cu_int4;

namespace {

float host_reference(const std::vector<unsigned char>& packed,
                     const std::vector<float>& scales,
                     const std::vector<float>& x,
                     int row,
                     int n_cols) {
    const int row_bytes = n_cols / 2;
    const int n_groups = n_cols / GROUP;
    double total = 0.0;
    for (int col = 0; col < n_cols; ++col) {
        const unsigned char byte = packed[row * row_bytes + col / 2];
        const int nibble = (col % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        const double weight = (nibble - 8) * static_cast<double>(scales[row * n_groups + col / GROUP]);
        total += weight * static_cast<double>(x[col]);
    }
    return static_cast<float>(total);
}

void run_variant(int variant,
                 const unsigned char* packed,
                 const float* scales,
                 const float* x,
                 float* out,
                 int n_rows,
                 int n_cols,
                 float* host_out) {
    const int blocks = (n_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaMemset(out, 0, sizeof(float) * n_rows);
    if (variant == 0) {
        int4_gemv_kernel<false, false><<<blocks, THREADS>>>(packed, scales, x, out, n_rows, n_cols);
    } else if (variant == 1) {
        int4_gemv_kernel<true, false><<<blocks, THREADS>>>(packed, scales, x, out, n_rows, n_cols);
    } else {
        int4_gemv_kernel<true, true><<<blocks, THREADS>>>(packed, scales, x, out, n_rows, n_cols);
    }
    const cudaError_t status = cudaDeviceSynchronize();
    std::printf("variant %d: %s\n", variant, cudaGetErrorString(status));
    if (status != cudaSuccess) {
        return;
    }
    cudaMemcpy(host_out, out, sizeof(float) * n_rows, cudaMemcpyDeviceToHost);
}

}

int main() {
    const int n_rows = 512;
    const int n_cols = 1024;
    const int row_bytes = n_cols / 2;
    const int n_groups = n_cols / GROUP;

    std::vector<unsigned char> packed(static_cast<size_t>(n_rows) * row_bytes);
    std::vector<float> scales(static_cast<size_t>(n_rows) * n_groups);
    std::vector<float> x(n_cols);
    std::srand(7);
    for (auto& byte : packed) {
        byte = static_cast<unsigned char>(std::rand() & 0xFF);
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
    std::printf("device free %.1f MB of %.1f MB, %d x %d, %d rows of %d bytes\n",
                free_bytes / 1e6, total_bytes / 1e6, n_rows, n_cols, n_rows, n_cols / 2);

    unsigned char* device_packed = nullptr;
    float* device_scales = nullptr;
    float* device_x = nullptr;
    float* device_out = nullptr;
    cudaMalloc(&device_packed, packed.size());
    cudaMalloc(&device_scales, scales.size() * sizeof(float));
    cudaMalloc(&device_x, x.size() * sizeof(float));
    cudaMalloc(&device_out, n_rows * sizeof(float));
    cudaMemcpy(device_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(device_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> host_out(n_rows);
    std::vector<float> reference(n_rows);
    for (int row = 0; row < n_rows; ++row) {
        reference[row] = host_reference(packed, scales, x, row, n_cols);
    }

    int failures = 0;
    for (int variant = 0; variant < 3; ++variant) {
        run_variant(variant, device_packed, device_scales, device_x, device_out, n_rows, n_cols, host_out.data());
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
        if (!ok) {
            ++failures;
        }
    }

    cudaFree(device_packed);
    cudaFree(device_scales);
    cudaFree(device_x);
    cudaFree(device_out);
    return failures == 0 ? 0 : 1;
}

#include "gpu_perf/cuda_gemm.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace gpu_perf {
namespace {

void require_cuda_success(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t element_count) {
        require_cuda_success(
            cudaMalloc(reinterpret_cast<void**>(&data_), element_count * sizeof(float)),
            "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] float* data() noexcept {
        return data_;
    }

private:
    float* data_{nullptr};
};

__global__ void gemm_naive_kernel(
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k_dimension,
    std::size_t n) {
    const std::size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || column >= n) {
        return;
    }

    float accumulator = 0.0F;
    for (std::size_t k = 0; k < k_dimension; ++k) {
        accumulator += a[row * k_dimension + k] * b[k * n + column];
    }
    c[row * n + column] = accumulator;
}

}  // namespace

Matrix gemm_cuda_naive(const Matrix& a, const Matrix& b) {
    if (a.columns() != b.rows()) {
        throw std::invalid_argument("GEMM inner dimensions must match");
    }

    Matrix c(a.rows(), b.columns());
    if (c.size() == 0 || a.columns() == 0) {
        return c;
    }

    DeviceBuffer device_a(a.size());
    DeviceBuffer device_b(b.size());
    DeviceBuffer device_c(c.size());

    require_cuda_success(
        cudaMemcpy(
            device_a.data(),
            a.data(),
            a.size() * sizeof(float),
            cudaMemcpyHostToDevice),
        "copy A to device");
    require_cuda_success(
        cudaMemcpy(
            device_b.data(),
            b.data(),
            b.size() * sizeof(float),
            cudaMemcpyHostToDevice),
        "copy B to device");

    constexpr unsigned int block_width = 16;
    const dim3 block(block_width, block_width);
    const dim3 grid(
        static_cast<unsigned int>((b.columns() + block_width - 1) / block_width),
        static_cast<unsigned int>((a.rows() + block_width - 1) / block_width));

    gemm_naive_kernel<<<grid, block>>>(
        device_a.data(),
        device_b.data(),
        device_c.data(),
        a.rows(),
        a.columns(),
        b.columns());

    require_cuda_success(cudaGetLastError(), "launch naive GEMM kernel");
    require_cuda_success(cudaDeviceSynchronize(), "execute naive GEMM kernel");
    require_cuda_success(
        cudaMemcpy(
            c.data(),
            device_c.data(),
            c.size() * sizeof(float),
            cudaMemcpyDeviceToHost),
        "copy C to host");

    return c;
}

}  // namespace gpu_perf

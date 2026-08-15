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

class CudaEvent {
public:
    CudaEvent() {
        require_cuda_success(cudaEventCreate(&event_), "cudaEventCreate");
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    [[nodiscard]] cudaEvent_t get() const noexcept {
        return event_;
    }

private:
    cudaEvent_t event_{nullptr};
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

void launch_naive_gemm(
    const float* device_a,
    const float* device_b,
    float* device_c,
    std::size_t m,
    std::size_t k_dimension,
    std::size_t n) {
    constexpr unsigned int block_width = 16;
    const dim3 block(block_width, block_width);
    const dim3 grid(
        static_cast<unsigned int>((n + block_width - 1) / block_width),
        static_cast<unsigned int>((m + block_width - 1) / block_width));

    gemm_naive_kernel<<<grid, block>>>(
        device_a, device_b, device_c, m, k_dimension, n);
    require_cuda_success(cudaGetLastError(), "launch naive GEMM kernel");
}

void copy_inputs_to_device(
    const Matrix& a,
    const Matrix& b,
    DeviceBuffer& device_a,
    DeviceBuffer& device_b) {
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
}

void copy_output_to_host(Matrix& c, DeviceBuffer& device_c) {
    require_cuda_success(
        cudaMemcpy(
            c.data(),
            device_c.data(),
            c.size() * sizeof(float),
            cudaMemcpyDeviceToHost),
        "copy C to host");
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

    copy_inputs_to_device(a, b, device_a, device_b);
    launch_naive_gemm(
        device_a.data(),
        device_b.data(),
        device_c.data(),
        a.rows(),
        a.columns(),
        b.columns());
    require_cuda_success(cudaDeviceSynchronize(), "execute naive GEMM kernel");
    copy_output_to_host(c, device_c);

    return c;
}

CudaGemmBenchmarkResult benchmark_cuda_naive(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations) {
    if (a.columns() != b.rows()) {
        throw std::invalid_argument("GEMM inner dimensions must match");
    }
    if (measured_iterations == 0) {
        throw std::invalid_argument("measured iteration count must be positive");
    }

    CudaGemmBenchmarkResult result;
    result.output = Matrix(a.rows(), b.columns());
    result.warmup_iterations = warmup_iterations;
    result.measured_iterations = measured_iterations;

    if (result.output.size() == 0 || a.columns() == 0) {
        return result;
    }

    DeviceBuffer device_a(a.size());
    DeviceBuffer device_b(b.size());
    DeviceBuffer device_c(result.output.size());
    copy_inputs_to_device(a, b, device_a, device_b);

    for (std::size_t iteration = 0; iteration < warmup_iterations; ++iteration) {
        launch_naive_gemm(
            device_a.data(), device_b.data(), device_c.data(),
            a.rows(), a.columns(), b.columns());
    }
    require_cuda_success(cudaDeviceSynchronize(), "complete naive GEMM warmups");

    CudaEvent start;
    CudaEvent stop;
    require_cuda_success(cudaEventRecord(start.get()), "record benchmark start event");
    for (std::size_t iteration = 0; iteration < measured_iterations; ++iteration) {
        launch_naive_gemm(
            device_a.data(), device_b.data(), device_c.data(),
            a.rows(), a.columns(), b.columns());
    }
    require_cuda_success(cudaEventRecord(stop.get()), "record benchmark stop event");
    require_cuda_success(cudaEventSynchronize(stop.get()), "wait for benchmark stop event");
    require_cuda_success(
        cudaEventElapsedTime(
            &result.total_kernel_time_ms,
            start.get(),
            stop.get()),
        "calculate elapsed kernel time");

    result.average_kernel_time_ms =
        result.total_kernel_time_ms / static_cast<float>(measured_iterations);
    const long double operations =
        2.0L * static_cast<long double>(a.rows()) *
        static_cast<long double>(a.columns()) *
        static_cast<long double>(b.columns());
    result.achieved_gflops = static_cast<double>(
        operations /
        (static_cast<long double>(result.average_kernel_time_ms) * 1.0e6L));

    copy_output_to_host(result.output, device_c);
    return result;
}

}  // namespace gpu_perf

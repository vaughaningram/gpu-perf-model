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

constexpr unsigned int tile_width = 16;
constexpr unsigned int microtile_width = 2;
constexpr unsigned int microtile_output_width = tile_width * microtile_width;

__global__ void gemm_tiled_kernel(
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k_dimension,
    std::size_t n) {
    __shared__ float a_tile[tile_width][tile_width];
    __shared__ float b_tile[tile_width][tile_width];

    const std::size_t column = blockIdx.x * tile_width + threadIdx.x;
    const std::size_t row = blockIdx.y * tile_width + threadIdx.y;
    float accumulator = 0.0F;

    for (std::size_t tile_start = 0; tile_start < k_dimension;
         tile_start += tile_width) {
        const std::size_t a_column = tile_start + threadIdx.x;
        const std::size_t b_row = tile_start + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] =
            row < m && a_column < k_dimension
                ? a[row * k_dimension + a_column]
                : 0.0F;
        b_tile[threadIdx.y][threadIdx.x] =
            b_row < k_dimension && column < n
                ? b[b_row * n + column]
                : 0.0F;
        __syncthreads();

#pragma unroll
        for (unsigned int inner = 0; inner < tile_width; ++inner) {
            accumulator += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && column < n) {
        c[row * n + column] = accumulator;
    }
}

__global__ void gemm_microtile_2x2_kernel(
    const float* a,
    const float* b,
    float* c,
    std::size_t m,
    std::size_t k_dimension,
    std::size_t n) {
    __shared__ float a_tile[microtile_output_width][tile_width];
    __shared__ float b_tile[tile_width][microtile_output_width];

    const std::size_t row0 =
        blockIdx.y * microtile_output_width + threadIdx.y;
    const std::size_t row1 = row0 + tile_width;
    const std::size_t column0 =
        blockIdx.x * microtile_output_width + threadIdx.x;
    const std::size_t column1 = column0 + tile_width;

    float accumulator00 = 0.0F;
    float accumulator01 = 0.0F;
    float accumulator10 = 0.0F;
    float accumulator11 = 0.0F;

    for (std::size_t tile_start = 0; tile_start < k_dimension;
         tile_start += tile_width) {
        const std::size_t a_column = tile_start + threadIdx.x;
        const std::size_t b_row = tile_start + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] =
            row0 < m && a_column < k_dimension
                ? a[row0 * k_dimension + a_column]
                : 0.0F;
        a_tile[threadIdx.y + tile_width][threadIdx.x] =
            row1 < m && a_column < k_dimension
                ? a[row1 * k_dimension + a_column]
                : 0.0F;
        b_tile[threadIdx.y][threadIdx.x] =
            b_row < k_dimension && column0 < n
                ? b[b_row * n + column0]
                : 0.0F;
        b_tile[threadIdx.y][threadIdx.x + tile_width] =
            b_row < k_dimension && column1 < n
                ? b[b_row * n + column1]
                : 0.0F;
        __syncthreads();

#pragma unroll
        for (unsigned int inner = 0; inner < tile_width; ++inner) {
            const float a0 = a_tile[threadIdx.y][inner];
            const float a1 = a_tile[threadIdx.y + tile_width][inner];
            const float b0 = b_tile[inner][threadIdx.x];
            const float b1 = b_tile[inner][threadIdx.x + tile_width];
            accumulator00 += a0 * b0;
            accumulator01 += a0 * b1;
            accumulator10 += a1 * b0;
            accumulator11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row0 < m && column0 < n) {
        c[row0 * n + column0] = accumulator00;
    }
    if (row0 < m && column1 < n) {
        c[row0 * n + column1] = accumulator01;
    }
    if (row1 < m && column0 < n) {
        c[row1 * n + column0] = accumulator10;
    }
    if (row1 < m && column1 < n) {
        c[row1 * n + column1] = accumulator11;
    }
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

void launch_tiled_gemm(
    const float* device_a,
    const float* device_b,
    float* device_c,
    std::size_t m,
    std::size_t k_dimension,
    std::size_t n) {
    const dim3 block(tile_width, tile_width);
    const dim3 grid(
        static_cast<unsigned int>((n + tile_width - 1) / tile_width),
        static_cast<unsigned int>((m + tile_width - 1) / tile_width));

    gemm_tiled_kernel<<<grid, block>>>(
        device_a, device_b, device_c, m, k_dimension, n);
    require_cuda_success(cudaGetLastError(), "launch tiled GEMM kernel");
}

void launch_microtile_2x2_gemm(
    const float* device_a,
    const float* device_b,
    float* device_c,
    std::size_t m,
    std::size_t k_dimension,
    std::size_t n) {
    const dim3 block(tile_width, tile_width);
    const dim3 grid(
        static_cast<unsigned int>(
            (n + microtile_output_width - 1) / microtile_output_width),
        static_cast<unsigned int>(
            (m + microtile_output_width - 1) / microtile_output_width));

    gemm_microtile_2x2_kernel<<<grid, block>>>(
        device_a, device_b, device_c, m, k_dimension, n);
    require_cuda_success(cudaGetLastError(), "launch 2x2 microtile GEMM kernel");
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

using GemmLaunchFunction = void (*)(
    const float*,
    const float*,
    float*,
    std::size_t,
    std::size_t,
    std::size_t);

CudaGemmBenchmarkResult benchmark_cuda(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations,
    GemmLaunchFunction launch) {
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
        launch(
            device_a.data(), device_b.data(), device_c.data(),
            a.rows(), a.columns(), b.columns());
    }
    require_cuda_success(cudaDeviceSynchronize(), "complete GEMM warmups");

    CudaEvent start;
    CudaEvent stop;
    require_cuda_success(cudaEventRecord(start.get()), "record benchmark start event");
    for (std::size_t iteration = 0; iteration < measured_iterations; ++iteration) {
        launch(
            device_a.data(), device_b.data(), device_c.data(),
            a.rows(), a.columns(), b.columns());
    }
    require_cuda_success(cudaEventRecord(stop.get()), "record benchmark stop event");
    require_cuda_success(cudaEventSynchronize(stop.get()), "wait for benchmark stop event");
    require_cuda_success(
        cudaEventElapsedTime(&result.total_kernel_time_ms, start.get(), stop.get()),
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

Matrix gemm_cuda_tiled(const Matrix& a, const Matrix& b) {
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
    launch_tiled_gemm(
        device_a.data(),
        device_b.data(),
        device_c.data(),
        a.rows(),
        a.columns(),
        b.columns());
    require_cuda_success(cudaDeviceSynchronize(), "execute tiled GEMM kernel");
    copy_output_to_host(c, device_c);

    return c;
}

Matrix gemm_cuda_microtile_2x2(const Matrix& a, const Matrix& b) {
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
    launch_microtile_2x2_gemm(
        device_a.data(), device_b.data(), device_c.data(),
        a.rows(), a.columns(), b.columns());
    require_cuda_success(
        cudaDeviceSynchronize(), "execute 2x2 microtile GEMM kernel");
    copy_output_to_host(c, device_c);
    return c;
}

CudaDeviceMetadata current_cuda_device_metadata() {
    int device = 0;
    require_cuda_success(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp properties{};
    require_cuda_success(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");

    CudaDeviceMetadata metadata;
    metadata.name = properties.name;
    metadata.compute_capability_major = properties.major;
    metadata.compute_capability_minor = properties.minor;
    metadata.global_memory_bytes = properties.totalGlobalMem;
    metadata.multiprocessor_count = properties.multiProcessorCount;
    require_cuda_success(
        cudaDriverGetVersion(&metadata.driver_version),
        "cudaDriverGetVersion");
    require_cuda_success(
        cudaRuntimeGetVersion(&metadata.runtime_version),
        "cudaRuntimeGetVersion");
    return metadata;
}

CudaGemmBenchmarkResult benchmark_cuda_naive(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations) {
    return benchmark_cuda(
        a, b, warmup_iterations, measured_iterations, launch_naive_gemm);
}

CudaGemmBenchmarkResult benchmark_cuda_tiled(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations) {
    return benchmark_cuda(
        a, b, warmup_iterations, measured_iterations, launch_tiled_gemm);
}

CudaGemmBenchmarkResult benchmark_cuda_microtile_2x2(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations) {
    return benchmark_cuda(
        a, b, warmup_iterations, measured_iterations,
        launch_microtile_2x2_gemm);
}

}  // namespace gpu_perf

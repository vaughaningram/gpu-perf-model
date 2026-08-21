#pragma once

#include "gpu_perf/matrix.hpp"

#include <cstddef>
#include <string>

namespace gpu_perf {

// Computes C = A * B with one CUDA thread responsible for one output element.
// This baseline intentionally performs no explicit tiling or shared-memory reuse.
[[nodiscard]] Matrix gemm_cuda_naive(const Matrix& a, const Matrix& b);

// Computes C = A * B with a 16x16 cooperative shared-memory tile. Bounds
// guards support dimensions that are not multiples of the tile width.
[[nodiscard]] Matrix gemm_cuda_tiled(const Matrix& a, const Matrix& b);

// Computes a 32x32 output tile with a 16x16 thread block. Each thread owns a
// 2x2 output microtile and four independent accumulators.
[[nodiscard]] Matrix gemm_cuda_microtile_2x2(const Matrix& a, const Matrix& b);
[[nodiscard]] Matrix gemm_cuda_microtile_4x1(const Matrix& a, const Matrix& b);

struct CudaGemmBenchmarkResult {
    Matrix output;
    std::size_t warmup_iterations{0};
    std::size_t measured_iterations{0};
    float total_kernel_time_ms{0.0F};
    float average_kernel_time_ms{0.0F};
    double achieved_gflops{0.0};
};

struct CudaDeviceMetadata {
    std::string name;
    int compute_capability_major{0};
    int compute_capability_minor{0};
    std::size_t global_memory_bytes{0};
    int multiprocessor_count{0};
    int driver_version{0};
    int runtime_version{0};
};

[[nodiscard]] CudaDeviceMetadata current_cuda_device_metadata();

// Copies inputs once, runs untimed warmups, and uses CUDA events to time only
// repeated kernel executions. The result is copied to the host after timing.
[[nodiscard]] CudaGemmBenchmarkResult benchmark_cuda_naive(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations);

[[nodiscard]] CudaGemmBenchmarkResult benchmark_cuda_tiled(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations);

[[nodiscard]] CudaGemmBenchmarkResult benchmark_cuda_microtile_2x2(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations);

[[nodiscard]] CudaGemmBenchmarkResult benchmark_cuda_microtile_4x1(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations);

}  // namespace gpu_perf

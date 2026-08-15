#pragma once

#include "gpu_perf/matrix.hpp"

#include <cstddef>

namespace gpu_perf {

// Computes C = A * B with one CUDA thread responsible for one output element.
// This baseline intentionally performs no explicit tiling or shared-memory reuse.
[[nodiscard]] Matrix gemm_cuda_naive(const Matrix& a, const Matrix& b);

struct CudaGemmBenchmarkResult {
    Matrix output;
    std::size_t warmup_iterations{0};
    std::size_t measured_iterations{0};
    float total_kernel_time_ms{0.0F};
    float average_kernel_time_ms{0.0F};
    double achieved_gflops{0.0};
};

// Copies inputs once, runs untimed warmups, and uses CUDA events to time only
// repeated kernel executions. The result is copied to the host after timing.
[[nodiscard]] CudaGemmBenchmarkResult benchmark_cuda_naive(
    const Matrix& a,
    const Matrix& b,
    std::size_t warmup_iterations,
    std::size_t measured_iterations);

}  // namespace gpu_perf

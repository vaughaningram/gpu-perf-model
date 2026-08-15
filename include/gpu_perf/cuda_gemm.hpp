#pragma once

#include "gpu_perf/matrix.hpp"

namespace gpu_perf {

// Computes C = A * B with one CUDA thread responsible for one output element.
// This baseline intentionally performs no explicit tiling or shared-memory reuse.
[[nodiscard]] Matrix gemm_cuda_naive(const Matrix& a, const Matrix& b);

}  // namespace gpu_perf


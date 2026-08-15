#pragma once

#include "gpu_perf/matrix.hpp"

namespace gpu_perf {

// Computes C = A * B using a deliberately straightforward reference loop.
[[nodiscard]] Matrix gemm_cpu_reference(const Matrix& a, const Matrix& b);

}  // namespace gpu_perf


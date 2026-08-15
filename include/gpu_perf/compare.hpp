#pragma once

#include "gpu_perf/matrix.hpp"

#include <cstddef>

namespace gpu_perf {

struct ComparisonResult {
    bool passed{false};
    std::size_t compared_elements{0};
    std::size_t mismatched_elements{0};
    float maximum_absolute_error{0.0F};
    float maximum_relative_error{0.0F};
};

// An element passes when abs(actual - expected) <= atol + rtol * abs(expected).
[[nodiscard]] ComparisonResult compare_matrices(
    const Matrix& expected,
    const Matrix& actual,
    float absolute_tolerance,
    float relative_tolerance);

}  // namespace gpu_perf


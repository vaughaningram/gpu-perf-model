#include "gpu_perf/compare.hpp"
#include "gpu_perf/gemm.hpp"
#include "gpu_perf/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gpu_perf {
namespace {

std::size_t checked_element_count(std::size_t rows, std::size_t columns) {
    if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::length_error("matrix dimensions overflow size_t");
    }
    return rows * columns;
}

std::uint32_t xorshift32(std::uint32_t& state) noexcept {
    // Xorshift has a zero-state trap, so substitute a fixed nonzero state.
    if (state == 0) {
        state = 0x6D2B79F5U;
    }
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

float unit_float(std::uint32_t value) noexcept {
    constexpr float inverse_24_bit_range = 1.0F / 16777216.0F;
    return static_cast<float>(value >> 8U) * inverse_24_bit_range;
}

}  // namespace

Matrix::Matrix(std::size_t rows, std::size_t columns)
    : rows_(rows), columns_(columns), values_(checked_element_count(rows, columns), 0.0F) {}

Matrix::Matrix(std::size_t rows, std::size_t columns, std::vector<float> values)
    : rows_(rows), columns_(columns), values_(std::move(values)) {
    if (values_.size() != checked_element_count(rows, columns)) {
        throw std::invalid_argument("matrix value count does not match its dimensions");
    }
}

std::size_t Matrix::rows() const noexcept {
    return rows_;
}

std::size_t Matrix::columns() const noexcept {
    return columns_;
}

std::size_t Matrix::size() const noexcept {
    return values_.size();
}

float* Matrix::data() noexcept {
    return values_.data();
}

const float* Matrix::data() const noexcept {
    return values_.data();
}

float& Matrix::operator()(std::size_t row, std::size_t column) noexcept {
    return values_[row * columns_ + column];
}

const float& Matrix::operator()(std::size_t row, std::size_t column) const noexcept {
    return values_[row * columns_ + column];
}

Matrix make_deterministic_matrix(
    std::size_t rows,
    std::size_t columns,
    std::uint32_t seed) {
    Matrix matrix(rows, columns);
    auto state = seed;

    for (std::size_t index = 0; index < matrix.size(); ++index) {
        matrix.data()[index] = 2.0F * unit_float(xorshift32(state)) - 1.0F;
    }
    return matrix;
}

Matrix gemm_cpu_reference(const Matrix& a, const Matrix& b) {
    if (a.columns() != b.rows()) {
        throw std::invalid_argument("GEMM inner dimensions must match");
    }

    Matrix c(a.rows(), b.columns());
    for (std::size_t row = 0; row < a.rows(); ++row) {
        for (std::size_t column = 0; column < b.columns(); ++column) {
            float accumulator = 0.0F;
            for (std::size_t k = 0; k < a.columns(); ++k) {
                accumulator += a(row, k) * b(k, column);
            }
            c(row, column) = accumulator;
        }
    }
    return c;
}

ComparisonResult compare_matrices(
    const Matrix& expected,
    const Matrix& actual,
    float absolute_tolerance,
    float relative_tolerance) {
    if (expected.rows() != actual.rows() || expected.columns() != actual.columns()) {
        throw std::invalid_argument("matrix dimensions must match for comparison");
    }
    if (absolute_tolerance < 0.0F || relative_tolerance < 0.0F) {
        throw std::invalid_argument("comparison tolerances must be nonnegative");
    }

    ComparisonResult result;
    result.compared_elements = expected.size();

    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float expected_value = expected.data()[index];
        const float actual_value = actual.data()[index];
        const float absolute_error = std::abs(actual_value - expected_value);
        const float denominator = std::max(std::abs(expected_value), std::numeric_limits<float>::min());
        const float relative_error = absolute_error / denominator;

        result.maximum_absolute_error = std::max(result.maximum_absolute_error, absolute_error);
        result.maximum_relative_error = std::max(result.maximum_relative_error, relative_error);

        const bool finite_values = std::isfinite(expected_value) && std::isfinite(actual_value);
        const float allowed_error = absolute_tolerance + relative_tolerance * std::abs(expected_value);
        if (!finite_values || absolute_error > allowed_error) {
            ++result.mismatched_elements;
        }
    }

    result.passed = result.mismatched_elements == 0;
    return result;
}

}  // namespace gpu_perf


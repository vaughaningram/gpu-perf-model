#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_perf {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t columns);
    Matrix(std::size_t rows, std::size_t columns, std::vector<float> values);

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t columns() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    [[nodiscard]] float* data() noexcept;
    [[nodiscard]] const float* data() const noexcept;

    float& operator()(std::size_t row, std::size_t column) noexcept;
    const float& operator()(std::size_t row, std::size_t column) const noexcept;

private:
    std::size_t rows_{0};
    std::size_t columns_{0};
    std::vector<float> values_;
};

// Generates the same values on every supported platform for a fixed seed.
[[nodiscard]] Matrix make_deterministic_matrix(
    std::size_t rows,
    std::size_t columns,
    std::uint32_t seed);

}  // namespace gpu_perf


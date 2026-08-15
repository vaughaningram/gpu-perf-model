#include "gpu_perf/compare.hpp"
#include "gpu_perf/cuda_gemm.hpp"
#include "gpu_perf/gemm.hpp"
#include "gpu_perf/matrix.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_gpu_matches_cpu(
    std::size_t m,
    std::size_t k,
    std::size_t n,
    std::uint32_t seed_a,
    std::uint32_t seed_b) {
    const auto a = gpu_perf::make_deterministic_matrix(m, k, seed_a);
    const auto b = gpu_perf::make_deterministic_matrix(k, n, seed_b);
    const auto expected = gpu_perf::gemm_cpu_reference(a, b);
    const auto actual = gpu_perf::gemm_cuda_naive(a, b);
    const auto comparison = gpu_perf::compare_matrices(expected, actual, 1.0e-4F, 1.0e-4F);

    require(
        comparison.passed,
        "naive CUDA GEMM disagreed with CPU reference: mismatches=" +
            std::to_string(comparison.mismatched_elements) +
            ", max_abs_error=" + std::to_string(comparison.maximum_absolute_error) +
            ", max_rel_error=" + std::to_string(comparison.maximum_relative_error));
}

void test_known_product() {
    const gpu_perf::Matrix a(2, 3, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const gpu_perf::Matrix b(3, 2, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
    const gpu_perf::Matrix expected(2, 2, {58.0F, 64.0F, 139.0F, 154.0F});
    const auto actual = gpu_perf::gemm_cuda_naive(a, b);

    require(
        gpu_perf::compare_matrices(expected, actual, 0.0F, 0.0F).passed,
        "naive CUDA GEMM failed the known product");
}

void test_invalid_dimensions() {
    bool threw = false;
    try {
        const gpu_perf::Matrix a(2, 3);
        const gpu_perf::Matrix b(4, 2);
        static_cast<void>(gpu_perf::gemm_cuda_naive(a, b));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "naive CUDA GEMM accepted incompatible dimensions");
}

}  // namespace

int main() {
    try {
        test_known_product();
        require_gpu_matches_cpu(16, 16, 16, 1U, 2U);
        require_gpu_matches_cpu(19, 13, 7, 3U, 4U);
        require_gpu_matches_cpu(1, 37, 29, 5U, 6U);
        test_invalid_dimensions();
    } catch (const std::exception& error) {
        std::cerr << "Naive CUDA correctness test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All naive CUDA correctness tests passed.\n";
    return EXIT_SUCCESS;
}


#include "gpu_perf/compare.hpp"
#include "gpu_perf/gemm.hpp"
#include "gpu_perf/matrix.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_known_rectangular_product() {
    const gpu_perf::Matrix a(2, 3, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const gpu_perf::Matrix b(3, 2, {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
    const gpu_perf::Matrix expected(2, 2, {58.0F, 64.0F, 139.0F, 154.0F});

    const auto actual = gpu_perf::gemm_cpu_reference(a, b);
    const auto comparison = gpu_perf::compare_matrices(expected, actual, 0.0F, 0.0F);

    require(comparison.passed, "known rectangular GEMM result was incorrect");
    require(comparison.compared_elements == 4, "comparison reported the wrong element count");
}

void test_deterministic_generation() {
    const auto first = gpu_perf::make_deterministic_matrix(3, 5, 12345U);
    const auto second = gpu_perf::make_deterministic_matrix(3, 5, 12345U);
    const auto different = gpu_perf::make_deterministic_matrix(3, 5, 54321U);

    require(
        gpu_perf::compare_matrices(first, second, 0.0F, 0.0F).passed,
        "equal seeds did not produce equal matrices");
    require(
        !gpu_perf::compare_matrices(first, different, 0.0F, 0.0F).passed,
        "different seeds unexpectedly produced equal matrices");
}

void test_tolerance_behavior() {
    const gpu_perf::Matrix expected(1, 2, {100.0F, 0.0F});
    const gpu_perf::Matrix close(1, 2, {100.05F, 0.0005F});
    const gpu_perf::Matrix far(1, 2, {101.0F, 0.1F});

    require(
        gpu_perf::compare_matrices(expected, close, 0.001F, 0.001F).passed,
        "comparison rejected values within tolerance");
    require(
        !gpu_perf::compare_matrices(expected, far, 0.001F, 0.001F).passed,
        "comparison accepted values outside tolerance");
}

void test_invalid_dimensions() {
    bool threw = false;
    try {
        const gpu_perf::Matrix a(2, 3);
        const gpu_perf::Matrix b(4, 2);
        static_cast<void>(gpu_perf::gemm_cpu_reference(a, b));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "GEMM accepted incompatible matrix dimensions");
}

}  // namespace

int main() {
    try {
        test_known_rectangular_product();
        test_deterministic_generation();
        test_tolerance_behavior();
        test_invalid_dimensions();
    } catch (const std::exception& error) {
        std::cerr << "CPU reference test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All CPU reference tests passed.\n";
    return EXIT_SUCCESS;
}


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

void require_tiled_gpu_matches_cpu(
    std::size_t m,
    std::size_t k,
    std::size_t n,
    std::uint32_t seed_a,
    std::uint32_t seed_b) {
    const auto a = gpu_perf::make_deterministic_matrix(m, k, seed_a);
    const auto b = gpu_perf::make_deterministic_matrix(k, n, seed_b);
    const auto expected = gpu_perf::gemm_cpu_reference(a, b);
    const auto actual = gpu_perf::gemm_cuda_tiled(a, b);
    const auto comparison = gpu_perf::compare_matrices(expected, actual, 1.0e-4F, 1.0e-4F);

    require(
        comparison.passed,
        "tiled CUDA GEMM disagreed with CPU reference: mismatches=" +
            std::to_string(comparison.mismatched_elements) +
            ", max_abs_error=" + std::to_string(comparison.maximum_absolute_error) +
            ", max_rel_error=" + std::to_string(comparison.maximum_relative_error));
}

void require_microtile_gpu_matches_cpu(
    std::size_t m,
    std::size_t k,
    std::size_t n,
    std::uint32_t seed_a,
    std::uint32_t seed_b) {
    const auto a = gpu_perf::make_deterministic_matrix(m, k, seed_a);
    const auto b = gpu_perf::make_deterministic_matrix(k, n, seed_b);
    const auto expected = gpu_perf::gemm_cpu_reference(a, b);
    const auto actual = gpu_perf::gemm_cuda_microtile_2x2(a, b);
    const auto comparison = gpu_perf::compare_matrices(expected, actual, 1.0e-4F, 1.0e-4F);

    require(
        comparison.passed,
        "microtile CUDA GEMM disagreed with CPU reference: mismatches=" +
            std::to_string(comparison.mismatched_elements) +
            ", max_abs_error=" + std::to_string(comparison.maximum_absolute_error) +
            ", max_rel_error=" + std::to_string(comparison.maximum_relative_error));
}

void require_microtile_4x1_gpu_matches_cpu(
    std::size_t m,
    std::size_t k,
    std::size_t n,
    std::uint32_t seed_a,
    std::uint32_t seed_b) {
    const auto a = gpu_perf::make_deterministic_matrix(m, k, seed_a);
    const auto b = gpu_perf::make_deterministic_matrix(k, n, seed_b);
    const auto expected = gpu_perf::gemm_cpu_reference(a, b);
    const auto actual = gpu_perf::gemm_cuda_microtile_4x1(a, b);
    const auto comparison = gpu_perf::compare_matrices(expected, actual, 1.0e-4F, 1.0e-4F);
    require(
        comparison.passed,
        "4x1 microtile CUDA GEMM disagreed with CPU reference: mismatches=" +
            std::to_string(comparison.mismatched_elements));
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

    threw = false;
    try {
        const gpu_perf::Matrix a(2, 3);
        const gpu_perf::Matrix b(4, 2);
        static_cast<void>(gpu_perf::gemm_cuda_tiled(a, b));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "tiled CUDA GEMM accepted incompatible dimensions");
}

void test_event_timed_benchmark() {
    const auto a = gpu_perf::make_deterministic_matrix(32, 17, 7U);
    const auto b = gpu_perf::make_deterministic_matrix(17, 24, 8U);
    const auto expected = gpu_perf::gemm_cpu_reference(a, b);
    const auto benchmark = gpu_perf::benchmark_cuda_naive(a, b, 2, 3);

    require(benchmark.warmup_iterations == 2, "benchmark lost the warmup count");
    require(benchmark.measured_iterations == 3, "benchmark lost the measured count");
    require(benchmark.total_kernel_time_ms > 0.0F, "benchmark reported no total time");
    require(benchmark.average_kernel_time_ms > 0.0F, "benchmark reported no average time");
    require(benchmark.achieved_gflops > 0.0, "benchmark reported no throughput");
    require(
        gpu_perf::compare_matrices(expected, benchmark.output, 1.0e-4F, 1.0e-4F).passed,
        "timed benchmark output disagreed with CPU reference");
}

void test_tiled_event_timed_benchmark() {
    const auto a = gpu_perf::make_deterministic_matrix(32, 17, 7U);
    const auto b = gpu_perf::make_deterministic_matrix(17, 24, 8U);
    const auto expected = gpu_perf::gemm_cpu_reference(a, b);
    const auto benchmark = gpu_perf::benchmark_cuda_tiled(a, b, 2, 3);

    require(benchmark.warmup_iterations == 2, "tiled benchmark lost the warmup count");
    require(benchmark.measured_iterations == 3, "tiled benchmark lost the measured count");
    require(benchmark.total_kernel_time_ms > 0.0F, "tiled benchmark reported no time");
    require(benchmark.achieved_gflops > 0.0, "tiled benchmark reported no throughput");
    require(
        gpu_perf::compare_matrices(expected, benchmark.output, 1.0e-4F, 1.0e-4F).passed,
        "timed tiled benchmark output disagreed with CPU reference");
}

void test_device_metadata() {
    const auto metadata = gpu_perf::current_cuda_device_metadata();
    require(!metadata.name.empty(), "CUDA device metadata has no name");
    require(metadata.compute_capability_major > 0, "invalid CUDA compute capability");
    require(metadata.global_memory_bytes > 0, "CUDA device metadata has no memory");
    require(metadata.multiprocessor_count > 0, "CUDA device metadata has no SMs");
    require(metadata.driver_version > 0, "CUDA driver version is unavailable");
    require(metadata.runtime_version > 0, "CUDA runtime version is unavailable");
}

}  // namespace

int main() {
    try {
        test_known_product();
        require_gpu_matches_cpu(16, 16, 16, 1U, 2U);
        require_gpu_matches_cpu(19, 13, 7, 3U, 4U);
        require_gpu_matches_cpu(1, 37, 29, 5U, 6U);
        require_tiled_gpu_matches_cpu(16, 16, 16, 11U, 12U);
        require_tiled_gpu_matches_cpu(19, 13, 7, 13U, 14U);
        require_tiled_gpu_matches_cpu(1, 37, 29, 15U, 16U);
        require_microtile_gpu_matches_cpu(32, 32, 32, 21U, 22U);
        require_microtile_gpu_matches_cpu(35, 13, 19, 23U, 24U);
        require_microtile_gpu_matches_cpu(1, 37, 29, 25U, 26U);
        require_microtile_4x1_gpu_matches_cpu(64, 32, 16, 31U, 32U);
        require_microtile_4x1_gpu_matches_cpu(67, 13, 19, 33U, 34U);
        test_invalid_dimensions();
        test_event_timed_benchmark();
        test_tiled_event_timed_benchmark();
        test_device_metadata();
    } catch (const std::exception& error) {
        std::cerr << "Naive CUDA correctness test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "All naive CUDA correctness tests passed.\n";
    return EXIT_SUCCESS;
}

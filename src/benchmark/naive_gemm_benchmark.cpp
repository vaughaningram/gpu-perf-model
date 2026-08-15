#include "gpu_perf/compare.hpp"
#include "gpu_perf/cuda_gemm.hpp"
#include "gpu_perf/gemm.hpp"
#include "gpu_perf/matrix.hpp"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::size_t parse_positive_size(const char* text, const char* name) {
    if (text[0] == '\0' || text[0] == '-') {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (text[consumed] != '\0' || value == 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<std::size_t>(value);
}

std::size_t parse_nonnegative_size(const char* text, const char* name) {
    if (text[0] == '\0' || text[0] == '-') {
        throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
    }
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (text[consumed] != '\0') {
        throw std::invalid_argument(std::string(name) + " must be a nonnegative integer");
    }
    return static_cast<std::size_t>(value);
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " [M K N [warmups measured_iterations]]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 4 && argc != 6) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        std::size_t m = 512;
        std::size_t k = 512;
        std::size_t n = 512;
        std::size_t warmups = 5;
        std::size_t measured_iterations = 20;

        if (argc >= 4) {
            m = parse_positive_size(argv[1], "M");
            k = parse_positive_size(argv[2], "K");
            n = parse_positive_size(argv[3], "N");
        }
        if (argc == 6) {
            warmups = parse_nonnegative_size(argv[4], "warmups");
            measured_iterations = parse_positive_size(argv[5], "measured_iterations");
        }

        const auto a = gpu_perf::make_deterministic_matrix(m, k, 0xA341316CU);
        const auto b = gpu_perf::make_deterministic_matrix(k, n, 0xC8013EA4U);

        const auto benchmark =
            gpu_perf::benchmark_cuda_naive(a, b, warmups, measured_iterations);

        // Correctness work occurs after the timed GPU region.
        const auto expected = gpu_perf::gemm_cpu_reference(a, b);
        const auto comparison =
            gpu_perf::compare_matrices(expected, benchmark.output, 1.0e-4F, 1.0e-4F);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "kernel=naive\n";
        std::cout << "m=" << m << '\n';
        std::cout << "k=" << k << '\n';
        std::cout << "n=" << n << '\n';
        std::cout << "warmup_iterations=" << benchmark.warmup_iterations << '\n';
        std::cout << "measured_iterations=" << benchmark.measured_iterations << '\n';
        std::cout << "total_kernel_time_ms=" << benchmark.total_kernel_time_ms << '\n';
        std::cout << "average_kernel_time_ms=" << benchmark.average_kernel_time_ms << '\n';
        std::cout << "achieved_gflops=" << benchmark.achieved_gflops << '\n';
        std::cout << "correctness=" << (comparison.passed ? "PASS" : "FAIL") << '\n';
        std::cout << "mismatched_elements=" << comparison.mismatched_elements << '\n';
        std::cout << "maximum_absolute_error=" << comparison.maximum_absolute_error << '\n';
        std::cout << "maximum_relative_error=" << comparison.maximum_relative_error << '\n';

        return comparison.passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark error: " << error.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}

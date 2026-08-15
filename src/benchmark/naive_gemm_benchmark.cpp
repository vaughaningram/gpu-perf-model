#include "gpu_perf/compare.hpp"
#include "gpu_perf/cuda_gemm.hpp"
#include "gpu_perf/gemm.hpp"
#include "gpu_perf/matrix.hpp"

#include <cstddef>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

#ifndef GPU_PERF_GIT_COMMIT
#define GPU_PERF_GIT_COMMIT "unknown"
#endif

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
              << " [M K N [warmups measured_iterations]] [--csv]\n";
}

std::string version_string(int encoded_version) {
    const int major = encoded_version / 1000;
    const int minor = (encoded_version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string current_utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::string environment_value(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? fallback : value;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (const char character : value) {
        if (character == '\"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }
    escaped += '\"';
    return escaped;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool csv_output = false;
        int value_argument_count = argc;
        if (argc > 1 && std::string(argv[argc - 1]) == "--csv") {
            csv_output = true;
            --value_argument_count;
        }

        if (value_argument_count != 1 &&
            value_argument_count != 4 &&
            value_argument_count != 6) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        std::size_t m = 512;
        std::size_t k = 512;
        std::size_t n = 512;
        std::size_t warmups = 5;
        std::size_t measured_iterations = 20;

        if (value_argument_count >= 4) {
            m = parse_positive_size(argv[1], "M");
            k = parse_positive_size(argv[2], "K");
            n = parse_positive_size(argv[3], "N");
        }
        if (value_argument_count == 6) {
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

        const auto device = gpu_perf::current_cuda_device_metadata();
        std::string hostname = environment_value("SLURMD_NODENAME", "");
        if (hostname.empty()) {
            hostname = environment_value("HOSTNAME", "unknown");
        }
        const std::string compute_capability =
            std::to_string(device.compute_capability_major) + "." +
            std::to_string(device.compute_capability_minor);

        std::cout << std::fixed << std::setprecision(6);
        if (csv_output) {
            std::cout
                << "schema_version,timestamp_utc,git_commit,hostname,gpu_name,"
                   "compute_capability,driver_version,cuda_runtime_version,kernel,"
                   "m,k,n,warmup_iterations,measured_iterations,total_kernel_time_ms,"
                   "average_kernel_time_ms,achieved_gflops,correctness,"
                   "mismatched_elements,maximum_absolute_error,maximum_relative_error\n";
            std::cout
                << "1," << current_utc_timestamp() << ','
                << GPU_PERF_GIT_COMMIT << ','
                << csv_escape(hostname) << ','
                << csv_escape(device.name) << ','
                << compute_capability << ','
                << version_string(device.driver_version) << ','
                << version_string(device.runtime_version) << ','
                << "naive," << m << ',' << k << ',' << n << ','
                << benchmark.warmup_iterations << ','
                << benchmark.measured_iterations << ','
                << benchmark.total_kernel_time_ms << ','
                << benchmark.average_kernel_time_ms << ','
                << benchmark.achieved_gflops << ','
                << (comparison.passed ? "PASS" : "FAIL") << ','
                << comparison.mismatched_elements << ','
                << comparison.maximum_absolute_error << ','
                << comparison.maximum_relative_error << '\n';
        } else {
            std::cout << "timestamp_utc=" << current_utc_timestamp() << '\n';
            std::cout << "git_commit=" << GPU_PERF_GIT_COMMIT << '\n';
            std::cout << "hostname=" << hostname << '\n';
            std::cout << "gpu_name=" << device.name << '\n';
            std::cout << "compute_capability=" << compute_capability << '\n';
            std::cout << "driver_version=" << version_string(device.driver_version) << '\n';
            std::cout << "cuda_runtime_version=" << version_string(device.runtime_version) << '\n';
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
        }

        return comparison.passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark error: " << error.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}

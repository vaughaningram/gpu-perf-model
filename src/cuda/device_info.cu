#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace {

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

}  // namespace

int main() {
    int device_count = 0;
    if (!check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
        return EXIT_FAILURE;
    }
    if (device_count < 1) {
        std::cerr << "No CUDA devices are visible to this process.\n";
        return EXIT_FAILURE;
    }

    int device = 0;
    if (!check_cuda(cudaGetDevice(&device), "cudaGetDevice")) {
        return EXIT_FAILURE;
    }

    cudaDeviceProp properties{};
    if (!check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties")) {
        return EXIT_FAILURE;
    }

    constexpr double bytes_per_gibibyte = 1024.0 * 1024.0 * 1024.0;

    std::cout << "visible_devices=" << device_count << '\n';
    std::cout << "selected_device=" << device << '\n';
    std::cout << "name=" << properties.name << '\n';
    std::cout << "compute_capability=" << properties.major << '.' << properties.minor << '\n';
    std::cout << "global_memory_gib="
              << static_cast<double>(properties.totalGlobalMem) / bytes_per_gibibyte << '\n';
    std::cout << "multiprocessor_count=" << properties.multiProcessorCount << '\n';
    std::cout << "warp_size=" << properties.warpSize << '\n';
    std::cout << "max_threads_per_block=" << properties.maxThreadsPerBlock << '\n';
    std::cout << "max_threads_per_multiprocessor=" << properties.maxThreadsPerMultiProcessor << '\n';
    std::cout << "shared_memory_per_block_bytes=" << properties.sharedMemPerBlock << '\n';
    std::cout << "registers_per_block=" << properties.regsPerBlock << '\n';

    return EXIT_SUCCESS;
}


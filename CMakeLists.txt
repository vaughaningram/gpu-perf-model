cmake_minimum_required(VERSION 3.24)

project(gpu_perf_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GPU_PERF_ENABLE_CUDA "Build CUDA components when a CUDA compiler is available" ON)
option(GPU_PERF_BUILD_TESTS "Build project tests" ON)

add_library(gpu_perf_reference
    src/reference/gemm_cpu.cpp
)

target_include_directories(gpu_perf_reference
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(gpu_perf_reference PRIVATE /W4 /permissive-)
else()
    target_compile_options(gpu_perf_reference PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GPU_PERF_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)
        set(GPU_PERF_CUDA_ARCHITECTURES "80" CACHE STRING
            "CUDA architectures to compile for (80 targets the canonical A100)")
        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")

        add_executable(gpu_device_info
            src/cuda/device_info.cu
        )
        set_target_properties(gpu_device_info PROPERTIES
            CUDA_ARCHITECTURES "${GPU_PERF_CUDA_ARCHITECTURES}"
        )

        add_library(gpu_perf_cuda
            src/cuda/gemm_naive.cu
        )
        target_include_directories(gpu_perf_cuda
            PUBLIC
                ${PROJECT_SOURCE_DIR}/include
        )
        target_link_libraries(gpu_perf_cuda PUBLIC gpu_perf_reference)
        set_target_properties(gpu_perf_cuda PROPERTIES
            CUDA_ARCHITECTURES "${GPU_PERF_CUDA_ARCHITECTURES}"
        )
    else()
        message(STATUS "CUDA compiler not found; building CPU-only components")
    endif()
endif()

if(GPU_PERF_BUILD_TESTS)
    enable_testing()

    add_executable(cpu_reference_tests
        tests/test_cpu_reference.cpp
    )

    target_link_libraries(cpu_reference_tests PRIVATE gpu_perf_reference)

    if(MSVC)
        target_compile_options(cpu_reference_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(cpu_reference_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME cpu_reference_tests COMMAND cpu_reference_tests)

    if(CMAKE_CUDA_COMPILER)
        add_executable(cuda_naive_correctness_tests
            tests/test_naive_cuda.cu
        )
        target_link_libraries(cuda_naive_correctness_tests PRIVATE gpu_perf_cuda)
        set_target_properties(cuda_naive_correctness_tests PROPERTIES
            CUDA_ARCHITECTURES "${GPU_PERF_CUDA_ARCHITECTURES}"
        )
        add_test(NAME cuda_naive_correctness_tests COMMAND cuda_naive_correctness_tests)
    endif()
endif()

#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="${repo_root}/build/naive_gemm_benchmark"
expected_gpu="${GPU_PERF_EXPECTED_GPU:-NVIDIA A100 80GB PCIe}"
size="${GPU_PERF_PROFILE_SIZE:-2048}"
kernel="${GPU_PERF_PROFILE_KERNEL:-naive}"
output="${GPU_PERF_PROFILE_OUTPUT:-${repo_root}/results/${kernel}_a100_80gb_${size}.ncu-rep}"
ncu_command="${GPU_PERF_NCU:-ncu}"

case "${kernel}" in
    naive)
        kernel_regex="gemm_naive_kernel"
        benchmark_options=()
        ;;
    tiled16)
        kernel_regex="gemm_tiled_kernel"
        benchmark_options=(--tiled)
        ;;
    microtile2x2)
        kernel_regex="gemm_microtile_2x2_kernel"
        benchmark_options=(--microtile)
        ;;
    *)
        echo "Unsupported profile kernel: ${kernel}" >&2
        echo "Expected naive, tiled16, or microtile2x2." >&2
        exit 1
        ;;
esac

if [[ ! -x "${benchmark}" ]]; then
    echo "Benchmark executable not found: ${benchmark}" >&2
    exit 1
fi

for command in nvidia-smi "${ncu_command}"; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Required command is unavailable: ${command}" >&2
        exit 1
    fi
done

actual_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${actual_gpu}" != "${expected_gpu}" ]]; then
    echo "GPU mismatch: expected '${expected_gpu}', received '${actual_gpu}'." >&2
    exit 1
fi

"${ncu_command}" \
    --kernel-name "regex:${kernel_regex}" \
    --launch-skip 5 \
    --launch-count 1 \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --section InstructionStats \
    --export "${output}" \
    --force-overwrite \
    "${benchmark}" "${size}" "${size}" "${size}" 5 1 "${benchmark_options[@]}"

echo "Nsight Compute report: ${output}" >&2

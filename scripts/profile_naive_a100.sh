#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="${repo_root}/build/naive_gemm_benchmark"
expected_gpu="${GPU_PERF_EXPECTED_GPU:-NVIDIA A100 80GB PCIe}"
size="${GPU_PERF_PROFILE_SIZE:-2048}"
output="${GPU_PERF_PROFILE_OUTPUT:-${repo_root}/results/naive_a100_80gb_${size}.ncu-rep}"
ncu_command="${GPU_PERF_NCU:-ncu}"

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
    --kernel-name regex:gemm_naive_kernel \
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
    "${benchmark}" "${size}" "${size}" "${size}" 5 1

echo "Nsight Compute report: ${output}" >&2

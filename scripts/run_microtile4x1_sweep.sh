#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="${repo_root}/build/naive_gemm_benchmark"
expected_gpu="${GPU_PERF_EXPECTED_GPU:-NVIDIA A100 80GB PCIe}"
sizes=(128 256 512 1024 2048)

if [[ ! -x "${benchmark}" ]]; then
    echo "Benchmark executable not found: ${benchmark}" >&2
    exit 1
fi
actual_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${actual_gpu}" != "${expected_gpu}" ]]; then
    echo "GPU mismatch: expected '${expected_gpu}', received '${actual_gpu}'." >&2
    exit 1
fi

first_result=true
for size in "${sizes[@]}"; do
    echo "Running microtile4x1 GEMM ${size}x${size}x${size}..." >&2
    result="$("${benchmark}" "${size}" "${size}" "${size}" \
        5 20 --microtile4x1 --csv)"
    if [[ "${first_result}" == true ]]; then
        printf '%s\n' "${result}"
        first_result=false
    else
        printf '%s\n' "${result}" | tail -n 1
    fi
done

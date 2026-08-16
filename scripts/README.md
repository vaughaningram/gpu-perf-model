# Scripts

## Naive GEMM size sweep

`run_naive_sweep.sh` runs square naive GEMMs at sizes 128, 256, 512, 1024,
and 2048. It requires the canonical NVIDIA A100 80 GB PCIe by default and exits
if Slurm assigned different hardware.

From the repository root on an allocated GPU compute node:

```bash
./scripts/run_naive_sweep.sh | tee results/naive_a100_80gb_baseline.csv
```

Progress messages go to standard error, while standard output contains one CSV
header followed by one row per size. Override iteration counts only for an
explicit experiment:

```bash
GPU_PERF_WARMUPS=10 GPU_PERF_MEASURED_ITERATIONS=50 \
    ./scripts/run_naive_sweep.sh
```

Changing `GPU_PERF_EXPECTED_GPU` is permitted for labeled bring-up experiments,
but those results must not be mixed with the primary A100 80 GB dataset.

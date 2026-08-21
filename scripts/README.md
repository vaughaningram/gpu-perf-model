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

## Tiled GEMM size sweep

After the M2 prediction has been frozen, run the controlled 16x16 tiled sweep:

```bash
bash scripts/run_tiled_sweep.sh | tee results/tiled16_a100_80gb_baseline.csv
```

The hardware guard, sizes, warmups, iterations, and CSV schema match the naive
sweep. The benchmark's default remains naive; `--tiled` explicitly selects the
shared-memory implementation.

## Naive GEMM focused profile

After building, collect the M1 Nsight Compute report on the canonical GPU:

```bash
./scripts/profile_naive_a100.sh
```

The default problem is 2048x2048x2048. Override it only for an explicitly
labeled comparison, for example `GPU_PERF_PROFILE_SIZE=512`.

If the `ncu` on `PATH` is incompatible with the installed driver, select a
compatible executable explicitly:

```bash
GPU_PERF_NCU=/path/to/ncu ./scripts/profile_naive_a100.sh
```

Select the M2 kernel with `GPU_PERF_PROFILE_KERNEL=tiled16`; the default remains
`naive` so existing M1 commands retain their meaning.

## Microtile tuning sweep

Run the frozen M4 2x2 output-microtile experiment with:

```bash
bash scripts/run_microtile_sweep.sh | \
    tee results/microtile2x2_a100_80gb_baseline.csv
```

For its focused profile, set `GPU_PERF_PROFILE_KERNEL=microtile2x2`.

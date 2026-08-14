# GPU Performance Modeling

An experimental project for predicting, measuring, and explaining GPU workload
performance. The primary workload is FP32 general matrix multiplication (GEMM),
with 2D convolution planned as a secondary study after the GEMM analysis is
complete.

## Central question

Can a workload's computation, data movement, and hardware-resource requirements
predict its performance—and can measurements explain where a simplified model
is wrong?

The project follows one repeated loop:

> workload → analytical model → bottleneck prediction → hardware measurement →
> discrepancy analysis → model-guided optimization

Prediction comes before profiling. The objective is architectural reasoning and
model-to-hardware correlation, not simply producing fast CUDA kernels.

Data-movement estimates explicitly distinguish the algorithmic minimum from
traffic implied by an actual kernel. Comparisons with profiler measurements will
name the relevant cache or DRAM level. Occupancy is treated as a constraint and
diagnostic—not as a target to maximize.

## Current milestone: M0 — Experimental foundation

The first milestone establishes a trustworthy baseline:

- [ ] Confirm the CUDA development and execution environment
- [ ] Implement a simple FP32 GEMM kernel
- [ ] Add a CPU correctness reference
- [ ] Add CUDA-event timing, warmups, and repeated trials
- [ ] Calculate achieved GFLOP/s
- [ ] Emit structured benchmark results
- [ ] Sweep several matrix sizes
- [ ] Document the naive kernel's memory behavior

Later milestones will add the analytical roofline model, shared-memory tiling,
resource-aware modeling, controlled tuning, accelerator sensitivity analysis,
2D convolution, and a final prediction-versus-measurement study.

## Repository layout

```text
src/          CUDA kernels, benchmark harness, and CPU reference
include/      Shared C++/CUDA headers
model/        Python analytical performance model
scripts/      Experiment and plotting tools
experiments/  Reproducible experiment definitions and notes
results/      Curated results and plots
docs/         Methodology and technical analysis
tests/        Automated correctness tests
```

The canonical local-development and Tufts Pax execution workflow is documented
in [`docs/environment.md`](docs/environment.md).

## Scope

This is intentionally not a cycle-accurate GPU simulator, CUDA compiler, RTL
accelerator, cuBLAS competitor, or blind autotuning project. Its sophistication
should come from defensible assumptions, controlled experiments, and careful
analysis rather than code volume.

## Build status

The project is currently being scaffolded. Build and run instructions will be
added as the M0 implementation establishes the supported CUDA environment.

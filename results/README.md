# Results

This directory contains curated, reproducible CSV results and plots. Large raw
profiler captures are intentionally excluded from version control.

`naive_a100_80gb_baseline.csv` is the first controlled M0 size sweep. Its rows
identify the exact measurement-code commit and A100 environment used.

`m1_a100_80gb_pcie_roofline_predictions.csv` freezes the analytical M1
predictions before comparison with that M0 sweep.

`m1_naive_roofline_comparison.csv` compares the frozen prediction endpoints
with M0 and records model residuals used to choose focused profiler metrics.

`m1_naive_a100_80gb_profile_summary.csv` records the focused 512 and 2048
profiler evidence used for the final M1 bottleneck explanation and size
contrast. The raw `.ncu-rep` files are excluded from Git because profiler
reports are tool-specific binary artifacts.

`m2_a100_80gb_tiled16_predictions.csv` freezes the first shared-memory tiled
GEMM predictions before its CUDA implementation or measurement.

`tiled16_a100_80gb_baseline.csv` is the controlled M2 size sweep, and
`m2_tiled16_a100_80gb_2048_profile_summary.csv` records the focused evidence
used to explain its performance.

`m3_cuda_resource_usage.csv` records compiler-reported registers, shared
memory, local memory, and stack use alongside the resource-model prediction.

`m4_microtile2x2_predictions.csv` freezes the first hypothesis-driven tuning
prediction before implementation or measurement.

`microtile2x2_a100_80gb_baseline.csv` is the controlled M4 sweep, and
`m4_microtile2x2_a100_80gb_2048_profile_summary.csv` records the focused
profiler evidence used to evaluate the frozen hypothesis.

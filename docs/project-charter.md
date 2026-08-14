# Project Charter

## Objective

Build a small, understandable analytical performance-modeling framework for GPU
compute workloads and validate it against real hardware.

## Method

For each implementation or configuration:

1. Quantify useful computation.
2. Estimate data movement and resource requirements.
3. Predict the limiting hardware resource and approximate performance.
4. Measure the implementation on GPU hardware.
5. Compare prediction with measurement.
6. Use targeted profiling to explain discrepancies.
7. Optimize from the resulting architectural insight.

## Primary workload

FP32 GEMM, `C = A × B`, with approximately `2MKN` floating-point operations for
matrices of dimensions `M × K` and `K × N`.

## Planned progression

- M0: trustworthy naive-GEMM experimental baseline
- M1: computation, traffic, arithmetic-intensity, and roofline model
- M2: shared-memory tiled GEMM
- M3: resource and occupancy modeling
- M4: small, hypothesis-driven tuning study
- M5: hypothetical accelerator sensitivity analysis
- M6: apply the methodology to 2D convolution
- M7: aggregate model-to-hardware correlation study
- M8: documentation and portfolio presentation

## Guardrails

- Prefer depth on GEMM over superficial workload breadth.
- Predict before measuring.
- Collect profiler metrics only when they answer a stated question.
- Treat model error as evidence to investigate.
- Avoid blind tuning and unexplained optimization.
- Do not add a conceptual layer until the current one is understood.


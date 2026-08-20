# Milestone Understanding Checkpoints

This file records what the project owner should be able to explain at the end
of each milestone. Completing the implementation and passing the understanding
checkpoint are separate events. The checklist is a learning guide, not a list
of facts to memorize before doing the work.

## M0 — Trustworthy experimental baseline

Be able to explain:

- how one CUDA thread maps to one output element of the naive GEMM;
- why GEMM performs approximately `2MKN` floating-point operations;
- why warmups, repeated trials, and CUDA-event kernel timing are used;
- what is included in kernel time and what is excluded, especially host/device
  copies and CPU correctness checking;
- how achieved GFLOP/s is calculated from operation count and elapsed time;
- why numerical correctness and recorded hardware/software metadata are needed
  before a performance number is trustworthy;
- why small problems underuse the GPU and why the largest M0 cases form a
  throughput plateau; and
- why a plateau is an observation, not by itself a bottleneck diagnosis.

Checkpoint status: implementation complete; revisit verbally as needed.

## M1 — Arithmetic intensity and roofline prediction

Be able to explain:

- the derivation of `2MKN` FLOPs;
- the difference between algorithmic-minimum traffic, kernel-implied scalar
  requests, and measured traffic at a named memory-hierarchy level;
- why a load in CUDA source does not necessarily become an HBM transaction;
- arithmetic intensity as `FLOPs / bytes`, including which traffic definition
  supplies the bytes;
- the A100 FP32 and HBM bandwidth ceilings and why the TF32 Tensor Core ceiling
  does not apply to this kernel;
- the roofline equation `min(peak compute, AI * peak bandwidth)`;
- the ridge point and how it separates bandwidth-limited and compute-limited
  regions;
- why the two traffic assumptions produce endpoints rather than a statistical
  confidence interval;
- why measured performance can exceed the no-reuse endpoint without violating
  the roofline model; and
- why a result below peak compute does not alone diagnose the bottleneck.

Checkpoint status: analytical model and comparison complete; profiler evidence
and final explanation pending.

## M2 — Shared-memory tiled GEMM

Be able to explain:

- how a tile lets threads cooperate to reuse A and B values;
- why global-memory traffic per useful FLOP decreases with useful tile reuse;
- the roles of global memory, shared memory, registers, and synchronization;
- how tile dimensions affect reuse, coalescing, edge handling, and resource use;
- why synchronization is required between tile load and compute phases;
- how to predict the tiled kernel before measuring it; and
- which measurements show whether the expected traffic reduction occurred.

Checkpoint status: not started.

## M3 — Resource and occupancy model

Be able to explain:

- how threads per block, registers per thread, and shared memory per block
  constrain resident blocks and warps per SM;
- the difference between theoretical occupancy and achieved occupancy;
- how occupancy helps hide latency;
- why higher occupancy does not automatically mean higher performance;
- how instruction-level parallelism and dependency chains interact with
  latency hiding; and
- how resource predictions compare with compiler and profiler measurements.

Checkpoint status: not started.

## M4 — Hypothesis-driven tuning

Be able to explain:

- the specific bottleneck hypothesis behind every attempted optimization;
- which variable changed and which variables were controlled;
- the predicted performance effect before measurement;
- whether profiler evidence supports the causal explanation; and
- why unsuccessful changes are useful evidence rather than results to hide.

Checkpoint status: not started.

## M5 — Accelerator sensitivity analysis

Be able to explain:

- how changing compute throughput or memory bandwidth moves a roofline;
- how the ridge point changes with the hardware balance;
- which workload or kernel properties remain fixed in a sensitivity study;
- when more compute provides little benefit and when more bandwidth provides
  little benefit; and
- why the analysis is a parameter study rather than a claim that a complete
  new accelerator has been designed.

Checkpoint status: not started.

## M6 — Apply the method to 2D convolution

Be able to explain:

- the convolution operation count and stated data-movement assumptions;
- how convolution reuse differs from GEMM reuse;
- how layout, boundary handling, and mapping choices affect traffic;
- how to form and freeze a roofline prediction before measurement; and
- which parts of the GEMM methodology transfer directly and which do not.

Checkpoint status: optional and not started.

## M7 — Model-to-hardware correlation study

Be able to explain:

- how prediction error is defined consistently across experiments;
- how results are grouped without mixing hardware, software, or methodology;
- where the simplified model correlates well and where it fails;
- which omitted effects explain systematic residuals; and
- the difference between fitting past measurements and making a useful
  prediction before measurement.

Checkpoint status: not started.

## M8 — Documentation and portfolio presentation

Be able to explain the complete chain without relying on the code:

```text
workload → assumptions → prediction → measurement → discrepancy →
profiler evidence → optimization or model refinement
```

Also be able to defend the experimental controls, reproduce the key results,
state the model's limitations, and distinguish demonstrated conclusions from
remaining hypotheses.

Checkpoint status: not started.

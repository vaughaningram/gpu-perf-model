# Project Roadmap and Completion Gates

## Target outcome

Produce an unusually strong undergraduate GPU performance-modeling project
that demonstrates the complete scientific loop:

```text
derive → predict → freeze → measure → profile → explain → refine
```

The final project should be understandable, reproducible, and defensible. Its
quality comes from correct assumptions, controlled evidence, and clear model
limitations rather than from the number of kernels or profiler counters.

## Current position

- M0 established a correct, reproducible naive-GEMM baseline.
- M1 built and tested the traffic/roofline model, froze predictions, and used
  profiling to identify L1TEX load-path pressure rather than HBM bandwidth.
- M2 predicted and implemented 16x16 shared-memory tiling. It achieved 4.638
  TFLOP/s, a 1.53x speedup, and moved dominant pressure to the MIO path used by
  shared-memory instructions.

M2's technical work is complete. Its owner understanding checkpoint remains.

## M2 gate — Explain the optimization

Before M3, the owner should be able to explain without reading the source:

1. how cooperative tile loading produces reuse;
2. why the model predicts 16x fewer global input requests;
3. why 16x fewer requests produced only a 1.53x speedup;
4. why request-based arithmetic intensity is not measured DRAM intensity;
5. how the profiler shows a bottleneck displacement from global-load pressure
   to shared-memory/MIO pressure; and
6. why approximately 98% achieved occupancy does not imply peak performance.

Deliverable: a concise owner-written or owner-spoken explanation incorporated
into the final presentation narrative.

## M3 — Resource and latency-hiding model

### Questions

- What limits resident blocks and warps for each kernel?
- How do registers, shared memory, threads, and architecture limits combine?
- Why can 98% occupancy coexist with many cycles having no eligible warp?

### Work

1. Record A100 SM limits from authoritative architecture documentation.
2. Collect compiler resource usage for naive and tiled kernels: registers per
   thread, static shared memory per block, threads per block, and spills.
3. Implement a Python occupancy model that calculates the block limit imposed
   independently by threads, warps, registers, shared memory, and maximum
   blocks per SM.
4. Test boundary cases and compare predictions with Nsight Compute's
   theoretical occupancy and block-limit fields.
5. Explain theoretical occupancy, achieved occupancy, eligible warps, issue
   efficiency, and latency hiding as distinct quantities.

### Completion gate

- Predicted block limits match compiler/profiler evidence.
- No claim treats higher occupancy as automatically faster.
- One diagram or table clearly traces resources to resident warps to issue
  behavior.

## M4 — Small hypothesis-driven tuning study

Limit the study to two or three changes. Each change must have a frozen
hypothesis and controlled comparison.

### Candidate hypotheses

1. **More work per thread:** compute a small output microtile per thread to
   reuse shared-memory operands and reduce MIO instructions per FMA.
2. **Tile-shape sensitivity:** compare a very small set of tile/microtile
   configurations predicted to change reuse and resource limits.
3. **Double buffering, only if justified:** overlap tile loading with useful
   work if profiling still shows exposed memory-pipeline latency and the added
   resource cost is modeled first.

Do not add all three automatically. M3 evidence selects the experiments.

### For every experiment

- state the expected counter and performance changes before implementation;
- change one conceptual variable;
- run the same correctness and A100 sweep methodology;
- compare performance, resource use, and only the counters needed to test the
  hypothesis; and
- preserve failed hypotheses with an explanation.

### Completion gate

- At least one successful and, if encountered, one unsuccessful optimization
  are causally explained.
- The final selected kernel is faster for understood reasons, not merely the
  winner of a search.

## M5 — Hardware sensitivity analysis

Parameterize the existing model over:

- FP32 compute throughput;
- memory bandwidth;
- selected on-chip/resource limits supported by M3 evidence; and
- workload size or arithmetic intensity.

Produce compact sensitivity plots showing when additional compute or bandwidth
stops helping each kernel. Use real A100 values as the anchor and clearly label
hypothetical points as parameter studies, not proposed chip designs.

### Completion gate

- Every plot has units, assumptions, and an interpretable conclusion.
- The study explains hardware/workload balance rather than merely drawing
  multiple rooflines.

## M6 — Optional transfer to 2D convolution

Only begin this milestone if M0–M5 and their explanations are strong. A small
direct-convolution case is enough.

The purpose is methodological transfer:

- derive operations and named traffic assumptions;
- freeze a prediction;
- implement one understandable kernel;
- measure and explain one discrepancy; and
- identify what transfers from GEMM and what is workload-specific.

Skip M6 if it would weaken the GEMM correlation study or final presentation.

## M7 — Model-to-hardware correlation study

Create the project's synthesis dataset across the retained naive, tiled, and
tuned kernels and selected sizes.

### Required analysis

- predicted versus measured GFLOP/s;
- absolute and relative model error;
- achieved fraction of the relevant ceiling;
- predicted versus observed bottleneck category;
- measured traffic at explicitly named hierarchy levels; and
- residual explanations tied to omitted model effects.

### Required visual artifacts

1. Roofline plot with frozen predictions and measured points.
2. Predicted-versus-measured scatter plot with an ideal correlation line.
3. Speedup and model-error comparison across kernels/sizes.
4. One resource/bottleneck table or diagram linking M1–M4.

### Completion gate

- The study distinguishes genuine prediction from post-measurement fitting.
- Outliers are explained rather than removed.
- Conclusions state where the simple model works and where it needs resource or
  hierarchy-aware refinement.

## M8 — Final undergraduate portfolio package

### Repository

- one-command CPU tests and documented A100 build/run commands;
- curated CSV data with commit, hardware, and software provenance;
- clean analytical-model APIs with equation tests;
- raw profiler binaries excluded, concise profiler summaries retained;
- a README that leads with the question, method, key result, and reproducible
  path; and
- no stale milestone claims or unexplained experimental files.

### Written report

Use a compact research-paper structure:

1. problem and research question;
2. background: GEMM mapping, hierarchy, arithmetic intensity, roofline;
3. methodology and controls;
4. model derivations and frozen predictions;
5. measurements and profiler evidence;
6. optimization study;
7. model correlation and limitations; and
8. conclusions and future work.

### Presentation

Build a 10–12 slide narrative centered on three moments:

1. the naive request model was wrong because caches matter;
2. tiling improved performance but displaced the bottleneck; and
3. resource-aware reasoning selected the next optimization.

### Final defense standard

The owner should be able to explain every important equation, experimental
control, plot, and conclusion; identify which statements are measurements,
predictions, or hypotheses; and describe at least three limitations without
undermining the value of the work.

## Scope controls

- Keep FP32 CUDA-core GEMM as the central story.
- Do not turn the project into a cuBLAS competition.
- Do not perform broad tile-size or parameter sweeps without a model-derived
  reason.
- Do not add Tensor Cores unless the original project is already complete and
  they form a clearly separated extension.
- Prefer one explained optimization over five unexplained fast kernels.
- Convolution is optional; correlation and presentation are mandatory.

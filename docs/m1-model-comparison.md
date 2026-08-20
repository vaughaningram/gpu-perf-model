# M1 Model Comparison and Profiling Plan

The analytical predictions were frozen in commit `acee1f9` before this
comparison was made.

## Comparison with M0

| Square size | Measured | Measured / no-reuse roofline | FP32 peak achieved | AI required at peak HBM |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 433.439 GFLOP/s | 0.899x | 2.22% | 0.224 FLOP/byte |
| 256 | 1,689.072 GFLOP/s | 3.498x | 8.66% | 0.873 FLOP/byte |
| 512 | 2,580.157 GFLOP/s | 5.339x | 13.23% | 1.333 FLOP/byte |
| 1024 | 2,995.931 GFLOP/s | 6.196x | 15.36% | 1.548 FLOP/byte |
| 2048 | 3,025.102 GFLOP/s | 6.255x | 15.51% | 1.563 FLOP/byte |

The 2048 case exceeds the hypothetical no-reuse HBM roofline by 6.25x. That
does not violate roofline theory: `naive_scalar_request_bytes` counts requests
in the source, not transactions at HBM. The result demonstrates that some
requests are coalesced, broadcast, or served by cache.

The same case reaches only 15.5% of the vendor FP32 ceiling. The
algorithmic-minimum model assumes perfect reuse and therefore predicts the
compute ceiling, but it does not say this source code can efficiently reach
that ceiling. The remaining gap may involve memory-hierarchy traffic,
instruction issue, dependency latency, occupancy, address arithmetic, or some
combination. These are hypotheses until profiled.

`AI required at peak HBM` is measured GFLOP/s divided by 1,935 GB/s. It is a
useful consistency threshold, not measured AI: if HBM were saturated, at least
that much work per HBM byte would be required to sustain the observed rate.

## Focused profiler questions

Profile the plateau case, 2048x2048x2048, first. Collect these Nsight Compute
sections:

- `SpeedOfLight` and `SpeedOfLight_RooflineChart`: determine whether compute or
  memory throughput is closest to its device limit and obtain profiler-derived
  FLOP/byte.
- `MemoryWorkloadAnalysis`: measure DRAM/L2/L1 traffic and cache behavior,
  replacing the two analytical traffic extremes with observed hierarchy data.
- `ComputeWorkloadAnalysis`: identify which execution pipelines are active and
  whether FP32 throughput is limiting.
- `Occupancy`: compare theoretical and achieved occupancy.
- `SchedulerStats` and `WarpStateStats`: test whether insufficient eligible
  warps and dependency or memory stalls leave issue slots unused.
- `InstructionStats`: quantify the instruction mix around the useful FFMA work.

Run `scripts/profile_naive_a100.sh` inside an A100 80GB PCIe Slurm allocation.
The report should be interpreted before collecting more sizes; 512 is the next
useful contrast if the plateau case alone does not distinguish the hypotheses.

Nsight Compute's definitions and interpretation guidance are documented in the
[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Profile result

The focused 2048x2048x2048 profile ran on the canonical A100 80GB PCIe on
2026-08-20. Nsight Compute 2025.2 collected 16 replay passes. The benchmark's
timer is invalid under replay instrumentation; the controlled M0 timing remains
the performance measurement.

| Evidence | Profile value | Interpretation |
| --- | ---: | --- |
| FP32 peak achieved | 16% | Agrees with the approximately 15.5% M0 result |
| DRAM throughput | 0.38%, 7.32 GB/s | HBM bandwidth is not the limiter |
| L1/TEX throughput | 95.31% | The most heavily utilized memory-hierarchy unit |
| L1/TEX hit rate | 87.59% | Most requests avoid DRAM but still use L1TEX resources |
| L2 hit rate | 98.64% | Almost all requests reaching L2 are served there |
| Compute (SM) throughput | 63.28% | Some execution resources are busy without approaching FP32 peak |
| ALU pipeline utilization | 51.7% | Integer/logic work, including indexing, is substantial |
| Issue slots busy | 57.75% | Schedulers frequently cannot issue an instruction |
| Cycles with no eligible warp | 42.25% | High occupancy does not provide a ready warp every cycle |
| LG throttle stall share | 38.7% | Frequent global-memory operations fill the L1 instruction queue |
| Long-scoreboard stall share | 35.7% | Warps wait on L1TEX load dependencies |
| Achieved occupancy | 98.27% | Insufficient occupancy is not the explanation |

### M1 conclusion

The naive GEMM is not limited by A100 HBM bandwidth. Hardware caching and
broadcast/coalescing make its actual DRAM traffic dramatically smaller than the
source-level scalar-request count, explaining why measurement exceeded the
pessimistic HBM endpoint by 6.25x.

The kernel nevertheless issues global loads in every inner-loop iteration.
Those requests are mostly served by cache, but they still consume the L1TEX
load path and create data dependencies. Nsight Compute reports near-saturated
L1/TEX throughput, frequent load-queue throttling, and long-scoreboard waits.
Schedulers have no eligible warp in 42.25% of cycles even though achieved
occupancy is 98.27%. Useful FP32 work consequently reaches only about 16% of
peak, while integer/logic address work also consumes execution capacity.

The appropriate next hypothesis is explicit shared-memory tiling: reduce the
number of global load instructions, expose controlled on-chip reuse, and then
test whether L1TEX pressure and load-dependency stalls fall. This conclusion
motivates M2; it does not yet claim how much speedup tiling will achieve.

## Size contrast: 512 versus 2048

A second focused profile at 512x512x512 tested whether the plateau case's
behavior was already present at a smaller size.

| Metric | 512 | 2048 | Change with size |
| --- | ---: | ---: | --- |
| FP32 peak achieved | 13% | 16% | More useful throughput at the plateau |
| L1/TEX throughput | 85.45% | 95.31% | L1TEX approaches saturation |
| DRAM throughput | 0.68% | 0.38% | HBM remains clearly non-limiting |
| Achieved occupancy | 76.70% | 98.27% | Larger grid supplies steadier parallel work |
| Cycles with no eligible warp | 47.68% | 42.25% | More warps improve latency hiding, but gaps remain |
| Long-scoreboard stall share | 54.4% | 35.7% | Waiting for loaded data dominates the smaller case |
| LG-throttle stall share | not reported as dominant | 38.7% | Sustained loads fill the L1 instruction queue at 2048 |

The 512 grid contains enough work to reach 13% of FP32 peak but does not keep
the entire GPU in steady state as effectively as 2048. Its lower achieved
occupancy leaves fewer eligible warps to cover L1TEX load latency. The larger
case supplies more parallel work and improves FP32 utilization, but its
sustained stream of global load instructions drives L1/TEX throughput to 95%
and exposes load-queue throttling. Thus increasing problem size explains the
rise toward the M0 plateau, while it does not remove the underlying load-path
bottleneck.

## Cross-node reproducibility

An uninstrumented repeat sweep ran on A100 80GB PCIe node `pax007` on
2026-08-20 with the same five warmups and twenty measured iterations. The raw
repeat file remains in Pax scratch storage rather than replacing the frozen M0
dataset.

| Size | M0 on pax106 | Repeat on pax007 | Relative difference |
| ---: | ---: | ---: | ---: |
| 128 | 433.439 GFLOP/s | 424.456 GFLOP/s | -2.07% |
| 256 | 1,689.072 GFLOP/s | 1,689.072 GFLOP/s | 0.00% |
| 512 | 2,580.157 GFLOP/s | 2,581.428 GFLOP/s | +0.05% |
| 1024 | 2,995.931 GFLOP/s | 2,998.287 GFLOP/s | +0.08% |
| 2048 | 3,025.102 GFLOP/s | 3,029.062 GFLOP/s | +0.13% |

The plateau cases reproduce within 0.13% across the two nodes. The 128 case is
more variable because its approximately 10-microsecond kernel duration makes
fixed launch, scheduling, and clock-state effects a larger fraction of the
measurement. The repeat supports treating the approximately 3.03 TFLOP/s
plateau as stable behavior of this implementation on the target SKU.

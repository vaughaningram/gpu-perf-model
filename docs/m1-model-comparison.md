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

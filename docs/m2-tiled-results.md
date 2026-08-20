# M2 Tiled-GEMM Results

## Measurement

The frozen 16x16 tiled prediction was implemented at commit `6f1b78d` and
measured on the canonical NVIDIA A100 80GB PCIe. All five square sizes passed
the CPU-reference correctness check.

| Size | Naive GFLOP/s | Tiled GFLOP/s | Speedup | Tiled request roofline achieved |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 433.439 | 611.343 | 1.41x | 8.39% |
| 256 | 1,689.072 | 2,229.116 | 1.32x | 29.70% |
| 512 | 2,580.157 | 3,915.519 | 1.52x | 51.38% |
| 1024 | 2,995.931 | 4,531.933 | 1.51x | 59.01% |
| 2048 | 3,025.102 | 4,638.113 | 1.53x | 60.16% |

At 2048, tiling reduced kernel time from 5.679 ms to 3.704 ms and increased
throughput from 3.025 to 4.638 TFLOP/s. This confirms the frozen directional
prediction that explicit reuse would materially outperform the naive plateau.
The result remains below the provisional 7.710 TFLOP/s request-based roofline,
so profiling is required to explain the residual rather than adjusting the
prediction after measurement.

## Focused 2048 profile

The tiled kernel was profiled with the same Nsight Compute 2025.2 sections as
the naive 2048 kernel. Replay-instrumented benchmark timing is excluded; the
controlled sweep above remains the performance measurement.

| Metric | Naive 2048 | Tiled16 2048 | Interpretation |
| --- | ---: | ---: | --- |
| FP32 peak achieved | 16% | 24% | Matches the useful-throughput improvement |
| Compute (SM) throughput | 63.28% | 71.91% | More execution capacity is used |
| DRAM throughput | 0.38% | 0.57% | Neither kernel is HBM-bandwidth-bound |
| L1/TEX throughput | 95.31% | 92.66% | Aggregate on-chip memory pressure remains high |
| L1/TEX hit rate | 87.59% | 2.78% | Far fewer global loads remain; most are compulsory tile loads |
| L2 hit rate | 98.64% | 98.23% | Global tile loads are overwhelmingly served by L2 |
| Executed instructions | 1.954 billion | 1.228 billion | Tiling removes about 37% of total instructions |
| Issue slots busy | 57.75% | 54.95% | The new instruction mix still leaves issue gaps |
| No eligible warp | 42.25% | 45.05% | High occupancy cannot hide every dependency/queue wait |
| Achieved occupancy | 98.27% | 97.94% | Occupancy is not the limiting resource |
| Dominant reported stall | LG throttle and long scoreboard | MIO throttle, 31.5% | Pressure moved from global loads to shared-memory/control issue |

## Conclusion

The first tiled kernel succeeds because each global tile load feeds multiple
floating-point operations. It performs 37% fewer total instructions and raises
FP32 utilization from 16% to 24%, producing a stable 1.53x large-problem
speedup. HBM bandwidth remains far from saturated, confirming again that the
request-based HBM roofline is not a precise prediction of this cache-resident
workload.

The optimization displaces rather than eliminates the bottleneck. Shared-memory
loads and block synchronization now put pressure on the MIO instruction path;
Nsight reports MIO-queue throttling as the dominant sampled stall. The
single-output-per-thread design also retains a serial accumulator dependency
chain. A later tuning hypothesis can therefore consider doing more output work
per thread or using wider/vectorized shared-memory operations, but M2 should
first stop at this explained result rather than begin blind tuning.

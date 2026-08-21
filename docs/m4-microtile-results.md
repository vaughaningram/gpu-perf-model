# M4 2x2 Microtile Results

## Prediction outcome

The frozen 2x2 output-microtile hypothesis was tested at commit `1a67cf6` on
the canonical A100 80GB PCIe. All sizes passed CPU-reference correctness.

| Size | Tiled16 GFLOP/s | Microtile GFLOP/s | Microtile speedup | Request roofline achieved |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 611.343 | 487.619 | 0.80x | 3.54% |
| 256 | 2,229.116 | 2,348.961 | 1.05x | 16.12% |
| 512 | 3,915.519 | 5,864.519 | 1.50x | 39.07% |
| 1024 | 4,531.933 | 8,280.955 | 1.83x | 54.33% |
| 2048 | 4,638.113 | 8,997.033 | 1.94x | 58.57% |

At 2048, the microtile reduced kernel time from 3.704 ms to 1.910 ms and
increased performance from 4.638 to 8.997 TFLOP/s. It is 2.97x faster than the
original naive kernel and achieves 46.1% of the A100's 19.5 TFLOP/s FP32 peak.
The smaller 128 case regresses because its reduced grid supplies only 16 blocks
and too little work to use the 108-SM GPU effectively; microtiling targets the
large steady-state regime.

## Resource prediction

The compiler reports 40 registers/thread, 4,096 bytes static shared memory,
and zero local-memory or stack use. This satisfies the frozen expectation of no
spills and no more than 48 registers/thread.

The M3 resource model predicted six register-limited blocks, 48 resident warps,
and 75% theoretical occupancy. Nsight Compute reports those exact values.
Achieved occupancy is 71.16%, down from tiled16's 97.94%.

## Focused profile comparison

| Metric | Tiled16 | Microtile2x2 | Result |
| --- | ---: | ---: | --- |
| FP32 peak achieved | 24% | 46% | Useful throughput nearly doubles |
| Executed instructions | 1.228B | 0.644B | 47.6% fewer instructions |
| FMA pipeline utilization | 37.6% | 50.3% | More issue capacity performs useful math |
| Warp cycles/instruction | 28.52 | 20.34 | Less time between issued instructions |
| Issue slots busy | 54.95% | 55.96% | Slightly more scheduler issue activity |
| No eligible warp | 45.05% | 44.04% | Does not worsen despite fewer resident warps |
| Theoretical occupancy | 100% | 75% | Registers intentionally trade residency for reuse |
| Achieved occupancy | 97.94% | 71.16% | Lower residency is measured, not hidden |
| DRAM throughput | 0.57% | 1.21% | HBM remains far from limiting |

The tiled kernel's dominant MIO-throttle diagnostic is no longer reported for
the microtile. Aggregate L1/TEX throughput remains high because shared-memory
operations use the on-chip memory path, but each set of shared operands now
feeds four FMAs and four independent accumulators.

## Conclusion

This experiment confirms the central M3/M4 hypothesis: maximum occupancy is not
the objective. Microtiling spends registers on four accumulators, lowers
theoretical occupancy from 100% to 75%, and nevertheless nearly doubles large
GEMM performance by reducing instructions per useful operation and increasing
instruction-level parallelism.

The 15.36 TFLOP/s request-based roofline remains an imperfect quantitative
predictor because HBM is not the active boundary. Its reuse direction was
correct, while the resource and profiler models explain why measured
performance stops at 8.997 TFLOP/s.

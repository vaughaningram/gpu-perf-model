# M4 Frozen 2x2 Microtile Predictions

## Design

A 16x16 thread block will compute a 32x32 output tile. Each thread owns a 2x2
output microtile and four independent accumulators. For every K-tile phase, the
block cooperatively loads a 32x16 A tile and a 16x32 B tile into 4 KiB of static
shared memory.

For each inner position, one thread reads two A values and two B values from
shared memory and performs four FMAs. The M2 kernel read two shared values for
one FMA, so the proposed mapping halves shared-memory operand loads per useful
FMA while also exposing four independent accumulator chains.

These predictions are frozen before CUDA implementation, compiler resource
measurement, or hardware timing.

## Traffic and roofline model

| Size | Global-request bytes | Request AI | Input reduction vs naive | Request roofline |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 589,824 | 7.1111 FLOP/byte | 32x | 13,760.0 GFLOP/s |
| 256 | 4,456,448 | 7.5294 FLOP/byte | 32x | 14,569.4 GFLOP/s |
| 512 | 34,603,008 | 7.7576 FLOP/byte | 32x | 15,010.9 GFLOP/s |
| 1024 | 272,629,760 | 7.8769 FLOP/byte | 32x | 15,241.8 GFLOP/s |
| 2048 | 2,164,260,864 | 7.9380 FLOP/byte | 32x | 15,360.0 GFLOP/s |

The request roofline remains labeled as hypothetical because M1 and M2 showed
that global requests do not equal DRAM transactions. Its useful prediction is
the change in implementation-level reuse, not a claim of HBM saturation.

## Register and occupancy sensitivity

The compiler register count is unknown until implementation. With 256 threads
and 4 KiB static shared memory, the A100 resource model predicts:

| Registers/thread | Resident blocks/SM | Resident warps/SM | Theoretical occupancy |
| ---: | ---: | ---: | ---: |
| 32 | 8 | 64 | 100% |
| 40 | 6 | 48 | 75% |
| 48 | 5 | 40 | 62.5% |
| 56 | 4 | 32 | 50% |
| 64 | 4 | 32 | 50% |

The frozen resource expectation is no register spills and no more than 48
registers/thread, retaining at least 62.5% theoretical occupancy. Lower
occupancy than tiled16 is expected and is not automatically a regression: each
eligible warp will expose more independent useful work.

## Frozen hypotheses

1. Correctness will hold for divisible and edge dimensions.
2. At 2048, performance should materially exceed tiled16's 4.638 TFLOP/s.
3. Total shared-memory operand loads per FMA should fall by approximately 2x.
4. MIO-throttle pressure per useful FMA should fall, while FP32 utilization and
   instruction-level parallelism should rise.
5. Register use may reduce occupancy, but the expected increase in useful work
   per warp should outweigh that loss if occupancy remains at least 62.5%.
6. Failure modes include register spilling, excessive address/control work,
   synchronization cost, bank conflicts, or too little latency hiding after a
   larger-than-expected occupancy reduction.

The experiment succeeds scientifically even if performance does not improve,
provided compiler and profiler evidence identify which frozen assumption failed.

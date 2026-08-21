# M4 4x1 Microtile Contrast Results

The frozen contrast predicted that an elongated 4x1 per-thread microtile would
remain faster than tiled16 but lose to the balanced 2x2 microtile. All sizes
passed correctness, and the large-size prediction was confirmed.

| Size | Microtile2x2 | Microtile4x1 | 4x1 / 2x2 |
| ---: | ---: | ---: | ---: |
| 128 | 487.619 GFLOP/s | 505.679 GFLOP/s | 1.04x |
| 256 | 2,348.961 GFLOP/s | 2,374.493 GFLOP/s | 1.01x |
| 512 | 5,864.519 GFLOP/s | 5,793.237 GFLOP/s | 0.99x |
| 1024 | 8,280.955 GFLOP/s | 8,200.008 GFLOP/s | 0.99x |
| 2048 | 8,997.033 GFLOP/s | 8,845.943 GFLOP/s | 0.98x |

At 2048, 4x1 is 1.68% slower than 2x2 while remaining 1.91x faster than
tiled16. The compiler reports 48 registers/thread and 5 KiB static shared
memory with no spills. The M3 model predicts five register-limited blocks and
62.5% theoretical occupancy; Nsight reports those exact values and 60.06%
achieved occupancy.

## Why 2x2 wins

| Metric | 2x2 | 4x1 |
| --- | ---: | ---: |
| Input-request reduction vs naive | 32x | 25.6x |
| Static shared memory | 4 KiB | 5 KiB |
| Registers/thread | 40 | 48 |
| Executed instructions | 643.8M | 674.1M |
| Theoretical occupancy | 75% | 62.5% |
| Achieved occupancy | 71.16% | 60.06% |
| Issue slots busy | 55.96% | 58.13% |
| No eligible warp | 44.04% | 41.87% |
| Warp cycles/instruction | 20.34 | 16.53 |
| FP32 peak achieved | 46% | 46% |

The lower occupancy is not, by itself, the explanation: 4x1 actually improves
issue-slot use, no-eligible cycles, and warp cycles per instruction. Its small
performance loss aligns instead with weaker balanced reuse and 4.7% more
executed instructions. The shape needs four A operands and one B operand for
four FMAs, while 2x2 needs two of each and uses less shared storage.

This controlled loser strengthens the tuning study. Holding four outputs per
thread constant shows that output count alone does not determine performance;
balanced reuse and total instruction cost matter. The retained M4 kernel is
therefore microtile2x2.

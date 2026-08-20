# M2 Frozen Tiled-GEMM Predictions

## Proposed implementation

The first M2 kernel will use 16x16 thread blocks and 16x16 shared-memory tiles.
Each thread loads at most one A value and one B value into shared memory per K
phase, synchronizes, performs up to 16 multiply-adds using the tile, and
synchronizes before the shared arrays are overwritten. Edge loads are guarded
and out-of-range values are replaced with zero.

These predictions were frozen before implementing or measuring the tiled CUDA
kernel.

## Global-request model

For tile width `T`, guarded global-request elements are:

```text
ceil(N/T) * M * K + ceil(M/T) * K * N + M * N
```

The first two terms load A once per output-column tile and B once per
output-row tile. The last term writes C. For square dimensions divisible by 16,
the input request count is 16x smaller than in the naive one-thread-per-output
kernel.

| Size | Tiled request bytes | Tiled request AI | Input-request reduction | Request-based roofline |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1,114,112 | 3.7647 FLOP/byte | 16x | 7,284.7 GFLOP/s (bandwidth) |
| 256 | 8,650,752 | 3.8788 FLOP/byte | 16x | 7,505.5 GFLOP/s (bandwidth) |
| 512 | 68,157,440 | 3.9385 FLOP/byte | 16x | 7,620.9 GFLOP/s (bandwidth) |
| 1024 | 541,065,216 | 3.9690 FLOP/byte | 16x | 7,680.0 GFLOP/s (bandwidth) |
| 2048 | 4,311,744,512 | 3.9844 FLOP/byte | 16x | 7,709.9 GFLOP/s (bandwidth) |

The request-based roofline applies the A100 peak HBM bandwidth to
kernel-implied global requests. M1 demonstrated that requests do not equal DRAM
transactions, so this is a deliberately labeled analytical ceiling rather
than a claim that HBM will be saturated.

## Frozen hypotheses

Before measurement, the predictions are:

1. The tiled kernel will pass the same CPU-reference correctness checks,
   including dimensions not divisible by 16.
2. At the 2048 plateau size, it should materially exceed the naive kernel's
   stable approximately 3.03 TFLOP/s because it removes most global-load
   instructions from the inner product.
3. L1/TEX throughput and LG-throttle or long-scoreboard pressure attributable
   to global loads should fall relative to naive GEMM.
4. The first implementation is not expected to reach the 19.5 TFLOP/s FP32
   peak. It introduces shared-memory loads, two block barriers per K tile,
   address/control instructions, and a single-accumulator dependency chain.
5. If performance does not improve, focused profiling should test whether
   barriers, shared-memory traffic, instruction issue, or a resource limit has
   replaced L1TEX global-load pressure as the dominant constraint.

No measured tiled result was used to select these statements or values.

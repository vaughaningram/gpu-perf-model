# Naive GEMM Baseline

## Implementation mapping

The baseline computes `C = A × B` for row-major FP32 matrices with one CUDA
thread responsible for one output element `C[row, column]`.

The launch uses a 16 × 16 thread block. Thread coordinates map to output
coordinates as follows:

```text
column = blockIdx.x * blockDim.x + threadIdx.x
row    = blockIdx.y * blockDim.y + threadIdx.y
```

Each valid thread loops over the inner dimension `K`:

```text
accumulator += A[row, k] * B[k, column]
```

It then writes one value to `C[row, column]`. Bounds checks allow dimensions
that are not multiples of 16.

## Work

There are `M × N` outputs. Each output performs approximately `K`
multiplications and `K` additions, giving:

```text
FLOPs ≈ 2MKN
```

The benchmark uses this count and CUDA-event kernel time to calculate achieved
GFLOP/s.

## Data movement and locality

At the source/kernel-request level, every output thread requests two FP32 input
values per inner-loop iteration and writes one FP32 output value. This gives the
simple kernel-implied counts:

```text
input load requests ≈ 2MKN floats
output store requests = MN floats
```

These counts are not DRAM traffic. Repeated requests may be served by L1 or L2
cache, and a warp's requests may combine into fewer memory transactions.

With `threadIdx.x` mapped to adjacent output columns, neighboring threads in a
warp request adjacent elements of `B[k, column]`, which supports coalesced
access. Threads sharing an output row request the same `A[row, k]` value during
an inner-loop iteration, creating reuse/broadcast potential. The kernel does not
explicitly stage either operand in shared memory, so it leaves reuse capture to
the hardware memory hierarchy.

An idealized algorithmic-minimum count is different:

```text
A elements read: MK
B elements read: KN
C elements written: MN
```

That minimum assumes each input element is fetched from the compared memory
level once and then reused ideally. M1 will calculate arithmetic intensity using
clearly named traffic definitions rather than treating these two counts as
interchangeable.

## First controlled sweep

The first controlled sweep ran on an NVIDIA A100 80 GB PCIe using CUDA 12.9.
All configurations passed CPU-reference comparison.

| Square size | Average kernel time (ms) | Achieved GFLOP/s |
| ---: | ---: | ---: |
| 128 | 0.009677 | 433.44 |
| 256 | 0.019866 | 1,689.07 |
| 512 | 0.104038 | 2,580.16 |
| 1024 | 0.716800 | 2,995.93 |
| 2048 | 5.679104 | 3,025.10 |

Throughput rises rapidly at small sizes and approaches a plateau near 3.0
TFLOP/s for the largest two cases. Small problems provide too little total work
to use the full machine efficiently and are more sensitive to fixed launch and
scheduling costs. At larger sizes, execution time scales close to the cubic
increase in GEMM work while achieved throughput stabilizes.

This plateau is an observation, not yet a bottleneck diagnosis. M1 will compare
the implementation against compute and bandwidth ceilings, and later focused
profiling will test the predicted explanation.

The source data is stored in
[`results/naive_a100_80gb_baseline.csv`](../results/naive_a100_80gb_baseline.csv).


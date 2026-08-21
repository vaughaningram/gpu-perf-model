# M3 Resource and Occupancy Model

## A100 SM limits

For compute capability 8.0, the model uses:

- 2,048 resident threads per SM;
- 64 resident warps per SM;
- 32 resident blocks per SM;
- 65,536 32-bit registers per SM; and
- an active 64 KiB shared-memory carveout for the measured kernels.

The architecture supports up to 164 KiB shared memory per SM and several
selectable carveouts. The 64 KiB active value is inferred from the measured
tiled-kernel limit: its 2 KiB static tile storage plus CUDA's documented 1 KiB
per-block reservation yields 3 KiB/block, and `floor(64 / 3) = 21`, exactly
matching Nsight Compute's `Block Limit Shared Mem = 21`.

Architecture sources:

- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [CUDA compute-capability tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

## Compiler resource evidence

CUDA 12.9 `cuobjdump --dump-resource-usage` reports:

| Kernel | Registers/thread | Static shared/block | Local memory | Stack |
| --- | ---: | ---: | ---: | ---: |
| Naive | 30 | 0 bytes | 0 | 0 |
| Tiled16 | 32 | 2,048 bytes | 0 | 0 |

Zero local memory and stack usage provide direct evidence that neither kernel
spills registers. Both kernels launch 256 threads, or 8 warps, per block.

The A100 allocates registers in per-warp units. Thirty registers per thread
requires 960 registers/warp and rounds to 1,024; 32 registers per thread uses
exactly 1,024. Both therefore allocate 8,192 registers/block and reach the same
`floor(65,536 / 8,192) = 8` register block limit.

## Model-to-profiler correlation

| Limit in blocks/SM | Naive | Tiled16 | Nsight match |
| --- | ---: | ---: | --- |
| Architectural block limit | 32 | 32 | yes |
| Thread limit | 8 | 8 | yes |
| Warp limit | 8 | 8 | yes |
| Register limit | 8 | 8 | yes |
| Shared-memory limit | 64 | 21 | yes |
| Predicted resident blocks | 8 | 8 | yes |
| Predicted theoretical occupancy | 100% | 100% | yes |

The tiled kernel uses shared memory, but shared memory does not constrain its
residency because the thread, warp, and register limits reach eight blocks
first. This is why both kernels can have 100% theoretical occupancy.

The model does not predict that every resident warp is ready to issue. M1 and
M2 measured approximately 98% achieved occupancy while schedulers had no
eligible warp in more than 40% of cycles. Residency supplies latency-hiding
opportunity; dependencies and full instruction queues determine whether a warp
is actually eligible.

The compiler values, model predictions, and Nsight block limits now agree. The
remaining M3 learning task is to explain why theoretical residency and achieved
occupancy describe available warps, while eligible warps and issue-slot use
describe whether dependencies and instruction queues allow those warps to run.

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

## Initial correlation

The initial model uses the profile-implied 32 registers per thread until the
compiler report is collected directly. Both kernels launch 256 threads, or 8
warps, per block.

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

## Next evidence

Collect compiler resource reports for both kernel entry points and replace the
profile-implied register count with directly reported registers and spill
information. The M3 gate is satisfied only when those compiler values and the
Nsight block limits agree with the model.

# Analytical model

The M1 model begins with hardware-independent workload quantities. It keeps two
traffic interpretations separate:

- `algorithmic_minimum_bytes`: read every FP32 element of A and B once and
  write every element of C once.
- `naive_scalar_request_bytes`: count the FP32 load/store requests implied by
  the one-thread-per-output source code before coalescing, broadcasts, caches,
  or memory-transaction granularity.

The second quantity is a source-level diagnostic. It is not measured HBM
traffic and must not be combined with HBM bandwidth without labeling the result
as a hypothetical no-reuse scenario.

Run one model evaluation from the repository root:

```bash
python -m model.gemm_model 512 512 512
python -m model.gemm_model 512 512 512 --roofline
python -m model.gemm_model 512 512 512 --roofline --tile-size 16
python -m model.gemm_model 512 512 512 --csv
```

The canonical hardware profile is the NVIDIA A100 80GB PCIe measured in M0:

- peak ordinary FP32: 19,500 GFLOP/s
- peak HBM2e bandwidth: 1,935 GB/s
- ridge point: 19,500 / 1,935 = 10.078 FLOP/byte

These are decimal vendor peaks from the
[NVIDIA A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf).
The FP32 value is used instead of the TF32 Tensor Core value because the naive
kernel issues ordinary FP32 multiply-add operations and no Tensor Core matrix
instructions.

For each traffic interpretation, the model applies:

```text
roofline performance = min(peak FP32, arithmetic intensity * peak bandwidth)
```

The tiled model counts guarded global requests for a square shared-memory tile.
It assumes each block loads one A tile and one B tile per K phase, then reuses
those values from shared memory. It does not claim those global requests equal
DRAM traffic, and it does not yet model shared-memory bandwidth, synchronization,
instruction issue, or occupancy costs.

Run the equation tests with:

```bash
python -m unittest tests.test_gemm_model
```

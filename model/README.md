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
python -m model.gemm_model 512 512 512 --csv
```

Run the equation tests with:

```bash
python -m unittest tests.test_gemm_model
```


# M7 Model-to-Hardware Correlation Study

## Dataset and error definition

The synthesis dataset joins 20 frozen predictions to measurements collected
later: four kernels at five square sizes. No prediction is refit after seeing
the result.

For each point:

```text
signed error = measured - predicted
absolute relative error = abs(measured - predicted) / measured
```

Positive signed error means the request model underpredicted measurement;
negative means it overpredicted.

## Large-problem correlation

| Kernel | Frozen 2048 prediction | Measured | Absolute relative error | Measured / predicted | Speedup vs naive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive | 0.484 TFLOP/s | 3.025 TFLOP/s | 84.0% | 6.25x | 1.00x |
| Tiled16 | 7.710 TFLOP/s | 4.638 TFLOP/s | 66.2% | 0.60x | 1.53x |
| Microtile4x1 | 12.307 TFLOP/s | 8.846 TFLOP/s | 39.1% | 0.72x | 2.92x |
| Microtile2x2 | 15.360 TFLOP/s | 8.997 TFLOP/s | 70.7% | 0.59x | 2.97x |

The simple model is not an accurate absolute timing predictor. Naive is
underpredicted because source scalar requests greatly exceed DRAM traffic after
coalescing, broadcast, and cache service. Optimized kernels are overpredicted
because multiplying request AI by peak HBM bandwidth omits shared-memory/MIO,
instruction, dependency, register, and issue constraints.

## What the model gets right

For sizes 512, 1024, and 2048, the frozen request model correctly ranks all
four implementations:

```text
microtile2x2 > microtile4x1 > tiled16 > naive
```

The model also correctly predicted every retained design direction:

- tiling would materially outperform naive at large sizes;
- 2x2 microtiling would materially outperform tiled16;
- balanced 2x2 would beat elongated 4x1; and
- increasing implementation reuse moves the design closer to the compute side
  of the request roofline.

Thus the model is useful for comparative architectural reasoning even though
its absolute performance estimates need hierarchy/resource corrections.

## Where it fails

At 128 and 256, the model's ranking is incomplete because it contains no grid
occupancy, fixed launch cost, or wave-quantization term. For example, the 128
2x2 microtile launches only 16 blocks across 108 SMs and loses to tiled16 even
though its request AI is higher.

Across all sizes, a single ambiguous byte count cannot explain performance.
The profiler progression names the missing effects:

| Kernel | Named 2048 constraint/effect |
| --- | --- |
| Naive | L1TEX global-load pressure and load dependencies |
| Tiled16 | Shared-memory/MIO issue pressure |
| Microtile2x2 | On-chip instruction/issue balance with register-limited residency |
| Microtile4x1 | Extra instructions and weaker balanced reuse |

The correlation study therefore supports a layered modeling conclusion:

1. operation/request models predict useful design direction;
2. grid and resource models explain residency and small-size effects; and
3. named profiler hierarchy/issue evidence explains remaining residuals.

## Reproducible artifacts

- [`results/m7_model_hardware_correlation.csv`](../results/m7_model_hardware_correlation.csv)
- [`results/m7_predicted_vs_measured.svg`](../results/m7_predicted_vs_measured.svg)
- [`results/m7_measured_performance.svg`](../results/m7_measured_performance.svg)
- [`results/m7_model_error.svg`](../results/m7_model_error.svg)

Regenerate them with:

```bash
python scripts/generate_correlation.py
```

The predicted-versus-measured scatter includes an ideal-correlation line. The
model-error plot uses a logarithmic error axis so small-size and plateau errors
remain visible without deleting outliers.

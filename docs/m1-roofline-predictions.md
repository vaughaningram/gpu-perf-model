# M1 Frozen Roofline Predictions

## Scope

These predictions apply to the naive FP32 GEMM and the NVIDIA A100 80GB PCIe
used for the controlled M0 sweep. They were recorded before comparison with the
M0 measurements. No measured result was used to tune the model.

## Hardware ceilings

The NVIDIA A100 data sheet gives the 80GB PCIe SKU a peak ordinary FP32 rate of
19.5 TFLOP/s and peak HBM2e bandwidth of 1,935 GB/s. The model uses decimal
units, so the roofline ridge point is:

```text
19,500 GFLOP/s / 1,935 GB/s = 10.0775 FLOP/byte
```

Source: [NVIDIA A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)

The 19.5 TFLOP/s FP32 ceiling is appropriate for this kernel. The much larger
TF32 number describes Tensor Core matrix instructions, which the naive kernel
does not issue.

## Frozen predictions

For each traffic assumption, performance is predicted with:

```text
P = min(19,500 GFLOP/s, AI * 1,935 GB/s)
```

| Square size | Minimum-traffic AI | Minimum-traffic roofline | Scalar-request AI | Scalar-request roofline |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 21.3333 | 19,500 GFLOP/s (compute) | 0.249027 | 481.868 GFLOP/s (bandwidth) |
| 256 | 42.6667 | 19,500 GFLOP/s (compute) | 0.249513 | 482.807 GFLOP/s (bandwidth) |
| 512 | 85.3333 | 19,500 GFLOP/s (compute) | 0.249756 | 483.278 GFLOP/s (bandwidth) |
| 1024 | 170.6667 | 19,500 GFLOP/s (compute) | 0.249878 | 483.514 GFLOP/s (bandwidth) |
| 2048 | 341.3333 | 19,500 GFLOP/s (compute) | 0.249939 | 483.632 GFLOP/s (bandwidth) |

These endpoints are not a statistical confidence interval. The
minimum-traffic endpoint assumes perfect reuse after one HBM fetch of each
input element. The scalar-request endpoint hypothetically charges every
source-level load request to HBM and assumes no cache reuse. Actual HBM traffic
can lie between those extremes.

The raw frozen values are in
[`results/m1_a100_80gb_pcie_roofline_predictions.csv`](../results/m1_a100_80gb_pcie_roofline_predictions.csv).

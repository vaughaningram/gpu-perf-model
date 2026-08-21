# M5 Hardware Sensitivity Analysis

## Purpose and scope

This milestone varies hardware parameters while holding each 2048x2048x2048
kernel model fixed. It is a sensitivity study anchored at the A100 80GB PCIe,
not a proposed accelerator design and not a claim that vendor peaks predict
application timing exactly.

The request-roofline curves use each implementation's kernel-implied global
request AI. Solid lines in the plots are model outputs; diamonds are measured
A100 performance at the 1x hardware point. Their separation is intentional and
shows where request traffic is not measured DRAM traffic.

## Compute and bandwidth balance

At fixed A100 FP32 throughput, the bandwidth scale required for each
request-roofline curve to reach the compute ceiling is:

| Kernel | Request AI | Bandwidth scale at ridge |
| --- | ---: | ---: |
| Naive | 0.2499 FLOP/byte | 40.32x |
| Tiled16 | 3.9844 FLOP/byte | 2.53x |
| Microtile4x1 | 6.3602 FLOP/byte | 1.58x |
| Microtile2x2 | 7.9380 FLOP/byte | 1.27x |

The progression is the model's central hardware-balance result: implementation
reuse moves GEMM toward the compute side of the roofline. Balanced 2x2 needs
far less hypothetical bandwidth growth than naive before additional bandwidth
stops helping its request model.

At fixed A100 bandwidth, doubling peak FP32 compute changes none of the four
base request ceilings; each is already on its request-bandwidth slope. Reducing
compute eventually moves the higher-AI microtiles onto the compute ceiling,
while the naive request model remains bandwidth-limited over the plotted range.

## Profile-aware correction

The simple curves must not be interpreted as expected speedups from purchasing
a higher-bandwidth GPU. Measured A100 DRAM utilization was:

| Kernel | DRAM throughput utilization |
| --- | ---: |
| Naive | 0.38% |
| Tiled16 | 0.57% |
| Microtile2x2 | 1.21% |
| Microtile4x1 | 1.25% |

None of the measured kernels saturates HBM. M1–M4 instead identified L1TEX,
shared-memory/MIO, instruction count, dependency, and issue effects. Therefore
more physical HBM bandwidth alone is unlikely to produce the request-roofline
gains. The first-order model correctly describes how explicit reuse changes
the implementation's work/request ratio, while the profile supplies the named
hierarchy level needed for a real bottleneck conclusion.

This mismatch is not a failed milestone. It demonstrates why sensitivity
results inherit the assumptions of the model being varied.

## Register-file sensitivity

Holding compiler register counts and block shapes fixed, the occupancy model
varies the A100 register-file capacity:

| Kernel | 0.5x registers | 1x A100 | 1.5x registers |
| --- | ---: | ---: | ---: |
| Naive | 50% | 100% | 100% |
| Tiled16 | 50% | 100% | 100% |
| Microtile2x2 | 37.5% | 75% | 100% |
| Microtile4x1 | 25% | 62.5% | 100% |

An A100 register file 1.5x as large would remove the modeled register-residency
limit for both microtiles. This is only a residency result, not a predicted
speedup. M4 showed that 2x2 at 75% occupancy substantially outperformed the
100%-occupancy tiled kernel, and 4x1's lower occupancy did not worsen its issue
metrics. More registers create additional latency-hiding opportunity only when
the workload has ready warps that can use it.

## Artifacts and reproduction

- [`results/m5_bandwidth_sensitivity.svg`](../results/m5_bandwidth_sensitivity.svg)
- [`results/m5_compute_sensitivity.svg`](../results/m5_compute_sensitivity.svg)
- [`results/m5_hardware_sensitivity.csv`](../results/m5_hardware_sensitivity.csv)
- [`results/m5_register_file_sensitivity.csv`](../results/m5_register_file_sensitivity.csv)

Regenerate every M5 artifact from the repository root:

```bash
python scripts/generate_sensitivity.py
```

The script uses only the Python standard library and project models.

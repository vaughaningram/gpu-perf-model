# M4 Frozen 4x1 Microtile Contrast

This controlled contrast holds four outputs per thread constant while changing
the output microtile from balanced 2x2 to elongated 4x1. A 16x16 thread block
therefore computes a 64x16 output tile using a 64x16 A tile and a 16x16 B tile.

The prediction was frozen before implementation or measurement.

| Size | Request bytes | Request AI | Input reduction vs naive | Request roofline |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 720,896 | 5.8182 | 25.6x | 11,258.2 GFLOP/s |
| 256 | 5,505,024 | 6.0952 | 25.6x | 11,794.3 GFLOP/s |
| 512 | 42,991,616 | 6.2439 | 25.6x | 12,082.0 GFLOP/s |
| 1024 | 339,738,624 | 6.3210 | 25.6x | 12,231.1 GFLOP/s |
| 2048 | 2,701,131,776 | 6.3602 | 25.6x | 12,307.1 GFLOP/s |

Compared with 2x2, the 4x1 shape has the same four accumulators and output area
but weaker balanced reuse:

- input-request reduction falls from 32x to 25.6x;
- static shared memory rises from 4 KiB to 5 KiB; and
- each group of four FMAs consumes four A operands and one B operand rather
  than two A and two B operands.

The frozen hypothesis is that 4x1 will remain correct and faster than tiled16
but will be slower than 2x2 at large sizes. Similar register pressure is
expected because both retain four accumulators. If 4x1 wins, the profile must
identify an advantage—such as access mapping or instruction scheduling—that
the simple reuse model omitted.

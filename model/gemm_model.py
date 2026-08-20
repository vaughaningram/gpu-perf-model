"""Hardware-independent analytical quantities for FP32 GEMM."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GemmProblem:
    """Dimensions and element size for C[M,N] = A[M,K] * B[K,N]."""

    m: int
    k: int
    n: int
    bytes_per_element: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("m", self.m),
            ("k", self.k),
            ("n", self.n),
            ("bytes_per_element", self.bytes_per_element),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class GemmWorkloadModel:
    """Computation, traffic assumptions, and arithmetic intensities."""

    m: int
    k: int
    n: int
    bytes_per_element: int
    flops: int
    algorithmic_minimum_bytes: int
    naive_scalar_request_bytes: int
    algorithmic_minimum_ai: float
    naive_scalar_request_ai: float

    def as_row(self) -> dict[str, int | float]:
        return asdict(self)


def model_gemm(problem: GemmProblem) -> GemmWorkloadModel:
    """Calculate FP32 GEMM work and two explicitly named traffic models."""

    m, k, n = problem.m, problem.k, problem.n
    flops = 2 * m * k * n

    # Idealized lower bound: fetch each A/B element once and write C once.
    algorithmic_minimum_elements = m * k + k * n + m * n
    algorithmic_minimum_bytes = (
        problem.bytes_per_element * algorithmic_minimum_elements
    )

    # Source-level scalar requests: every output thread loads one A and one B
    # value for each k, then stores one C value. This is not HBM traffic.
    naive_scalar_request_elements = 2 * m * k * n + m * n
    naive_scalar_request_bytes = (
        problem.bytes_per_element * naive_scalar_request_elements
    )

    return GemmWorkloadModel(
        m=m,
        k=k,
        n=n,
        bytes_per_element=problem.bytes_per_element,
        flops=flops,
        algorithmic_minimum_bytes=algorithmic_minimum_bytes,
        naive_scalar_request_bytes=naive_scalar_request_bytes,
        algorithmic_minimum_ai=flops / algorithmic_minimum_bytes,
        naive_scalar_request_ai=flops / naive_scalar_request_bytes,
    )


def _positive_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _write_csv(result: GemmWorkloadModel) -> None:
    row = result.as_row()
    writer = csv.DictWriter(sys.stdout, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)


def _write_human_readable(result: GemmWorkloadModel) -> None:
    print(f"problem={result.m}x{result.k}x{result.n}")
    print(f"bytes_per_element={result.bytes_per_element}")
    print(f"flops={result.flops}")
    print(f"algorithmic_minimum_bytes={result.algorithmic_minimum_bytes}")
    print(f"naive_scalar_request_bytes={result.naive_scalar_request_bytes}")
    print(f"algorithmic_minimum_ai={result.algorithmic_minimum_ai:.6f}")
    print(f"naive_scalar_request_ai={result.naive_scalar_request_ai:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate hardware-independent FP32 GEMM model quantities."
    )
    parser.add_argument("m", type=_positive_integer)
    parser.add_argument("k", type=_positive_integer)
    parser.add_argument("n", type=_positive_integer)
    parser.add_argument("--csv", action="store_true", help="emit one CSV row")
    arguments = parser.parse_args()

    result = model_gemm(GemmProblem(arguments.m, arguments.k, arguments.n))
    if arguments.csv:
        _write_csv(result)
    else:
        _write_human_readable(result)


if __name__ == "__main__":
    main()


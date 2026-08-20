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


@dataclass(frozen=True)
class TiledGemmModel:
    """Global-request model for a square shared-memory GEMM tile."""

    workload: GemmWorkloadModel
    tile_size: int
    output_row_tiles: int
    output_column_tiles: int
    tiled_global_request_bytes: int
    tiled_global_request_ai: float
    input_request_reduction: float


def model_tiled_gemm(
    problem: GemmProblem, tile_size: int = 16
) -> TiledGemmModel:
    """Count guarded global loads made by a square shared-memory tiled kernel.

    Each output tile loads every required A row segment once per output-column
    tile and every required B column segment once per output-row tile. This
    counts source-level global requests, not measured DRAM transactions.
    """

    if (
        not isinstance(tile_size, int)
        or isinstance(tile_size, bool)
        or tile_size <= 0
    ):
        raise ValueError("tile_size must be a positive integer")

    workload = model_gemm(problem)
    output_row_tiles = (problem.m + tile_size - 1) // tile_size
    output_column_tiles = (problem.n + tile_size - 1) // tile_size

    # Edge guards mean only valid elements become global requests. Summing all
    # K tiles covers each valid K element exactly once for each output tile.
    tiled_global_request_elements = (
        output_column_tiles * problem.m * problem.k
        + output_row_tiles * problem.k * problem.n
        + problem.m * problem.n
    )
    tiled_global_request_bytes = (
        problem.bytes_per_element * tiled_global_request_elements
    )

    naive_input_bytes = problem.bytes_per_element * 2 * problem.m * problem.k * problem.n
    tiled_input_bytes = problem.bytes_per_element * (
        output_column_tiles * problem.m * problem.k
        + output_row_tiles * problem.k * problem.n
    )

    return TiledGemmModel(
        workload=workload,
        tile_size=tile_size,
        output_row_tiles=output_row_tiles,
        output_column_tiles=output_column_tiles,
        tiled_global_request_bytes=tiled_global_request_bytes,
        tiled_global_request_ai=workload.flops / tiled_global_request_bytes,
        input_request_reduction=naive_input_bytes / tiled_input_bytes,
    )


@dataclass(frozen=True)
class HardwareCeilings:
    """Peak ceilings expressed in decimal GFLOP/s and GB/s."""

    name: str
    fp32_gflops: float
    memory_bandwidth_gbytes_per_second: float

    @property
    def ridge_point_flops_per_byte(self) -> float:
        return self.fp32_gflops / self.memory_bandwidth_gbytes_per_second


A100_80GB_PCIE = HardwareCeilings(
    name="NVIDIA A100 80GB PCIe",
    fp32_gflops=19_500.0,
    memory_bandwidth_gbytes_per_second=1_935.0,
)


@dataclass(frozen=True)
class RooflinePrediction:
    """Roofline predictions for the two workload traffic assumptions."""

    workload: GemmWorkloadModel
    hardware: HardwareCeilings
    algorithmic_minimum_gflops: float
    algorithmic_minimum_limit: str
    naive_scalar_request_gflops: float
    naive_scalar_request_limit: str


def _roofline_endpoint(
    arithmetic_intensity: float, hardware: HardwareCeilings
) -> tuple[float, str]:
    bandwidth_ceiling = (
        arithmetic_intensity * hardware.memory_bandwidth_gbytes_per_second
    )
    if bandwidth_ceiling < hardware.fp32_gflops:
        return bandwidth_ceiling, "bandwidth"
    return hardware.fp32_gflops, "compute"


def predict_roofline(
    workload: GemmWorkloadModel,
    hardware: HardwareCeilings = A100_80GB_PCIE,
) -> RooflinePrediction:
    """Apply min(peak compute, AI * peak bandwidth) to both AI definitions."""

    algorithmic_performance, algorithmic_limit = _roofline_endpoint(
        workload.algorithmic_minimum_ai, hardware
    )
    naive_performance, naive_limit = _roofline_endpoint(
        workload.naive_scalar_request_ai, hardware
    )
    return RooflinePrediction(
        workload=workload,
        hardware=hardware,
        algorithmic_minimum_gflops=algorithmic_performance,
        algorithmic_minimum_limit=algorithmic_limit,
        naive_scalar_request_gflops=naive_performance,
        naive_scalar_request_limit=naive_limit,
    )


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


def _write_roofline(result: GemmWorkloadModel) -> None:
    prediction = predict_roofline(result)
    print(f"hardware={prediction.hardware.name}")
    print(f"peak_fp32_gflops={prediction.hardware.fp32_gflops:.6f}")
    print(
        "peak_memory_bandwidth_gbytes_per_second="
        f"{prediction.hardware.memory_bandwidth_gbytes_per_second:.6f}"
    )
    print(
        "ridge_point_flops_per_byte="
        f"{prediction.hardware.ridge_point_flops_per_byte:.6f}"
    )
    print(
        "algorithmic_minimum_roofline_gflops="
        f"{prediction.algorithmic_minimum_gflops:.6f}"
    )
    print(f"algorithmic_minimum_limit={prediction.algorithmic_minimum_limit}")
    print(
        "naive_scalar_request_roofline_gflops="
        f"{prediction.naive_scalar_request_gflops:.6f}"
    )
    print(f"naive_scalar_request_limit={prediction.naive_scalar_request_limit}")


def _write_tiled(problem: GemmProblem, tile_size: int) -> None:
    tiled = model_tiled_gemm(problem, tile_size)
    performance, limit = _roofline_endpoint(
        tiled.tiled_global_request_ai, A100_80GB_PCIE
    )
    print(f"tile_size={tiled.tile_size}")
    print(f"output_row_tiles={tiled.output_row_tiles}")
    print(f"output_column_tiles={tiled.output_column_tiles}")
    print(f"tiled_global_request_bytes={tiled.tiled_global_request_bytes}")
    print(f"tiled_global_request_ai={tiled.tiled_global_request_ai:.6f}")
    print(f"input_request_reduction={tiled.input_request_reduction:.6f}")
    print(f"tiled_global_request_roofline_gflops={performance:.6f}")
    print(f"tiled_global_request_limit={limit}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate hardware-independent FP32 GEMM model quantities."
    )
    parser.add_argument("m", type=_positive_integer)
    parser.add_argument("k", type=_positive_integer)
    parser.add_argument("n", type=_positive_integer)
    parser.add_argument("--csv", action="store_true", help="emit one CSV row")
    parser.add_argument(
        "--roofline",
        action="store_true",
        help="append NVIDIA A100 80GB PCIe roofline predictions",
    )
    parser.add_argument(
        "--tile-size",
        type=_positive_integer,
        help="append a square shared-memory tiled global-request model",
    )
    arguments = parser.parse_args()

    result = model_gemm(GemmProblem(arguments.m, arguments.k, arguments.n))
    if arguments.csv:
        _write_csv(result)
    else:
        _write_human_readable(result)
        if arguments.roofline:
            _write_roofline(result)
        if arguments.tile_size is not None:
            _write_tiled(GemmProblem(arguments.m, arguments.k, arguments.n), arguments.tile_size)


if __name__ == "__main__":
    main()

"""Parameterized roofline and register-file sensitivity calculations."""

from __future__ import annotations

from dataclasses import dataclass, replace

from model.gemm_model import A100_80GB_PCIE
from model.resource_model import (
    A100_CC80_64K_SHARED,
    KernelResources,
    model_occupancy,
)


@dataclass(frozen=True)
class KernelSensitivityInput:
    name: str
    request_ai: float
    measured_gflops: float
    resources: KernelResources
    measured_dram_percent: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        for name in ("request_ai", "measured_gflops"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.measured_dram_percent <= 100:
            raise ValueError("measured_dram_percent must be between 0 and 100")


@dataclass(frozen=True)
class RooflineSensitivityPoint:
    kernel: str
    compute_scale: float
    bandwidth_scale: float
    predicted_gflops: float
    limiting_ceiling: str


@dataclass(frozen=True)
class RegisterSensitivityPoint:
    kernel: str
    register_file_scale: float
    resident_blocks: int
    resident_warps: int
    theoretical_occupancy: float
    limiting_resources: tuple[str, ...]


def request_roofline_point(
    kernel: KernelSensitivityInput,
    compute_scale: float,
    bandwidth_scale: float,
) -> RooflineSensitivityPoint:
    if compute_scale <= 0 or bandwidth_scale <= 0:
        raise ValueError("hardware scales must be positive")
    compute = A100_80GB_PCIE.fp32_gflops * compute_scale
    bandwidth = (
        A100_80GB_PCIE.memory_bandwidth_gbytes_per_second * bandwidth_scale
    )
    bandwidth_performance = kernel.request_ai * bandwidth
    if bandwidth_performance < compute:
        performance = bandwidth_performance
        limit = "bandwidth"
    else:
        performance = compute
        limit = "compute"
    return RooflineSensitivityPoint(
        kernel.name,
        compute_scale,
        bandwidth_scale,
        performance,
        limit,
    )


def register_sensitivity_point(
    kernel: KernelSensitivityInput,
    register_file_scale: float,
) -> RegisterSensitivityPoint:
    if register_file_scale <= 0:
        raise ValueError("register_file_scale must be positive")
    scaled_registers = int(
        A100_CC80_64K_SHARED.registers * register_file_scale
    )
    sm = replace(A100_CC80_64K_SHARED, registers=scaled_registers)
    occupancy = model_occupancy(kernel.resources, sm)
    return RegisterSensitivityPoint(
        kernel.name,
        register_file_scale,
        occupancy.resident_blocks,
        occupancy.resident_warps,
        occupancy.theoretical_occupancy,
        occupancy.limiting_resources,
    )

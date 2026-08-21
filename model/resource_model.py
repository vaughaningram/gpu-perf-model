"""A100 SM resource and theoretical-occupancy model."""

from __future__ import annotations

from dataclasses import dataclass


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _ceil_divide(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _round_up(value: int, multiple: int) -> int:
    return _ceil_divide(value, multiple) * multiple


@dataclass(frozen=True)
class SmLimits:
    """Architecture limits and the active shared-memory carveout."""

    name: str
    max_threads: int
    max_warps: int
    max_blocks: int
    registers: int
    active_shared_memory_bytes: int
    warp_size: int = 32
    register_allocation_granularity_per_warp: int = 256
    reserved_shared_memory_per_block: int = 1024

    def __post_init__(self) -> None:
        for name in (
            "max_threads",
            "max_warps",
            "max_blocks",
            "registers",
            "active_shared_memory_bytes",
            "warp_size",
            "register_allocation_granularity_per_warp",
        ):
            _positive_integer(name, getattr(self, name))
        _nonnegative_integer(
            "reserved_shared_memory_per_block",
            self.reserved_shared_memory_per_block,
        )


A100_CC80_64K_SHARED = SmLimits(
    name="NVIDIA A100 CC 8.0 with 64 KiB shared-memory carveout",
    max_threads=2048,
    max_warps=64,
    max_blocks=32,
    registers=65_536,
    active_shared_memory_bytes=64 * 1024,
)


@dataclass(frozen=True)
class KernelResources:
    """Per-block resources reported by the compiler or kernel design."""

    threads_per_block: int
    registers_per_thread: int
    static_shared_memory_bytes: int

    def __post_init__(self) -> None:
        _positive_integer("threads_per_block", self.threads_per_block)
        _positive_integer("registers_per_thread", self.registers_per_thread)
        _nonnegative_integer(
            "static_shared_memory_bytes", self.static_shared_memory_bytes
        )


@dataclass(frozen=True)
class OccupancyPrediction:
    warps_per_block: int
    allocated_registers_per_block: int
    allocated_shared_memory_per_block: int
    block_limit_sm: int
    block_limit_threads: int
    block_limit_warps: int
    block_limit_registers: int
    block_limit_shared_memory: int
    resident_blocks: int
    resident_warps: int
    theoretical_occupancy: float
    limiting_resources: tuple[str, ...]


def model_occupancy(
    kernel: KernelResources,
    sm: SmLimits = A100_CC80_64K_SHARED,
) -> OccupancyPrediction:
    """Calculate resident blocks and warps from independently named limits."""

    warps_per_block = _ceil_divide(kernel.threads_per_block, sm.warp_size)
    registers_per_warp = _round_up(
        kernel.registers_per_thread * sm.warp_size,
        sm.register_allocation_granularity_per_warp,
    )
    allocated_registers_per_block = warps_per_block * registers_per_warp
    allocated_shared_memory_per_block = (
        kernel.static_shared_memory_bytes + sm.reserved_shared_memory_per_block
    )

    limits = {
        "sm": sm.max_blocks,
        "threads": sm.max_threads // kernel.threads_per_block,
        "warps": sm.max_warps // warps_per_block,
        "registers": sm.registers // allocated_registers_per_block,
        "shared_memory": (
            sm.active_shared_memory_bytes // allocated_shared_memory_per_block
        ),
    }
    resident_blocks = min(limits.values())
    resident_warps = resident_blocks * warps_per_block
    theoretical_occupancy = resident_warps / sm.max_warps
    limiting_resources = tuple(
        name for name, limit in limits.items() if limit == resident_blocks
    )

    return OccupancyPrediction(
        warps_per_block=warps_per_block,
        allocated_registers_per_block=allocated_registers_per_block,
        allocated_shared_memory_per_block=allocated_shared_memory_per_block,
        block_limit_sm=limits["sm"],
        block_limit_threads=limits["threads"],
        block_limit_warps=limits["warps"],
        block_limit_registers=limits["registers"],
        block_limit_shared_memory=limits["shared_memory"],
        resident_blocks=resident_blocks,
        resident_warps=resident_warps,
        theoretical_occupancy=theoretical_occupancy,
        limiting_resources=limiting_resources,
    )

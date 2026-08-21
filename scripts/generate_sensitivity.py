"""Generate dependency-free M5 sensitivity CSVs and SVG plots."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.resource_model import KernelResources
from model.sensitivity_model import (
    KernelSensitivityInput,
    register_sensitivity_point,
    request_roofline_point,
)


RESULTS = ROOT / "results"
COLORS = {
    "naive": "#4C78A8",
    "tiled16": "#F58518",
    "microtile2x2": "#54A24B",
    "microtile4x1": "#E45756",
}
SCALES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
REGISTER_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)

KERNELS = (
    KernelSensitivityInput(
        "naive", 0.249939, 3025.102135, KernelResources(256, 30, 0), 0.38
    ),
    KernelSensitivityInput(
        "tiled16", 3.984436, 4638.113320,
        KernelResources(256, 32, 2048), 0.57
    ),
    KernelSensitivityInput(
        "microtile2x2", 7.937984, 8997.032608,
        KernelResources(256, 40, 4096), 1.21
    ),
    KernelSensitivityInput(
        "microtile4x1", 6.360248, 8845.943238,
        KernelResources(256, 48, 5120), 1.25
    ),
)


def write_roofline_sensitivity() -> None:
    path = RESULTS / "m5_hardware_sensitivity.csv"
    fields = (
        "kernel",
        "sweep",
        "compute_scale",
        "bandwidth_scale",
        "request_roofline_gflops",
        "predicted_limit",
        "measured_a100_gflops",
        "measured_a100_dram_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for kernel in KERNELS:
            for sweep, factors in (("compute", SCALES), ("bandwidth", SCALES)):
                for factor in factors:
                    compute_scale = factor if sweep == "compute" else 1.0
                    bandwidth_scale = factor if sweep == "bandwidth" else 1.0
                    point = request_roofline_point(
                        kernel, compute_scale, bandwidth_scale
                    )
                    writer.writerow(
                        {
                            "kernel": kernel.name,
                            "sweep": sweep,
                            "compute_scale": f"{compute_scale:.2f}",
                            "bandwidth_scale": f"{bandwidth_scale:.2f}",
                            "request_roofline_gflops": f"{point.predicted_gflops:.6f}",
                            "predicted_limit": point.limiting_ceiling,
                            "measured_a100_gflops": f"{kernel.measured_gflops:.6f}",
                            "measured_a100_dram_percent": f"{kernel.measured_dram_percent:.2f}",
                        }
                    )


def write_register_sensitivity() -> None:
    path = RESULTS / "m5_register_file_sensitivity.csv"
    fields = (
        "kernel",
        "register_file_scale",
        "resident_blocks",
        "resident_warps",
        "theoretical_occupancy_percent",
        "limiting_resources",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for kernel in KERNELS:
            for scale in REGISTER_SCALES:
                point = register_sensitivity_point(kernel, scale)
                writer.writerow(
                    {
                        "kernel": kernel.name,
                        "register_file_scale": f"{scale:.2f}",
                        "resident_blocks": point.resident_blocks,
                        "resident_warps": point.resident_warps,
                        "theoretical_occupancy_percent": (
                            f"{100 * point.theoretical_occupancy:.2f}"
                        ),
                        "limiting_resources": ";".join(point.limiting_resources),
                    }
                )


def write_svg(sweep: str, filename: str, title: str, x_label: str) -> None:
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 60, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = 20_000.0

    def x_position(value: float) -> float:
        return left + (value - SCALES[0]) / (SCALES[-1] - SCALES[0]) * plot_width

    def y_position(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for tick in (0, 5_000, 10_000, 15_000, 20_000):
        y = y_position(tick)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>'
        )
        lines.append(
            f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick/1000:.0f}</text>'
        )
    for scale in SCALES:
        x = x_position(scale)
        lines.append(
            f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{scale:g}x</text>'
        )
    lines.extend(
        [
            f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>',
            f'<text x="20" y="{top+plot_height/2}" transform="rotate(-90 20 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">Performance (TFLOP/s)</text>',
            f'<line x1="{x_position(1.0):.1f}" y1="{top}" x2="{x_position(1.0):.1f}" y2="{height-bottom}" stroke="#777" stroke-dasharray="5,5"/>',
        ]
    )
    for index, kernel in enumerate(KERNELS):
        points = []
        for scale in SCALES:
            compute_scale = scale if sweep == "compute" else 1.0
            bandwidth_scale = scale if sweep == "bandwidth" else 1.0
            point = request_roofline_point(kernel, compute_scale, bandwidth_scale)
            points.append(
                f"{x_position(scale):.1f},{y_position(point.predicted_gflops):.1f}"
            )
        color = COLORS[kernel.name]
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        measured_x = x_position(1.0)
        measured_y = y_position(kernel.measured_gflops)
        lines.append(
            f'<polygon points="{measured_x:.1f},{measured_y-6:.1f} {measured_x+6:.1f},{measured_y:.1f} {measured_x:.1f},{measured_y+6:.1f} {measured_x-6:.1f},{measured_y:.1f}" fill="{color}" stroke="black"/>'
        )
        legend_y = 75 + index * 22
        lines.append(
            f'<line x1="650" y1="{legend_y}" x2="680" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="688" y="{legend_y+5}" font-family="sans-serif" font-size="12">{kernel.name}</text>'
        )
    lines.append(
        '<text x="650" y="170" font-family="sans-serif" font-size="11">solid: request roofline</text>'
    )
    lines.append(
        '<text x="650" y="186" font-family="sans-serif" font-size="11">diamond: measured A100 at 1x</text>'
    )
    lines.append("</svg>")
    (RESULTS / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    write_roofline_sensitivity()
    write_register_sensitivity()
    write_svg(
        "bandwidth",
        "m5_bandwidth_sensitivity.svg",
        "Request-Roofline Sensitivity to HBM Bandwidth",
        "HBM bandwidth relative to A100",
    )
    write_svg(
        "compute",
        "m5_compute_sensitivity.svg",
        "Request-Roofline Sensitivity to FP32 Compute",
        "FP32 throughput relative to A100",
    )


if __name__ == "__main__":
    main()

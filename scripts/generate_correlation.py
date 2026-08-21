"""Join frozen predictions to measurements and generate M7 SVG plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.correlation_model import compare_prediction


RESULTS = ROOT / "results"
SIZES = (128, 256, 512, 1024, 2048)
COLORS = {
    "naive": "#4C78A8",
    "tiled16": "#F58518",
    "microtile2x2": "#54A24B",
    "microtile4x1": "#E45756",
}
SOURCES = {
    "naive": (
        "m1_a100_80gb_pcie_roofline_predictions.csv",
        "naive_a100_80gb_baseline.csv",
        "naive_scalar_request_ai",
        "naive_scalar_request_roofline_gflops",
        "naive_scalar_request_limit",
        "source scalar requests",
        "L1TEX/global-load dependencies",
    ),
    "tiled16": (
        "m2_a100_80gb_tiled16_predictions.csv",
        "tiled16_a100_80gb_baseline.csv",
        "tiled_global_request_ai",
        "tiled_global_request_roofline_gflops",
        "predicted_limit",
        "tiled global requests",
        "shared-memory/MIO issue",
    ),
    "microtile2x2": (
        "m4_microtile2x2_predictions.csv",
        "microtile2x2_a100_80gb_baseline.csv",
        "global_request_ai",
        "request_roofline_gflops",
        "predicted_limit",
        "microtile global requests",
        "on-chip instruction/issue balance",
    ),
    "microtile4x1": (
        "m4_microtile4x1_predictions.csv",
        "microtile4x1_a100_80gb_baseline.csv",
        "global_request_ai",
        "request_roofline_gflops",
        "predicted_limit",
        "microtile global requests",
        "extra instructions/weaker balanced reuse",
    ),
}


def read_rows(filename: str) -> dict[int, dict[str, str]]:
    with (RESULTS / filename).open(newline="", encoding="utf-8") as source:
        return {int(row["m"]): row for row in csv.DictReader(source)}


def build_rows() -> list[dict[str, str | float | int]]:
    naive_measurements = read_rows(SOURCES["naive"][1])
    rows: list[dict[str, str | float | int]] = []
    for kernel, source in SOURCES.items():
        prediction_file, measurement_file, ai_field, prediction_field, limit_field, traffic, diagnosis = source
        predictions = read_rows(prediction_file)
        measurements = read_rows(measurement_file)
        for size in SIZES:
            predicted = float(predictions[size][prediction_field])
            measured = float(measurements[size]["achieved_gflops"])
            comparison = compare_prediction(predicted, measured)
            rows.append(
                {
                    "kernel": kernel,
                    "m": size,
                    "k": size,
                    "n": size,
                    "request_ai": float(predictions[size][ai_field]),
                    "traffic_definition": traffic,
                    "predicted_gflops": predicted,
                    "predicted_limit": predictions[size][limit_field],
                    "measured_gflops": measured,
                    "signed_error_gflops": comparison.signed_error_gflops,
                    "absolute_error_gflops": comparison.absolute_error_gflops,
                    "absolute_relative_error_percent": comparison.relative_error_percent,
                    "measured_over_predicted": comparison.measured_over_predicted,
                    "measured_fp32_peak_percent": comparison.measured_fp32_peak_percent,
                    "measured_speedup_vs_naive": (
                        measured / float(naive_measurements[size]["achieved_gflops"])
                    ),
                    "profile_diagnosis_at_2048": diagnosis,
                }
            )
    return rows


def write_csv(rows: list[dict[str, str | float | int]]) -> None:
    path = RESULTS / "m7_model_hardware_correlation.csv"
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def svg_start(title: str, y_label: str) -> tuple[list[str], dict[str, int]]:
    dims = {"width": 900, "height": 560, "left": 90, "right": 30, "top": 60, "bottom": 80}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{dims["width"]}" height="{dims["height"]}" viewBox="0 0 {dims["width"]} {dims["height"]}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="450" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<text x="20" y="270" transform="rotate(-90 20 270)" text-anchor="middle" font-family="sans-serif" font-size="14">{y_label}</text>',
    ]
    return lines, dims


def add_legend(lines: list[str]) -> None:
    for index, (kernel, color) in enumerate(COLORS.items()):
        y = 72 + index * 20
        lines.append(f'<circle cx="690" cy="{y}" r="5" fill="{color}"/>')
        lines.append(f'<text x="702" y="{y+4}" font-family="sans-serif" font-size="12">{kernel}</text>')


def write_scatter(rows: list[dict[str, str | float | int]]) -> None:
    lines, d = svg_start("Frozen Prediction vs Measured Performance", "Measured performance (TFLOP/s)")
    plot_w = d["width"] - d["left"] - d["right"]
    plot_h = d["height"] - d["top"] - d["bottom"]
    maximum = 20_000.0

    def x(v: float) -> float:
        return d["left"] + v / maximum * plot_w

    def y(v: float) -> float:
        return d["top"] + (1 - v / maximum) * plot_h

    lines.extend([
        f'<line x1="{d["left"]}" y1="{d["top"]}" x2="{d["left"]}" y2="{d["height"]-d["bottom"]}" stroke="#333"/>',
        f'<line x1="{d["left"]}" y1="{d["height"]-d["bottom"]}" x2="{d["width"]-d["right"]}" y2="{d["height"]-d["bottom"]}" stroke="#333"/>',
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(maximum)}" y2="{y(maximum)}" stroke="#777" stroke-dasharray="6,5"/>',
        '<text x="450" y="540" text-anchor="middle" font-family="sans-serif" font-size="14">Frozen request-roofline prediction (TFLOP/s)</text>',
    ])
    for tick in (0, 5_000, 10_000, 15_000, 20_000):
        lines.append(f'<text x="{x(tick):.1f}" y="505" text-anchor="middle" font-family="sans-serif" font-size="12">{tick/1000:.0f}</text>')
        lines.append(f'<text x="80" y="{y(tick)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick/1000:.0f}</text>')
    for row in rows:
        lines.append(
            f'<circle cx="{x(float(row["predicted_gflops"])):.1f}" cy="{y(float(row["measured_gflops"])):.1f}" r="5" fill="{COLORS[str(row["kernel"])]}" fill-opacity="0.8"/>'
        )
    add_legend(lines)
    lines.append("</svg>")
    (RESULTS / "m7_predicted_vs_measured.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_size_plot(rows: list[dict[str, str | float | int]]) -> None:
    lines, d = svg_start("Measured GEMM Performance Progression", "Measured performance (TFLOP/s)")
    plot_w = d["width"] - d["left"] - d["right"]
    plot_h = d["height"] - d["top"] - d["bottom"]
    y_max = 10_000.0

    def x(size: int) -> float:
        return d["left"] + SIZES.index(size) / (len(SIZES) - 1) * plot_w

    def y(value: float) -> float:
        return d["top"] + (1 - value / y_max) * plot_h

    lines.extend([
        f'<line x1="{d["left"]}" y1="{d["top"]}" x2="{d["left"]}" y2="{d["height"]-d["bottom"]}" stroke="#333"/>',
        f'<line x1="{d["left"]}" y1="{d["height"]-d["bottom"]}" x2="{d["width"]-d["right"]}" y2="{d["height"]-d["bottom"]}" stroke="#333"/>',
        '<text x="450" y="540" text-anchor="middle" font-family="sans-serif" font-size="14">Square matrix dimension</text>',
    ])
    for size in SIZES:
        lines.append(f'<text x="{x(size):.1f}" y="505" text-anchor="middle" font-family="sans-serif" font-size="12">{size}</text>')
    for tick in (0, 2_000, 4_000, 6_000, 8_000, 10_000):
        lines.append(f'<line x1="{d["left"]}" y1="{y(tick):.1f}" x2="{d["width"]-d["right"]}" y2="{y(tick):.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="80" y="{y(tick)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick/1000:.0f}</text>')
    for kernel in SOURCES:
        selected = sorted((r for r in rows if r["kernel"] == kernel), key=lambda r: int(r["m"]))
        points = " ".join(f'{x(int(r["m"])):.1f},{y(float(r["measured_gflops"])):.1f}' for r in selected)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[kernel]}" stroke-width="3"/>')
        for r in selected:
            lines.append(f'<circle cx="{x(int(r["m"])):.1f}" cy="{y(float(r["measured_gflops"])):.1f}" r="4" fill="{COLORS[kernel]}"/>')
    add_legend(lines)
    lines.append("</svg>")
    (RESULTS / "m7_measured_performance.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_error_plot(rows: list[dict[str, str | float | int]]) -> None:
    lines, d = svg_start("Request-Model Absolute Relative Error", "Absolute relative error (%) — log scale")
    plot_w = d["width"] - d["left"] - d["right"]
    plot_h = d["height"] - d["top"] - d["bottom"]
    log_min, log_max = 0.0, 4.0

    def x(size: int) -> float:
        return d["left"] + SIZES.index(size) / (len(SIZES) - 1) * plot_w

    def y(value: float) -> float:
        logged = math.log10(max(value, 1.0))
        return d["top"] + (1 - (logged-log_min)/(log_max-log_min)) * plot_h

    lines.extend([
        f'<line x1="{d["left"]}" y1="{d["top"]}" x2="{d["left"]}" y2="{d["height"]-d["bottom"]}" stroke="#333"/>',
        f'<line x1="{d["left"]}" y1="{d["height"]-d["bottom"]}" x2="{d["width"]-d["right"]}" y2="{d["height"]-d["bottom"]}" stroke="#333"/>',
        '<text x="450" y="540" text-anchor="middle" font-family="sans-serif" font-size="14">Square matrix dimension</text>',
    ])
    for size in SIZES:
        lines.append(f'<text x="{x(size):.1f}" y="505" text-anchor="middle" font-family="sans-serif" font-size="12">{size}</text>')
    for tick in (1, 10, 100, 1000, 10000):
        lines.append(f'<line x1="{d["left"]}" y1="{y(tick):.1f}" x2="{d["width"]-d["right"]}" y2="{y(tick):.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="80" y="{y(tick)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick}</text>')
    for kernel in SOURCES:
        selected = sorted((r for r in rows if r["kernel"] == kernel), key=lambda r: int(r["m"]))
        points = " ".join(f'{x(int(r["m"])):.1f},{y(float(r["absolute_relative_error_percent"])):.1f}' for r in selected)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[kernel]}" stroke-width="3"/>')
    add_legend(lines)
    lines.append("</svg>")
    (RESULTS / "m7_model_error.svg").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_scatter(rows)
    write_size_plot(rows)
    write_error_plot(rows)


if __name__ == "__main__":
    main()

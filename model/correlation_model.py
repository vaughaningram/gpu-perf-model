"""Prediction-versus-measurement correlation quantities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelComparison:
    predicted_gflops: float
    measured_gflops: float
    signed_error_gflops: float
    absolute_error_gflops: float
    relative_error_percent: float
    measured_over_predicted: float
    measured_fp32_peak_percent: float


def compare_prediction(
    predicted_gflops: float,
    measured_gflops: float,
    fp32_peak_gflops: float = 19_500.0,
) -> ModelComparison:
    """Compare a prediction with a later measurement without refitting it."""

    for name, value in (
        ("predicted_gflops", predicted_gflops),
        ("measured_gflops", measured_gflops),
        ("fp32_peak_gflops", fp32_peak_gflops),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    signed_error = measured_gflops - predicted_gflops
    absolute_error = abs(signed_error)
    return ModelComparison(
        predicted_gflops=predicted_gflops,
        measured_gflops=measured_gflops,
        signed_error_gflops=signed_error,
        absolute_error_gflops=absolute_error,
        relative_error_percent=100.0 * absolute_error / measured_gflops,
        measured_over_predicted=measured_gflops / predicted_gflops,
        measured_fp32_peak_percent=100.0 * measured_gflops / fp32_peak_gflops,
    )

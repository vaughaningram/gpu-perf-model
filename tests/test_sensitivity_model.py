import unittest

from model.resource_model import KernelResources
from model.sensitivity_model import (
    KernelSensitivityInput,
    register_sensitivity_point,
    request_roofline_point,
)


class SensitivityModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = KernelSensitivityInput(
            "test", 4.0, 1000.0, KernelResources(256, 40, 4096), 1.0
        )

    def test_request_roofline_selects_bandwidth(self) -> None:
        point = request_roofline_point(self.kernel, 1.0, 1.0)

        self.assertEqual(point.limiting_ceiling, "bandwidth")
        self.assertEqual(point.predicted_gflops, 4.0 * 1935.0)

    def test_request_roofline_selects_compute(self) -> None:
        point = request_roofline_point(self.kernel, 0.25, 1.0)

        self.assertEqual(point.limiting_ceiling, "compute")
        self.assertEqual(point.predicted_gflops, 0.25 * 19_500.0)

    def test_register_scaling_changes_residency(self) -> None:
        half = register_sensitivity_point(self.kernel, 0.5)
        normal = register_sensitivity_point(self.kernel, 1.0)
        doubled = register_sensitivity_point(self.kernel, 2.0)

        self.assertEqual(half.resident_blocks, 3)
        self.assertEqual(normal.resident_blocks, 6)
        self.assertEqual(doubled.resident_blocks, 8)
        self.assertEqual(doubled.theoretical_occupancy, 1.0)

    def test_invalid_scales_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            request_roofline_point(self.kernel, 0.0, 1.0)
        with self.assertRaises(ValueError):
            register_sensitivity_point(self.kernel, -1.0)


if __name__ == "__main__":
    unittest.main()

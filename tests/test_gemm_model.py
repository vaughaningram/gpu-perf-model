import unittest

from model.gemm_model import (
    A100_80GB_PCIE,
    GemmProblem,
    HardwareCeilings,
    model_gemm,
    predict_roofline,
)


class GemmModelTests(unittest.TestCase):
    def test_two_by_two_example(self) -> None:
        result = model_gemm(GemmProblem(m=2, k=2, n=2))

        self.assertEqual(result.flops, 16)
        self.assertEqual(result.algorithmic_minimum_bytes, 48)
        self.assertEqual(result.naive_scalar_request_bytes, 80)
        self.assertAlmostEqual(result.algorithmic_minimum_ai, 1.0 / 3.0)
        self.assertAlmostEqual(result.naive_scalar_request_ai, 0.2)

    def test_rectangular_example(self) -> None:
        result = model_gemm(GemmProblem(m=4, k=5, n=1))

        self.assertEqual(result.flops, 40)
        self.assertEqual(result.algorithmic_minimum_bytes, 116)
        self.assertEqual(result.naive_scalar_request_bytes, 176)

    def test_large_integer_counts_are_exact(self) -> None:
        result = model_gemm(GemmProblem(m=2048, k=2048, n=2048))

        self.assertEqual(result.flops, 17_179_869_184)
        self.assertEqual(
            result.algorithmic_minimum_bytes,
            4 * (2048 * 2048 + 2048 * 2048 + 2048 * 2048),
        )
        self.assertEqual(
            result.naive_scalar_request_bytes,
            4 * (2 * 2048 * 2048 * 2048 + 2048 * 2048),
        )

    def test_nonpositive_dimensions_are_rejected(self) -> None:
        for dimensions in ((0, 2, 2), (2, -1, 2), (2, 2, 0)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    GemmProblem(*dimensions)

    def test_boolean_is_not_accepted_as_an_integer_dimension(self) -> None:
        with self.assertRaises(ValueError):
            GemmProblem(True, 2, 2)

    def test_a100_80gb_pcie_ridge_point(self) -> None:
        self.assertAlmostEqual(
            A100_80GB_PCIE.ridge_point_flops_per_byte,
            19_500.0 / 1_935.0,
        )

    def test_roofline_selects_bandwidth_and_compute_limits(self) -> None:
        workload = model_gemm(GemmProblem(m=512, k=512, n=512))
        prediction = predict_roofline(workload)

        self.assertEqual(prediction.algorithmic_minimum_limit, "compute")
        self.assertEqual(prediction.algorithmic_minimum_gflops, 19_500.0)
        self.assertEqual(prediction.naive_scalar_request_limit, "bandwidth")
        self.assertAlmostEqual(
            prediction.naive_scalar_request_gflops,
            workload.naive_scalar_request_ai * 1_935.0,
        )

    def test_roofline_can_be_evaluated_for_other_hardware(self) -> None:
        hardware = HardwareCeilings("test device", 100.0, 25.0)
        workload = model_gemm(GemmProblem(m=2, k=2, n=2))
        prediction = predict_roofline(workload, hardware)

        self.assertAlmostEqual(prediction.algorithmic_minimum_gflops, 25.0 / 3.0)
        self.assertEqual(prediction.algorithmic_minimum_limit, "bandwidth")


if __name__ == "__main__":
    unittest.main()

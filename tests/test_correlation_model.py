import unittest

from model.correlation_model import compare_prediction


class CorrelationModelTests(unittest.TestCase):
    def test_underprediction_is_signed_positive(self) -> None:
        result = compare_prediction(100.0, 125.0, 200.0)

        self.assertEqual(result.signed_error_gflops, 25.0)
        self.assertEqual(result.absolute_error_gflops, 25.0)
        self.assertEqual(result.relative_error_percent, 20.0)
        self.assertEqual(result.measured_over_predicted, 1.25)
        self.assertEqual(result.measured_fp32_peak_percent, 62.5)

    def test_overprediction_is_signed_negative(self) -> None:
        result = compare_prediction(150.0, 100.0)

        self.assertEqual(result.signed_error_gflops, -50.0)
        self.assertEqual(result.absolute_error_gflops, 50.0)
        self.assertEqual(result.relative_error_percent, 50.0)

    def test_nonpositive_inputs_are_rejected(self) -> None:
        for values in ((0.0, 1.0, 2.0), (1.0, -1.0, 2.0), (1.0, 1.0, 0.0)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    compare_prediction(*values)


if __name__ == "__main__":
    unittest.main()

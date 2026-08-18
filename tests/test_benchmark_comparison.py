import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.csdemo.benchmark_comparison import compare_benchmarks, write_markdown_report


class BenchmarkComparisonTests(unittest.TestCase):
    def test_higher_is_better_metric_reports_percentage_point_gap(self) -> None:
        benchmarks = pd.DataFrame(
            [
                {
                    "benchmark_id": "other_accuracy",
                    "metric": "accuracy",
                    "reported_value": 0.60,
                    "direction": "higher",
                }
            ]
        )

        result = compare_benchmarks({"accuracy": 0.65}, benchmarks).iloc[0]

        self.assertEqual(result["comparison_status"], "compared")
        self.assertAlmostEqual(result["raw_difference_ours_minus_reported"], 0.05)
        self.assertAlmostEqual(result["performance_advantage_ours"], 0.05)
        self.assertAlmostEqual(result["difference_percentage_points"], 5.0)

    def test_lower_is_better_metric_flips_performance_direction(self) -> None:
        benchmarks = pd.DataFrame(
            [
                {
                    "benchmark_id": "other_log_loss",
                    "metric": "log_loss",
                    "reported_value": 0.60,
                    "direction": "lower",
                }
            ]
        )

        result = compare_benchmarks({"log_loss": 0.58}, benchmarks).iloc[0]

        self.assertAlmostEqual(result["raw_difference_ours_minus_reported"], -0.02)
        self.assertAlmostEqual(result["performance_advantage_ours"], 0.02)
        self.assertTrue(result["ours_performs_better"])

    def test_missing_current_metric_is_kept_and_marked(self) -> None:
        benchmarks = pd.DataFrame(
            [
                {
                    "benchmark_id": "other_f1",
                    "metric": "f1",
                    "reported_value": 0.70,
                    "direction": "higher",
                }
            ]
        )

        result = compare_benchmarks({"accuracy": 0.65}, benchmarks).iloc[0]

        self.assertEqual(result["comparison_status"], "current_metric_unavailable")
        self.assertTrue(pd.isna(result["current_value"]))
        self.assertTrue(pd.isna(result["performance_advantage_ours"]))

    def test_unknown_metric_direction_is_rejected(self) -> None:
        benchmarks = pd.DataFrame(
            [
                {
                    "benchmark_id": "bad_direction",
                    "metric": "accuracy",
                    "reported_value": 0.60,
                    "direction": "sideways",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "direction"):
            compare_benchmarks({"accuracy": 0.65}, benchmarks)

    def test_non_comparable_report_does_not_claim_model_superiority(self) -> None:
        benchmarks = pd.DataFrame(
            [
                {
                    "benchmark_id": "mid_round_accuracy",
                    "source_title": "Mid-round model",
                    "source_url": "https://example.com",
                    "model": "Random forest",
                    "prediction_point": "In-round snapshot",
                    "metric": "accuracy",
                    "reported_value": 0.88,
                    "direction": "higher",
                    "comparability": "not_comparable",
                }
            ]
        )
        comparison = compare_benchmarks({"accuracy": 0.65}, benchmarks)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_markdown_report(
                comparison, {"accuracy": 0.65}, path, stage_label="M11"
            )
            report = path.read_text(encoding="utf-8")

        self.assertIn("仅数值差，不判断优劣", report)
        self.assertNotIn("我们的模型较差 0.230000", report)


if __name__ == "__main__":
    unittest.main()

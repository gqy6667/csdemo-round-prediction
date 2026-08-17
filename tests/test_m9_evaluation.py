import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.csdemo.m9_evaluation import (
    bootstrap_metric_intervals,
    calibration_table,
    confusion_counts,
)


class M9EvaluationTests(unittest.TestCase):
    def test_group_bootstrap_is_reproducible_and_reports_core_metrics(self) -> None:
        predictions = pd.DataFrame(
            {
                "series_id": np.repeat(["s1", "s2", "s3", "s4"], 2),
                "y_true": [0, 1] * 4,
                "ct_win_probability": [0.10, 0.80, 0.20, 0.90, 0.30, 0.70, 0.40, 0.60],
            }
        )

        first = bootstrap_metric_intervals(predictions, n_bootstrap=50, seed=7)
        second = bootstrap_metric_intervals(predictions, n_bootstrap=50, seed=7)

        assert_frame_equal(first, second)
        self.assertEqual(
            set(first["metric"]),
            {"accuracy", "auc", "log_loss", "brier_score", "ece10"},
        )
        self.assertTrue((first["successful_bootstraps"] == 50).all())
        self.assertTrue((first["ci_lower_95"] <= first["ci_upper_95"]).all())

    def test_calibration_table_uses_equal_width_probability_bins(self) -> None:
        table = calibration_table(
            y_true=[0, 0, 1, 1],
            probability=[0.05, 0.15, 0.85, 0.95],
            n_bins=2,
        )

        self.assertEqual(table["count"].tolist(), [2, 2])
        np.testing.assert_allclose(table["mean_probability"], [0.10, 0.90])
        np.testing.assert_allclose(table["observed_ct_win_rate"], [0.0, 1.0])

    def test_confusion_counts_names_all_four_outcomes(self) -> None:
        counts = confusion_counts(
            y_true=[0, 0, 1, 1],
            probability=[0.10, 0.80, 0.40, 0.90],
            threshold=0.5,
        )

        self.assertEqual(
            counts,
            {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 1},
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from src.csdemo.m11_robustness import (
    assign_equipment_band,
    assign_round_stage,
    group_metrics_with_intervals,
    outcome_error_pattern,
    pre_round_error_pattern,
    select_high_confidence_errors,
)


class M11RobustnessTests(unittest.TestCase):
    def test_round_stage_and_equipment_bands_have_fixed_boundaries(self) -> None:
        self.assertEqual(
            assign_round_stage(pd.Series([1, 10, 11, 20, 21, 35])).tolist(),
            ["early_01_10", "early_01_10", "middle_11_20", "middle_11_20", "late_21_plus", "late_21_plus"],
        )
        self.assertEqual(
            assign_equipment_band(pd.Series([-6000, -5000, -1500, 0, 1500, 5000, 6000])).tolist(),
            ["t_major", "t_moderate", "balanced", "balanced", "balanced", "ct_moderate", "ct_major"],
        )

    def test_high_confidence_selection_keeps_only_wrong_predictions(self) -> None:
        predictions = pd.DataFrame(
            {
                "round_id": ["r1", "r2", "r3", "r4"],
                "y_true": [0, 1, 1, 0],
                "predicted_label": [1, 0, 1, 1],
                "ct_win_probability": [0.95, 0.10, 0.90, 0.60],
            }
        )

        selected = select_high_confidence_errors(
            predictions, minimum_confidence=0.8, max_cases=30
        )

        self.assertEqual(selected["round_id"].tolist(), ["r1", "r2"])
        self.assertTrue((selected["assigned_side_probability"] >= 0.8).all())

    def test_error_patterns_separate_pre_round_and_first_kill_information(self) -> None:
        row = pd.Series(
            {
                "predicted_label": 1,
                "ct_eq_value": 26000,
                "t_eq_value": 17000,
                "eq_value_diff_ct": 9000,
                "rifle_diff_ct": 2,
                "awp_diff_ct": 1,
                "first_kill_side": "T",
            }
        )

        self.assertEqual(
            pre_round_error_pattern(row), "favored_side_major_equipment_upset"
        )
        self.assertEqual(
            outcome_error_pattern(row), "predicted_favorite_lost_first_kill"
        )

    def test_group_metrics_include_sample_size_and_series_bootstrap_ci(self) -> None:
        predictions = pd.DataFrame(
            {
                "map_name": np.repeat(["map_a", "map_b"], 8),
                "series_id": np.tile(np.repeat(["s1", "s2", "s3", "s4"], 2), 2),
                "y_true": [0, 1] * 8,
                "ct_win_probability": [0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6] * 2,
            }
        )

        result = group_metrics_with_intervals(
            predictions, "map_name", n_bootstrap=30, seed=7
        )

        self.assertEqual(set(result["map_name"]), {"map_a", "map_b"})
        self.assertTrue((result["rounds"] == 8).all())
        self.assertTrue((result["series"] == 4).all())
        self.assertIn("auc_ci_lower_95", result.columns)
        self.assertIn("auc_ci_upper_95", result.columns)


if __name__ == "__main__":
    unittest.main()

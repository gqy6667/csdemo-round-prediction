import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.csdemo.m19_first_kill_explanation import (
    FIRST_KILL_EVENT_FEATURES,
    audit_post_first_kill_features,
    build_internal_model_gap,
    build_macro_feature_groups,
    build_source_importance_summary,
    build_source_feature_groups,
    build_target_gap_table,
    grouped_permutation_auc_importance,
    map_encoded_feature_to_source,
)


class M19FirstKillExplanationTests(unittest.TestCase):
    def test_encoded_feature_mapping_handles_numeric_map_and_weapon_columns(self) -> None:
        raw = [
            "eq_value_diff_ct",
            "map_name",
            "first_kill_weapon",
        ]

        self.assertEqual(
            map_encoded_feature_to_source("eq_value_diff_ct", raw),
            "eq_value_diff_ct",
        )
        self.assertEqual(
            map_encoded_feature_to_source("map_name_de_inferno", raw),
            "map_name",
        )
        self.assertEqual(
            map_encoded_feature_to_source("first_kill_weapon_AK-47", raw),
            "first_kill_weapon",
        )

    def test_encoded_feature_mapping_rejects_an_unknown_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not map"):
            map_encoded_feature_to_source(
                "round_winner_after_first_kill",
                ["map_name", "first_kill_time"],
            )

    def test_leakage_audit_allows_only_the_post_first_kill_contract(self) -> None:
        encoded = [
            "eq_value_diff_ct",
            "map_name_de_nuke",
            "first_kill_advantage_ct",
            "first_kill_weapon_AWP",
            "series_id",
            "round_winner",
        ]
        raw = [
            "eq_value_diff_ct",
            "map_name",
            "first_kill_advantage_ct",
            "first_kill_weapon",
        ]

        audit = audit_post_first_kill_features(encoded, raw)

        results = audit.set_index("encoded_feature")["audit_result"].to_dict()
        self.assertEqual(results["eq_value_diff_ct"], "pass")
        self.assertEqual(results["first_kill_weapon_AWP"], "pass")
        self.assertEqual(results["series_id"], "fail")
        self.assertEqual(results["round_winner"], "fail")

    def test_source_groups_cover_every_encoded_column_exactly_once(self) -> None:
        raw = ["round_num", "map_name", "first_kill_weapon"]
        encoded = [
            "round_num",
            "map_name_de_nuke",
            "map_name_de_inferno",
            "first_kill_weapon_AWP",
            "first_kill_weapon_AK-47",
        ]

        groups = build_source_feature_groups(encoded, raw)

        self.assertEqual(set(groups), set(raw))
        flattened = [column for columns in groups.values() for column in columns]
        self.assertCountEqual(flattened, encoded)
        self.assertEqual(groups["map_name"], ["map_name_de_nuke", "map_name_de_inferno"])

    def test_macro_groups_keep_four_first_kill_features_separate_from_purchase(self) -> None:
        raw = ["round_num", "eq_value_diff_ct", *FIRST_KILL_EVENT_FEATURES]
        encoded = [
            "round_num",
            "eq_value_diff_ct",
            "first_kill_advantage_ct",
            "first_kill_time",
            "first_kill_headshot",
            "first_kill_weapon_AWP",
            "first_kill_weapon_AK-47",
        ]

        groups = build_macro_feature_groups(encoded, raw)

        self.assertEqual(set(groups), {"purchase_end", "first_kill_event"})
        self.assertEqual(len(groups["purchase_end"]), 2)
        self.assertEqual(len(groups["first_kill_event"]), 5)

    def test_grouped_permutation_ranks_real_signal_above_noise(self) -> None:
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, size=500)
        x = pd.DataFrame(
            {
                "signal_a": y,
                "signal_b": 1 - y,
                "noise": rng.normal(size=500),
            }
        )
        model = LogisticRegression(random_state=5).fit(x, y)
        groups = {"signal": ["signal_a", "signal_b"], "noise": ["noise"]}

        result = grouped_permutation_auc_importance(
            model,
            x,
            y,
            groups,
            n_repeats=10,
            seed=5,
        ).set_index("feature_group")

        self.assertGreater(
            result.loc["signal", "auc_decrease_mean"],
            result.loc["noise", "auc_decrease_mean"],
        )
        self.assertEqual(result.loc["signal", "encoded_column_count"], 2)

    def test_source_importance_aggregates_encoded_gain_and_shap(self) -> None:
        gain = pd.DataFrame(
            {
                "feature": ["round_num", "map_name_a", "map_name_b"],
                "gain_normalized": [0.2, 0.3, 0.5],
                "split_count": [2, 3, 5],
            }
        )
        shap = pd.DataFrame(
            {
                "feature": ["round_num", "map_name_a", "map_name_b"],
                "mean_abs_shap": [0.1, 0.2, 0.4],
            }
        )
        grouped = pd.DataFrame(
            {
                "feature_group": ["round_num", "map_name"],
                "auc_decrease_mean": [0.01, 0.03],
                "auc_decrease_std": [0.001, 0.002],
                "encoded_column_count": [1, 2],
            }
        )
        contract = audit_post_first_kill_features(
            ["round_num", "map_name_a", "map_name_b"],
            ["round_num", "map_name"],
        )

        result = build_source_importance_summary(
            gain, shap, grouped, contract
        ).set_index("source_feature")

        self.assertAlmostEqual(result.loc["map_name", "gain_normalized"], 0.8)
        self.assertAlmostEqual(result.loc["map_name", "mean_abs_shap"], 0.6)
        self.assertEqual(result.loc["map_name", "split_count"], 8)
        self.assertAlmostEqual(
            result.loc["map_name", "grouped_auc_decrease_mean"], 0.03
        )

    def test_target_gap_reports_margin_when_higher_metric_passes(self) -> None:
        summary = self._m18_summary()

        gaps = build_target_gap_table(summary).set_index("target_id")

        self.assertEqual(gaps.loc["test_auc", "remaining"], 0.0)
        self.assertAlmostEqual(gaps.loc["test_auc", "margin"], 0.03)
        self.assertTrue(bool(gaps.loc["test_auc", "passed"]))

    def test_target_gap_reports_remaining_when_lower_metric_misses(self) -> None:
        summary = self._m18_summary()
        summary["metrics"]["log_loss"] = 0.57

        gaps = build_target_gap_table(summary).set_index("target_id")

        self.assertAlmostEqual(gaps.loc["test_log_loss", "remaining"], 0.02)
        self.assertEqual(gaps.loc["test_log_loss", "margin"], 0.0)
        self.assertFalse(bool(gaps.loc["test_log_loss", "passed"]))

    def test_internal_gap_respects_metric_direction_without_inventing_a_target(self) -> None:
        comparison = pd.DataFrame(
            {
                "model": ["logistic_regression", "xgboost_tuned"],
                "split": ["test", "test"],
                "accuracy": [0.74, 0.75],
                "auc": [0.81, 0.812],
                "log_loss": [0.525, 0.522],
                "brier_score": [0.177, 0.175],
                "ece10": [0.014, 0.016],
            }
        )

        gaps = build_internal_model_gap(comparison).set_index("metric")

        self.assertAlmostEqual(gaps.loc["auc", "performance_advantage_xgboost"], 0.002)
        self.assertAlmostEqual(
            gaps.loc["log_loss", "performance_advantage_xgboost"], 0.003
        )
        self.assertAlmostEqual(gaps.loc["ece10", "performance_advantage_xgboost"], -0.002)
        self.assertTrue(gaps["formal_target"].isna().all())

    @staticmethod
    def _m18_summary() -> dict:
        return {
            "metrics": {
                "accuracy": 0.74,
                "auc": 0.81,
                "log_loss": 0.52,
                "brier_score": 0.175,
                "ece10": 0.015,
            },
            "global_assessment": {
                "auc_ci_lower_95": 0.80,
                "log_loss_ci_upper_95": 0.53,
            },
            "source_auc_gap": {"absolute_difference": 0.01},
            "robustness": {
                "large_map_min_auc": 0.79,
                "large_map_min_auc_ci_lower": 0.75,
            },
        }


if __name__ == "__main__":
    unittest.main()

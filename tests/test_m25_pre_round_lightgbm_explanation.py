import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.csdemo.m25_pre_round_lightgbm_explanation import (
    BLOCKING_CHECKS,
    audit_frozen_prediction_replay,
    audit_pre_round_features,
    build_macro_feature_groups,
    build_model_importance_comparison,
    build_source_feature_groups,
    decide_acceptance,
    encoded_permutation_auc_importance,
    grouped_permutation_auc_importance,
    lightgbm_gain_importance,
    lightgbm_tree_shap_contributions,
    load_m12_xgboost_importance,
    map_encoded_feature_to_source,
)


class _FakeBooster:
    def __init__(self) -> None:
        self.last_num_iteration = None

    def feature_name(self):
        return ["signal", "noise"]

    def feature_importance(self, importance_type, iteration=None):
        self.last_num_iteration = iteration
        if importance_type == "gain":
            return np.array([9.0, 0.0])
        if importance_type == "split":
            return np.array([3, 0])
        raise ValueError(importance_type)

    def num_trees(self):
        return 2

    def predict(self, x, *, pred_contrib, num_iteration):
        if not pred_contrib:
            raise AssertionError("TreeSHAP must request pred_contrib")
        self.last_num_iteration = num_iteration
        values = np.asarray(x, dtype=float)
        base = np.full(len(values), 0.25)
        return np.column_stack([0.8 * values[:, 0], -0.1 * values[:, 1], base])


class _FakeModel:
    def __init__(self) -> None:
        self.booster_ = _FakeBooster()

    def predict_proba(self, x):
        values = np.asarray(x, dtype=float)
        raw = 0.25 + 0.8 * values[:, 0] - 0.1 * values[:, 1]
        probability = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1.0 - probability, probability])


class M25PreRoundLightGBMExplanationTests(unittest.TestCase):
    def test_encoded_feature_mapping_handles_numeric_and_map_columns(self) -> None:
        raw = ["round_num", "map_name", "eq_value_diff_ct"]

        self.assertEqual(
            map_encoded_feature_to_source("eq_value_diff_ct", raw),
            "eq_value_diff_ct",
        )
        self.assertEqual(
            map_encoded_feature_to_source("map_name_de_ancient", raw),
            "map_name",
        )

    def test_encoded_feature_mapping_rejects_unknown_or_future_columns(self) -> None:
        raw = ["round_num", "map_name"]

        with self.assertRaisesRegex(ValueError, "does not map"):
            map_encoded_feature_to_source("first_kill_time", raw)

    def test_leakage_audit_allows_only_purchase_end_contract(self) -> None:
        raw = ["round_num", "map_name", "eq_value_diff_ct"]
        encoded = [
            "round_num",
            "map_name_de_nuke",
            "eq_value_diff_ct",
            "series_id",
            "ct_win",
            "first_kill_time",
            "team_name",
        ]

        audit = audit_pre_round_features(encoded, raw).set_index("encoded_feature")

        self.assertEqual(audit.loc["round_num", "audit_result"], "pass")
        self.assertEqual(audit.loc["map_name_de_nuke", "audit_result"], "pass")
        self.assertEqual(audit.loc["series_id", "reason"], "identifier")
        self.assertEqual(audit.loc["ct_win", "reason"], "label_or_split")
        self.assertEqual(audit.loc["first_kill_time", "reason"], "future_information")
        self.assertEqual(audit.loc["team_name", "reason"], "identity_feature")

    def test_source_groups_cover_each_encoded_column_exactly_once(self) -> None:
        raw = ["round_num", "map_name", "eq_value_diff_ct"]
        encoded = [
            "round_num",
            "map_name_de_nuke",
            "map_name_de_ancient",
            "eq_value_diff_ct",
        ]

        groups = build_source_feature_groups(encoded, raw)

        self.assertEqual(set(groups), set(raw))
        flattened = [column for columns in groups.values() for column in columns]
        self.assertCountEqual(flattened, encoded)
        self.assertEqual(
            groups["map_name"],
            ["map_name_de_nuke", "map_name_de_ancient"],
        )

    def test_macro_groups_follow_the_five_m14_feature_groups(self) -> None:
        raw = [
            "map_name",
            "ct_score",
            "ct_eq_value",
            "ct_armor",
            "ct_awp",
        ]
        encoded = [
            "map_name_de_nuke",
            "map_name_de_ancient",
            "ct_score",
            "ct_eq_value",
            "ct_armor",
            "ct_awp",
        ]

        groups = build_macro_feature_groups(encoded, raw)

        self.assertEqual(
            set(groups),
            {"context", "score", "economy", "armor_utility", "weapons"},
        )
        self.assertEqual(len(groups["context"]), 2)
        self.assertEqual(groups["weapons"], ["ct_awp"])

    def test_grouped_permutation_ranks_signal_above_noise(self) -> None:
        rng = np.random.default_rng(25)
        y = rng.integers(0, 2, size=500)
        x = pd.DataFrame(
            {
                "signal_a": y,
                "signal_b": 1 - y,
                "noise": rng.normal(size=500),
            }
        )
        model = LogisticRegression(random_state=25).fit(x, y)

        result = grouped_permutation_auc_importance(
            model,
            x,
            y,
            {"signal": ["signal_a", "signal_b"], "noise": ["noise"]},
            n_repeats=8,
            seed=25,
        ).set_index("feature_group")

        self.assertGreater(
            result.loc["signal", "auc_decrease_mean"],
            result.loc["noise", "auc_decrease_mean"],
        )
        self.assertEqual(result.loc["signal", "encoded_column_count"], 2)

    def test_encoded_permutation_preserves_repeat_range(self) -> None:
        rng = np.random.default_rng(26)
        y = rng.integers(0, 2, size=400)
        x = pd.DataFrame(
            {"signal": y, "noise": rng.normal(size=len(y))}
        )
        model = LogisticRegression(random_state=26).fit(x, y)

        result = encoded_permutation_auc_importance(
            model,
            x,
            y,
            n_repeats=6,
            seed=26,
        ).set_index("feature")

        self.assertEqual(result.loc["signal", "n_repeats"], 6)
        self.assertGreater(
            result.loc["signal", "auc_decrease_mean"],
            result.loc["noise", "auc_decrease_mean"],
        )
        self.assertLessEqual(
            result.loc["signal", "auc_decrease_min"],
            result.loc["signal", "auc_decrease_mean"],
        )
        self.assertGreaterEqual(
            result.loc["signal", "auc_decrease_max"],
            result.loc["signal", "auc_decrease_mean"],
        )

    def test_lightgbm_gain_includes_zero_importance_columns_and_tree_contract(self) -> None:
        model = _FakeModel()
        bundle = {
            "model": model,
            "columns": ["signal", "noise"],
            "best_iteration": 2,
        }

        importance = lightgbm_gain_importance(bundle).set_index("feature")

        self.assertEqual(len(importance), 2)
        self.assertAlmostEqual(importance["gain_normalized"].sum(), 1.0)
        self.assertEqual(importance.loc["noise", "gain"], 0.0)
        self.assertEqual(importance.loc["signal", "split_count"], 3)
        self.assertEqual(importance.loc["signal", "deployment_tree_count"], 2)
        self.assertEqual(model.booster_.last_num_iteration, 2)

    def test_native_tree_shap_reconstructs_lightgbm_probability(self) -> None:
        model = _FakeModel()
        bundle = {
            "model": model,
            "columns": ["signal", "noise"],
            "best_iteration": 2,
        }
        x = pd.DataFrame({"signal": [0.0, 1.0], "noise": [2.0, -1.0]})

        shap_values, base_values = lightgbm_tree_shap_contributions(bundle, x)
        raw = base_values + shap_values.sum(axis=1).to_numpy()
        reconstructed = 1.0 / (1.0 + np.exp(-raw))

        np.testing.assert_allclose(
            reconstructed,
            model.predict_proba(x)[:, 1],
            rtol=0,
            atol=1e-15,
        )
        self.assertEqual(shap_values.columns.tolist(), x.columns.tolist())
        self.assertEqual(model.booster_.last_num_iteration, 2)

    def test_tree_shap_rejects_wrong_column_order(self) -> None:
        bundle = {
            "model": _FakeModel(),
            "columns": ["signal", "noise"],
            "best_iteration": 2,
        }
        x = pd.DataFrame({"noise": [0.0], "signal": [1.0]})

        with self.assertRaisesRegex(ValueError, "exactly match"):
            lightgbm_tree_shap_contributions(bundle, x)

    def test_model_importance_comparison_reports_rank_and_top_overlap(self) -> None:
        lightgbm = self._importance_table(
            ["a", "b", "c"],
            gain=[1, 2, 3],
            permutation=[1, 3, 2],
            shap=[2, 1, 3],
        )
        xgboost = self._importance_table(
            ["a", "b", "c"],
            gain=[3, 2, 1],
            permutation=[1, 2, 3],
            shap=[1, 2, 3],
        )

        detail, agreement = build_model_importance_comparison(
            lightgbm,
            xgboost,
            top_n=2,
        )
        agreement = agreement.set_index("method")

        self.assertEqual(len(detail), 3)
        self.assertAlmostEqual(agreement.loc["gain", "spearman_rank"], -1.0)
        self.assertEqual(agreement.loc["gain", "top_overlap_count"], 1)
        self.assertAlmostEqual(agreement.loc["gain", "top_jaccard"], 1 / 3)
        self.assertIn("gain_rank_difference_lgbm_minus_xgb", detail.columns)

    def test_model_importance_comparison_rejects_feature_set_mismatch(self) -> None:
        lightgbm = self._importance_table(["a", "b"], [1, 2], [1, 2], [1, 2])
        xgboost = self._importance_table(["a", "c"], [1, 2], [1, 2], [1, 2])

        with self.assertRaisesRegex(ValueError, "feature sets"):
            build_model_importance_comparison(lightgbm, xgboost)

    def test_m12_loader_tolerates_only_csv_scale_mean_rank_rounding(self) -> None:
        gain = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "gain_rank": [1, 2, 3],
                "gain_normalized": [0.6, 0.3, 0.1],
            }
        )
        permutation = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "permutation_rank": [1, 3, 2],
                "auc_decrease_mean": [0.2, 0.0, 0.1],
                "auc_decrease_std": [0.01, 0.01, 0.01],
            }
        )
        shap = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "shap_rank": [2, 1, 3],
                "mean_abs_shap": [0.3, 0.4, 0.1],
            }
        )
        saved = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "gain_rank": [1, 2, 3],
                "gain_normalized": [0.6, 0.3, 0.1],
                "permutation_rank": [1, 3, 2],
                "auc_decrease_mean": [0.2, 0.0, 0.1],
                "auc_decrease_std": [0.01, 0.01, 0.01],
                "shap_rank": [2, 1, 3],
                "mean_abs_shap": [0.3, 0.4, 0.1],
                "mean_rank": [4 / 3 + 5e-15, 2.0, 8 / 3],
            }
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gain.to_csv(root / "gain_importance.csv", index=False)
            permutation.to_csv(
                root / "permutation_importance_auc.csv", index=False
            )
            shap.to_csv(root / "shap_importance.csv", index=False)
            saved.to_csv(root / "importance_comparison.csv", index=False)

            rebuilt, _ = load_m12_xgboost_importance(root)

        self.assertEqual(len(rebuilt), 3)

    def test_prediction_replay_joins_complete_keys_not_row_order(self) -> None:
        rows = pd.DataFrame(
            {
                "series_id": ["s1", "s1", "s2", "s2"],
                "game_id": ["g1", "g1", "g2", "g2"],
                "round_id": ["r1", "r2", "r1", "r2"],
                "y_true": [1, 0, 1, 0],
                "ct_win_probability": [0.8, 0.2, 0.7, 0.1],
            }
        )
        saved = rows.sample(frac=1.0, random_state=25).reset_index(drop=True)
        expected_metrics = {
            "accuracy": 1.0,
            "auc": 1.0,
            "log_loss": 0.22708064055624455,
            "brier_score": 0.045,
            "ece10": 0.2,
        }

        audit = audit_frozen_prediction_replay(
            saved,
            rows,
            expected_metrics,
            tolerance=1e-12,
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["key_mismatch_count"], 0)
        self.assertLessEqual(audit["metric_max_absolute_difference"], 1e-12)

    def test_prediction_replay_rejects_duplicate_complete_key(self) -> None:
        rows = pd.DataFrame(
            {
                "series_id": ["s1", "s1"],
                "game_id": ["g1", "g1"],
                "round_id": ["r1", "r1"],
                "y_true": [0, 1],
                "ct_win_probability": [0.2, 0.8],
            }
        )

        audit = audit_frozen_prediction_replay(
            rows,
            rows,
            {
                "accuracy": 1.0,
                "auc": 1.0,
                "log_loss": 0.2,
                "brier_score": 0.04,
                "ece10": 0.2,
            },
        )

        self.assertFalse(audit["passed"])
        self.assertGreater(audit["saved_duplicate_key_rows"], 0)

    def test_acceptance_requires_every_blocking_check(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        passed = decide_acceptance(checks)
        checks["shap_reconstruction"] = False
        failed = decide_acceptance(checks)

        self.assertEqual(passed["status"], "passed")
        self.assertTrue(passed["ready_for_m26"])
        self.assertEqual(failed["blocking_failures"], ["shap_reconstruction"])
        self.assertFalse(failed["ready_for_m26"])

    @staticmethod
    def _importance_table(features, gain, permutation, shap) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "feature": features,
                "gain_rank": gain,
                "permutation_rank": permutation,
                "shap_rank": shap,
            }
        )
        frame["mean_rank"] = frame[
            ["gain_rank", "permutation_rank", "shap_rank"]
        ].mean(axis=1)
        return frame


if __name__ == "__main__":
    unittest.main()

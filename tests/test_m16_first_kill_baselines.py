import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.csdemo.m16_first_kill_baselines import (
    FIRST_KILL_MODEL_FEATURES,
    FORMAL_MODEL_NAMES,
    REDUNDANT_FIRST_KILL_FEATURES,
    assess_metric_targets,
    audit_training_data,
    build_feature_contract,
    build_prediction_table,
    canonical_feature_names,
    compare_external_models,
    decide_acceptance,
    fingerprint_file,
    make_formal_models,
    prepare_profile_splits,
    verify_m15_artifact,
)
from src.csdemo.schema import ID_COLUMNS, PRE_ROUND_FEATURES


def split_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_id": "train-series",
                "game_id": "train-game",
                "round_id": "train-round-1",
                "split": "train",
                "ct_win": 1,
                "map_name": "de_mirage",
                "first_kill_weapon": "AK-47",
                "first_kill_advantage_ct": 1,
            },
            {
                "series_id": "train-series",
                "game_id": "train-game",
                "round_id": "train-round-2",
                "split": "train",
                "ct_win": 0,
                "map_name": "de_mirage",
                "first_kill_weapon": "M4A4",
                "first_kill_advantage_ct": -1,
            },
            {
                "series_id": "val-series",
                "game_id": "val-game",
                "round_id": "val-round-1",
                "split": "val",
                "ct_win": 1,
                "map_name": "de_inferno",
                "first_kill_weapon": "AWP",
                "first_kill_advantage_ct": 1,
            },
            {
                "series_id": "test-series",
                "game_id": "test-game",
                "round_id": "test-round-1",
                "split": "test",
                "ct_win": 0,
                "map_name": "de_ancient",
                "first_kill_weapon": "Zeus x27",
                "first_kill_advantage_ct": -1,
            },
        ]
    )


class M16FirstKillBaselineTests(unittest.TestCase):
    def test_canonical_profile_contains_one_side_signal_without_redundancy(self) -> None:
        features = canonical_feature_names()

        self.assertEqual(features[: len(PRE_ROUND_FEATURES)], PRE_ROUND_FEATURES)
        self.assertEqual(features[-4:], FIRST_KILL_MODEL_FEATURES)
        self.assertEqual(len(features), len(set(features)))
        self.assertFalse(set(features) & set(REDUNDANT_FIRST_KILL_FEATURES))
        self.assertFalse(set(features) & set(ID_COLUMNS + ["ct_win", "split"]))

    def test_profile_preparation_uses_only_training_categories_and_aligns_columns(self) -> None:
        features = ["map_name", "first_kill_weapon", "first_kill_advantage_ct"]

        prepared = prepare_profile_splits(split_fixture(), features)

        train_columns = prepared["train"][0].columns.tolist()
        self.assertEqual(prepared["val"][0].columns.tolist(), train_columns)
        self.assertEqual(prepared["test"][0].columns.tolist(), train_columns)
        self.assertIn("first_kill_weapon_AK-47", train_columns)
        self.assertNotIn("first_kill_weapon_Zeus x27", train_columns)
        self.assertFalse(set(ID_COLUMNS) & set(train_columns))

    def test_training_data_audit_rejects_cross_split_series_and_duplicate_keys(self) -> None:
        frame = split_fixture()
        duplicate = frame.iloc[[0]].copy()
        duplicate["split"] = "test"
        broken = pd.concat([frame, duplicate], ignore_index=True)

        result = audit_training_data(broken)

        self.assertFalse(result["passed"])
        self.assertEqual(result["duplicate_key_rows"], 1)
        self.assertEqual(result["cross_split_series"], 1)

    def test_formal_xgboost_is_the_frozen_untuned_first_kill_model(self) -> None:
        models = make_formal_models()
        params = models["xgboost_untuned"].get_params()

        self.assertEqual(tuple(models), FORMAL_MODEL_NAMES)
        self.assertEqual(params["n_estimators"], 500)
        self.assertEqual(params["max_depth"], 4)
        self.assertEqual(params["learning_rate"], 0.03)
        self.assertEqual(params["subsample"], 0.85)
        self.assertEqual(params["colsample_bytree"], 0.85)
        self.assertIsNone(params.get("early_stopping_rounds"))

    def test_metric_assessment_respects_high_and_low_directions(self) -> None:
        result = assess_metric_targets(
            {
                "accuracy": 0.68,
                "auc": 0.75,
                "log_loss": 0.58,
                "brier_score": 0.20,
            }
        )

        self.assertTrue(result["all_minimum_passed"])
        self.assertFalse(result["all_stage_passed"])
        self.assertAlmostEqual(result["metrics"]["auc"]["stage_gap"], 0.03)
        self.assertAlmostEqual(result["metrics"]["log_loss"]["stage_gap"], 0.03)

    def test_prediction_table_keeps_keys_and_validates_every_model_probability(self) -> None:
        test_rows = split_fixture().loc[lambda frame: frame["split"].eq("test")]
        probabilities = {
            "constant_train_prior": np.array([0.54]),
            "logistic_regression": np.array([0.30]),
            "xgboost_untuned": np.array([0.25]),
        }

        result = build_prediction_table(test_rows, probabilities)

        self.assertEqual(result.loc[0, "round_id"], "test-round-1")
        self.assertEqual(result.loc[0, "ct_win"], 0)
        self.assertEqual(result.loc[0, "xgboost_untuned_probability"], 0.25)
        self.assertEqual(result.loc[0, "xgboost_untuned_prediction"], 0)

        probabilities["xgboost_untuned"] = np.array([1.2])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_prediction_table(test_rows, probabilities)

    def test_feature_contract_records_canonical_and_redundant_columns(self) -> None:
        contract = build_feature_contract()
        indexed = contract.set_index("feature")

        self.assertTrue(indexed.loc["first_kill_advantage_ct", "included"])
        self.assertEqual(indexed.loc["first_kill_advantage_ct", "group"], "first_kill")
        self.assertFalse(indexed.loc["first_kill_is_ct", "included"])
        self.assertEqual(indexed.loc["first_kill_is_ct", "reason"], "deterministic_redundancy")

    def test_external_metrics_map_to_the_declared_local_model(self) -> None:
        local = pd.DataFrame(
            [
                {
                    "model": "logistic_regression",
                    "split": "test",
                    "accuracy": 0.70,
                    "auc": 0.78,
                },
                {
                    "model": "xgboost_untuned",
                    "split": "test",
                    "accuracy": 0.72,
                    "auc": 0.80,
                },
            ]
        )
        external = pd.DataFrame(
            [
                {
                    "benchmark_id": "external_logistic_auc",
                    "current_model": "logistic_regression",
                    "metric": "auc",
                    "reported_value": 0.76,
                    "direction": "higher",
                },
                {
                    "benchmark_id": "external_xgb_auc",
                    "current_model": "xgboost_untuned",
                    "metric": "auc",
                    "reported_value": 0.79,
                    "direction": "higher",
                },
            ]
        )

        result = compare_external_models(local, external).set_index("benchmark_id")

        self.assertEqual(result.loc["external_logistic_auc", "current_value"], 0.78)
        self.assertAlmostEqual(
            result.loc["external_logistic_auc", "raw_difference_ours_minus_reported"],
            0.02,
        )
        self.assertEqual(result.loc["external_xgb_auc", "current_value"], 0.80)

    def test_m15_artifact_verification_rejects_a_hash_change(self) -> None:
        with TemporaryDirectory() as directory:
            data_path = Path(directory) / "first_kill.parquet"
            data_path.write_bytes(b"accepted-data")
            summary = {"data_artifact": fingerprint_file(data_path)}

            accepted = verify_m15_artifact(data_path, summary)
            data_path.write_bytes(b"changed-data")
            rejected = verify_m15_artifact(data_path, summary)

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])
        self.assertNotEqual(rejected["actual_sha256"], rejected["expected_sha256"])

    def test_acceptance_reports_the_exact_blocking_failure(self) -> None:
        checks = {
            "m15_artifact": True,
            "data_contract": True,
            "feature_contract": True,
            "model_probabilities": True,
            "frozen_xgboost": True,
            "minimum_metrics": False,
            "automated_tests": True,
            "external_report": True,
        }

        result = decide_acceptance(checks)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["blocking_failures"], ["minimum_metrics"])
        self.assertFalse(result["ready_for_m17"])


if __name__ == "__main__":
    unittest.main()

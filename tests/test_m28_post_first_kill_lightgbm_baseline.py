import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.csdemo.m16_first_kill_baselines import canonical_feature_names
from src.csdemo.m28_post_first_kill_lightgbm_baseline import (
    BLOCKING_CHECKS,
    BOOTSTRAP_SAMPLES,
    LIGHTGBM_BASE_PARAMS,
    assess_metric_targets,
    audit_data_contract,
    build_feature_contract,
    build_prediction_table,
    decide_acceptance,
    fit_lightgbm,
    make_lightgbm_model,
    model_metric_differences,
    paired_model_bootstrap,
    prepare_first_kill_splits,
    replay_frozen_xgboost,
    run,
)
from src.csdemo.schema import ID_COLUMNS


def _row(
    series_id: str,
    game_id: str,
    round_id: str,
    split: str,
    label: int,
    map_name: str,
    weapon: str,
) -> dict:
    row = {feature: 0 for feature in canonical_feature_names()}
    row.update(
        {
            "series_id": series_id,
            "game_id": game_id,
            "round_id": round_id,
            "split": split,
            "ct_win": label,
            "map_name": map_name,
            "round_num": 1,
            "first_kill_advantage_ct": 1 if label else -1,
            "first_kill_time": 12.0,
            "first_kill_headshot": label,
            "first_kill_weapon": weapon,
        }
    )
    return row


def split_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(
                "s-train",
                "g-train",
                "r-train-1",
                "train",
                0,
                "de_mirage",
                "AK-47",
            ),
            _row(
                "s-train",
                "g-train",
                "r-train-2",
                "train",
                1,
                "de_nuke",
                "M4A1-S",
            ),
            _row(
                "s-val",
                "g-val",
                "r-val",
                "val",
                1,
                "de_inferno",
                "AWP",
            ),
            _row(
                "s-test",
                "g-test",
                "r-test",
                "test",
                0,
                "de_ancient",
                "USP-S",
            ),
        ]
    )


class RecordingEstimator:
    def __init__(self) -> None:
        self.args = None
        self.kwargs = None

    def fit(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self


class FrozenEstimator:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.predict_calls = 0

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.predict_calls += 1
        self.last_columns = features.columns.tolist()
        return np.column_stack([1.0 - self.probabilities, self.probabilities])

    def fit(self, *_args, **_kwargs):
        raise AssertionError("The frozen M21 XGBoost model must never be fitted")


class M28PostFirstKillLightGBMBaselineTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_feature_contract_is_exact_m21_canonical_profile(self) -> None:
        contract = build_feature_contract()
        included = contract.loc[contract["included"], "feature"].tolist()
        excluded = contract.loc[~contract["included"], "feature"].tolist()

        self.assertEqual(included, canonical_feature_names())
        self.assertEqual(len(included), 40)
        self.assertEqual(len(excluded), 5)
        self.assertFalse(set(included) & set(ID_COLUMNS + ["ct_win", "split"]))

    def test_preparation_learns_map_and_weapon_categories_from_train_only(self) -> None:
        prepared = prepare_first_kill_splits(split_fixture())
        train_columns = prepared["train"][0].columns.tolist()

        self.assertIn("map_name_de_mirage", train_columns)
        self.assertIn("first_kill_weapon_AK-47", train_columns)
        self.assertNotIn("map_name_de_ancient", train_columns)
        self.assertNotIn("first_kill_weapon_USP-S", train_columns)
        self.assertEqual(prepared["val"][0].columns.tolist(), train_columns)
        self.assertEqual(prepared["test"][0].columns.tolist(), train_columns)

    def test_data_audit_rejects_duplicate_keys_and_cross_split_series(self) -> None:
        frame = split_fixture()
        duplicate = frame.iloc[[0]].copy()
        duplicate["split"] = "test"
        broken = pd.concat([frame, duplicate], ignore_index=True)

        result = audit_data_contract(broken)

        self.assertFalse(result["passed"])
        self.assertEqual(result["duplicate_key_rows"], 1)
        self.assertEqual(result["cross_split_series"], 1)

    def test_lightgbm_parameters_and_bootstrap_protocol_are_frozen(self) -> None:
        model = make_lightgbm_model()
        params = model.get_params()

        for name, expected in LIGHTGBM_BASE_PARAMS.items():
            self.assertEqual(params[name], expected, name)
        self.assertEqual(params["device_type"], "cpu")
        self.assertEqual(params["random_state"], 42)
        self.assertEqual(BOOTSTRAP_SAMPLES, 2000)

    def test_fit_uses_train_and_validation_without_test(self) -> None:
        prepared = prepare_first_kill_splits(split_fixture())
        estimator = RecordingEstimator()

        fitted = fit_lightgbm(estimator, prepared)

        self.assertIs(fitted, estimator)
        self.assertIs(estimator.args[0], prepared["train"][0])
        self.assertIs(estimator.args[1], prepared["train"][1])
        self.assertEqual(len(estimator.kwargs["eval_set"]), 1)
        self.assertIs(estimator.kwargs["eval_set"][0][0], prepared["val"][0])
        self.assertIs(estimator.kwargs["eval_set"][0][1], prepared["val"][1])
        self.assertIsNot(estimator.kwargs["eval_set"][0][0], prepared["test"][0])
        self.assertEqual(estimator.kwargs["eval_metric"], "binary_logloss")

    def test_frozen_xgboost_replay_aligns_complete_keys_without_fit(self) -> None:
        test_rows = pd.DataFrame(
            [
                {"series_id": "s1", "game_id": "g1", "round_id": "r1", "ct_win": 1},
                {"series_id": "s2", "game_id": "g2", "round_id": "r2", "ct_win": 0},
            ]
        )
        test_x = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        estimator = FrozenEstimator([0.8, 0.2])
        saved = pd.DataFrame(
            [
                {"series_id": "s2", "game_id": "g2", "round_id": "r2", "ct_win": 0, "xgboost_tuned_probability": 0.2},
                {"series_id": "s1", "game_id": "g1", "round_id": "r1", "ct_win": 1, "xgboost_tuned_probability": 0.8},
            ]
        )

        probability, audit = replay_frozen_xgboost(
            {"model": estimator, "columns": ["a", "b"]},
            test_x,
            test_rows,
            saved,
        )

        np.testing.assert_allclose(probability, [0.8, 0.2])
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["key_mismatch_count"], 0)
        self.assertEqual(estimator.predict_calls, 1)

    def test_metric_targets_and_differences_respect_metric_direction(self) -> None:
        assessment = assess_metric_targets(
            {
                "accuracy": 0.72,
                "auc": 0.79,
                "log_loss": 0.54,
                "brier_score": 0.18,
                "ece10": 0.02,
            }
        )
        comparison = pd.DataFrame(
            [
                {"model": "lightgbm_baseline", "split": "test", "accuracy": 0.75, "auc": 0.81, "log_loss": 0.52, "brier_score": 0.17, "ece10": 0.02},
                {"model": "xgboost_frozen", "split": "test", "accuracy": 0.74, "auc": 0.80, "log_loss": 0.53, "brier_score": 0.18, "ece10": 0.03},
            ]
        )

        differences = model_metric_differences(comparison)

        self.assertTrue(assessment["all_minimum_passed"])
        self.assertTrue(assessment["all_stage_passed"])
        self.assertAlmostEqual(
            differences["auc"]["performance_advantage_lightgbm"], 0.01
        )
        self.assertAlmostEqual(
            differences["log_loss"]["performance_advantage_lightgbm"], 0.01
        )

    def test_paired_bootstrap_uses_five_metrics_and_complete_series(self) -> None:
        predictions = pd.DataFrame(
            {
                "series_id": ["s1", "s1", "s2", "s2"],
                "game_id": ["g1", "g1", "g2", "g2"],
                "round_id": ["r1", "r2", "r3", "r4"],
                "y_true": [0, 1, 0, 1],
                "lightgbm_probability": [0.2, 0.8, 0.3, 0.7],
                "xgboost_probability": [0.3, 0.7, 0.4, 0.6],
            }
        )

        result = paired_model_bootstrap(predictions, n_bootstrap=20, seed=42)

        self.assertEqual(result["metric"].nunique(), 5)
        self.assertTrue(result["successful_bootstraps"].eq(20).all())
        self.assertTrue(result["bootstrap_unit"].eq("series_id_paired").all())

    def test_acceptance_does_not_require_lightgbm_to_win(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        accepted = decide_acceptance(checks)
        checks["paired_uncertainty"] = False
        rejected = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["ready_for_m29"])
        self.assertNotIn("lightgbm_beats_xgboost", BLOCKING_CHECKS)
        self.assertEqual(rejected["blocking_failures"], ["paired_uncertainty"])

    def test_real_runner_writes_controlled_baseline_and_uncertainty(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            summary = run(
                project_root=self.root,
                model_dir=temporary / "models",
                report_dir=temporary / "reports",
                n_bootstrap=20,
                run_tests=False,
                run_compile=False,
            )

            self.assertEqual(summary["acceptance"]["status"], "passed")
            self.assertEqual(summary["data"]["rows"], 41027)
            self.assertEqual(summary["features"]["raw_count"], 40)
            self.assertEqual(summary["features"]["encoded_count"], 82)
            self.assertEqual(summary["xgboost_fit_calls"], 0)
            self.assertEqual(summary["paired_uncertainty"]["metric_count"], 5)
            self.assertLessEqual(
                summary["xgboost_replay"]["max_absolute_probability_difference"],
                1e-7,
            )
            for filename in (
                "m28_summary.json",
                "m28_checks.csv",
                "m28_experiment_manifest.json",
                "m28_post_first_kill_lightgbm_controlled_baseline_report.md",
                "m28_model_comparison.csv",
                "m28_test_predictions.csv",
                "global_bootstrap_95ci.csv",
                "paired_lightgbm_vs_xgboost_bootstrap.csv",
                "feature_contract.csv",
                "encoded_feature_columns.csv",
                "lightgbm_training_history.csv",
                "external_benchmark_comparison.csv",
                "external_benchmark_comparison.md",
            ):
                self.assertTrue((temporary / "reports" / filename).is_file(), filename)
            self.assertTrue(
                (
                    temporary
                    / "models/post_first_kill_lightgbm_baseline.joblib"
                ).is_file()
            )
            manifest = json.loads(
                (temporary / "reports/m28_experiment_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("m28_module", manifest["inputs"])
            self.assertIn("m28_tests", manifest["inputs"])
            for artifact in [*manifest["inputs"].values(), *manifest["outputs"].values()]:
                path = Path(artifact["path"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )

    def test_prediction_table_rejects_invalid_probability(self) -> None:
        test_rows = split_fixture().loc[lambda frame: frame["split"].eq("test")]
        probabilities = {
            "xgboost_frozen": np.array([0.4]),
            "lightgbm_baseline": np.array([0.3]),
        }

        result = build_prediction_table(test_rows, probabilities)

        self.assertEqual(result.loc[0, "round_id"], "r-test")
        probabilities["lightgbm_baseline"] = np.array([1.2])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_prediction_table(test_rows, probabilities)


if __name__ == "__main__":
    unittest.main()

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.csdemo.m24_pre_round_lightgbm_evaluation import (
    BLOCKING_CHECKS,
    KEY_COLUMNS,
    assess_global_intervals,
    assess_group_robustness,
    audit_calibration_protocol,
    audit_prediction_replay,
    decide_acceptance,
    paired_model_bootstrap,
    prepare_analysis_table,
    replay_frozen_model,
    verify_m23_prerequisite,
)
from src.csdemo.m22_pre_round_lightgbm_baseline import prepare_pre_round_splits
from src.csdemo.schema import PRE_ROUND_FEATURES


class M24PreRoundLightGBMEvaluationTests(unittest.TestCase):
    def test_analysis_join_uses_complete_key_for_features_and_first_kill(self) -> None:
        predictions = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "game_id": ["lan:g1", "online:g2"],
                "round_id": ["same-round", "same-round"],
                "y_true": [1, 0],
                "ct_win_probability": [0.8, 0.3],
                "predicted_label": [1, 0],
                "map_name": ["stale_map", "stale_map"],
                "round_num": [99, 99],
            }
        )
        features = pd.DataFrame(
            {
                "series_id": ["s2", "s1"],
                "game_id": ["online:g2", "lan:g1"],
                "round_id": ["same-round", "same-round"],
                "split": ["test", "test"],
                "ct_win": [0, 1],
                "map_name": ["de_nuke", "de_inferno"],
                "round_num": [12, 3],
                "eq_value_diff_ct": [-2000, 3000],
                "ct_eq_value": [18000, 25000],
                "t_eq_value": [20000, 22000],
                "rifle_diff_ct": [-1, 1],
                "awp_diff_ct": [0, 1],
            }
        )
        kills = pd.DataFrame(
            {
                "series_id": ["s2", "s1"],
                "game_id": ["online:g2", "lan:g1"],
                "round_id": ["same-round", "same-round"],
                "is_first_kill": [1, 1],
                "killer_side": ["T", "CT"],
                "victim_side": ["CT", "T"],
                "weapon": ["AWP", "AK-47"],
                "headshot": [0, 1],
                "time": [35.0, 10.0],
            }
        )

        analysis = prepare_analysis_table(predictions, features, kills)

        self.assertEqual(list(KEY_COLUMNS), ["series_id", "game_id", "round_id"])
        by_game = analysis.set_index("game_id")
        self.assertEqual(by_game.loc["lan:g1", "map_name"], "de_inferno")
        self.assertEqual(by_game.loc["online:g2", "map_name"], "de_nuke")
        self.assertEqual(by_game.loc["lan:g1", "first_kill_side"], "CT")
        self.assertEqual(by_game.loc["online:g2", "first_kill_side"], "T")
        self.assertEqual(by_game.loc["lan:g1", "source_subset"], "lan")

    def test_analysis_join_rejects_incomplete_feature_key_set(self) -> None:
        predictions = pd.DataFrame(
            {
                "series_id": ["s1"],
                "game_id": ["lan:g1"],
                "round_id": ["r1"],
                "y_true": [1],
                "ct_win_probability": [0.8],
                "predicted_label": [1],
            }
        )
        features = pd.DataFrame(
            {
                "series_id": ["s1"],
                "game_id": ["lan:g1"],
                "round_id": ["different"],
                "split": ["test"],
                "ct_win": [1],
                "map_name": ["de_nuke"],
                "round_num": [1],
                "eq_value_diff_ct": [0],
                "ct_eq_value": [20000],
                "t_eq_value": [20000],
                "rifle_diff_ct": [0],
                "awp_diff_ct": [0],
            }
        )
        kills = pd.DataFrame(
            columns=[
                *KEY_COLUMNS,
                "is_first_kill",
                "killer_side",
                "victim_side",
                "weapon",
                "headshot",
                "time",
            ]
        )

        with self.assertRaisesRegex(ValueError, "complete key set"):
            prepare_analysis_table(predictions, features, kills)

    def test_prediction_replay_aligns_complete_keys_and_detects_drift(self) -> None:
        saved = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "game_id": ["lan:g1", "online:g2"],
                "round_id": ["r1", "r2"],
                "ct_win": [0, 1],
                "lightgbm_tuned_probability": [0.2, 0.8],
            }
        )
        replayed = saved[list(KEY_COLUMNS)].iloc[::-1].reset_index(drop=True)
        replayed["y_true"] = [1, 0]
        replayed["ct_win_probability"] = [0.8, 0.2]

        result = audit_prediction_replay(saved, replayed, tolerance=1e-12)

        self.assertTrue(result["passed"])
        self.assertEqual(result["key_mismatch_count"], 0)
        self.assertEqual(result["label_mismatch_count"], 0)
        broken = replayed.copy()
        broken.loc[0, "ct_win_probability"] += 1e-5
        self.assertFalse(
            audit_prediction_replay(saved, broken, tolerance=1e-12)["passed"]
        )

    def test_m23_prerequisite_rejects_model_or_task_drift(self) -> None:
        class Predictor:
            def predict_proba(self, values):
                return values

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "pre_round.parquet"
            model_path = root / "pre_round_lightgbm_tuned.joblib"
            data_path.write_bytes(b"frozen-data")
            model_path.write_bytes(b"frozen-model")
            data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            params = {"random_state": 42, "num_leaves": 15}
            bundle = {
                "model": Predictor(),
                "task": "pre_round",
                "model_name": "lightgbm_tuned",
                "raw_features": list(PRE_ROUND_FEATURES),
                "columns": [f"feature_{index}" for index in range(43)],
                "params": params,
                "data_sha256": data_sha,
            }
            summary = {
                "acceptance": {"status": "passed", "ready_for_m24": True},
                "data": {"sha256": data_sha},
                "features": {"raw_count": 36, "encoded_count": 43},
                "model": {
                    "params": params,
                    "model_artifact": {"sha256": model_sha},
                },
            }

            result = verify_m23_prerequisite(
                data_path, model_path, summary, bundle
            )

            self.assertTrue(result["passed"])
            drifted = {**bundle, "task": "post_first_kill"}
            self.assertFalse(
                verify_m23_prerequisite(
                    data_path, model_path, summary, drifted
                )["passed"]
            )

    def test_frozen_replay_predicts_validation_and_test_without_fit(self) -> None:
        class FrozenPredictor:
            def __init__(self) -> None:
                self.predict_calls = 0
                self.fit_calls = 0

            def predict_proba(self, values):
                self.predict_calls += 1
                probability = np.linspace(0.25, 0.75, len(values))
                return np.column_stack([1.0 - probability, probability])

            def fit(self, *args, **kwargs):
                self.fit_calls += 1
                raise AssertionError("M24 must not fit the frozen model")

        rows = []
        for index, split in enumerate(("train", "train", "val", "val", "test", "test")):
            row = {
                "series_id": f"s{index}",
                "game_id": f"lan:g{index}",
                "round_id": f"r{index}",
                "split": split,
                "ct_win": index % 2,
            }
            for feature in PRE_ROUND_FEATURES:
                row[feature] = "de_nuke" if feature == "map_name" else 0
            row["round_num"] = index + 1
            rows.append(row)
        data = pd.DataFrame(rows)
        encoded = prepare_pre_round_splits(data)["train"][0].columns.tolist()
        predictor = FrozenPredictor()
        bundle = {
            "model": predictor,
            "raw_features": list(PRE_ROUND_FEATURES),
            "columns": encoded,
        }

        outputs, audit = replay_frozen_model(data, bundle)

        self.assertEqual(set(outputs), {"val", "test"})
        self.assertEqual(audit["lightgbm_fit_calls"], 0)
        self.assertEqual(predictor.fit_calls, 0)
        self.assertEqual(predictor.predict_calls, 2)
        self.assertEqual(len(outputs["test"]), 2)
        self.assertNotIn("ct_win", outputs["test"].columns)

    def test_paired_bootstrap_uses_performance_direction_and_complete_series(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": np.repeat([f"s{i}" for i in range(8)], 4),
                "game_id": np.repeat([f"lan:g{i}" for i in range(8)], 4),
                "round_id": [f"r{i}" for i in range(32)],
                "y_true": np.tile([0, 0, 1, 1], 8),
                "lightgbm_probability": np.tile([0.1, 0.2, 0.8, 0.9], 8),
                "xgboost_probability": np.tile([0.6, 0.4, 0.6, 0.4], 8),
            }
        )

        result = paired_model_bootstrap(frame, n_bootstrap=40, seed=7)

        self.assertEqual(len(result), 5)
        self.assertTrue(result["successful_bootstraps"].eq(40).all())
        self.assertTrue(result["bootstrap_unit"].eq("series_id_paired").all())
        by_metric = result.set_index("metric")
        self.assertGreater(by_metric.loc["accuracy", "performance_advantage_lightgbm"], 0)
        self.assertLess(by_metric.loc["log_loss", "raw_difference_lightgbm_minus_xgboost"], 0)
        self.assertGreater(by_metric.loc["log_loss", "performance_advantage_lightgbm"], 0)

    def test_global_interval_assessment_separates_minimum_and_stage_targets(self) -> None:
        intervals = pd.DataFrame(
            {
                "metric": ["accuracy", "auc", "log_loss", "brier_score", "ece10"],
                "ci_lower_95": [0.63, 0.711, 0.58, 0.19, 0.01],
                "ci_upper_95": [0.67, 0.74, 0.604, 0.21, 0.04],
                "successful_bootstraps": [2000] * 5,
            }
        )

        result = assess_global_intervals(intervals, n_bootstrap=2000)

        self.assertTrue(result["bootstrap_complete"])
        self.assertTrue(result["minimum_passed"])
        self.assertTrue(result["stage_passed"])
        intervals.loc[intervals["metric"].eq("auc"), "ci_lower_95"] = 0.699
        self.assertFalse(
            assess_global_intervals(intervals, n_bootstrap=2000)["minimum_passed"]
        )

    def test_group_assessment_requires_four_outputs_but_not_stage_targets(self) -> None:
        map_table = pd.DataFrame(
            {
                "map_name": ["de_nuke"],
                "rounds": [400],
                "auc": [0.68],
                "auc_ci_lower_95": [0.65],
            }
        )
        grouped = {
            "map": map_table,
            "source": pd.DataFrame({"source_subset": ["lan"], "rounds": [400]}),
            "round_stage": pd.DataFrame({"round_stage": ["early"], "rounds": [400]}),
            "equipment_band": pd.DataFrame({"equipment_band": ["even"], "rounds": [400]}),
        }
        source_gap = {"absolute_difference": 0.03, "ci_includes_zero": True}

        result = assess_group_robustness(grouped, source_gap)

        self.assertTrue(result["outputs_complete"])
        self.assertTrue(result["large_map_minimum_passed"])
        self.assertFalse(result["large_map_stage_passed"])
        self.assertTrue(result["source_gap_passed"])
        self.assertFalse(
            assess_group_robustness(
                {name: table for name, table in grouped.items() if name != "source"},
                source_gap,
            )["outputs_complete"]
        )

    def test_calibration_protocol_is_validation_only_and_grouped(self) -> None:
        validation = pd.DataFrame(
            {
                "series_id": ["s1", "s1", "s2", "s2"],
                "game_id": ["lan:g1", "lan:g1", "online:g2", "online:g2"],
                "round_id": ["r1", "r2", "r3", "r4"],
                "y_true": [0, 1, 0, 1],
                "ct_win_probability": [0.2, 0.8, 0.3, 0.7],
            }
        )
        comparison = pd.DataFrame(
            {
                "method": ["uncalibrated", "sigmoid", "isotonic"],
                "log_loss": [0.50, 0.51, 0.52],
                "brier_score": [0.16, 0.17, 0.18],
            }
        )
        oof = validation[list(KEY_COLUMNS) + ["y_true"]].copy()
        oof["fold"] = [0, 0, 1, 1]
        for method in ("uncalibrated", "sigmoid", "isotonic"):
            oof[f"probability_{method}"] = validation["ct_win_probability"]

        result = audit_calibration_protocol(
            validation,
            comparison,
            oof,
            "uncalibrated",
            n_splits=2,
        )

        self.assertTrue(result["passed"])
        leaked = comparison.assign(test_log_loss=[0.4, 0.3, 0.2])
        self.assertFalse(
            audit_calibration_protocol(
                validation,
                leaked,
                oof,
                "uncalibrated",
                n_splits=2,
            )["passed"]
        )

    def test_acceptance_requires_every_blocker_but_not_model_superiority(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        accepted = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["ready_for_m25"])
        self.assertNotIn("lightgbm_significantly_better", BLOCKING_CHECKS)
        checks["paired_comparison"] = False
        rejected = decide_acceptance(checks)
        self.assertEqual(rejected["blocking_failures"], ["paired_comparison"])

    def test_reproduction_script_freezes_m23_inputs_and_bootstrap_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "run_pre_round_lightgbm_evaluation.ps1"

        source = script.read_text(encoding="utf-8")

        self.assertIn("src.csdemo.m24_pre_round_lightgbm_evaluation", source)
        self.assertIn("pre_round_lightgbm_tuned.joblib", source)
        self.assertIn("m23_summary.json", source)
        self.assertIn("[int]$BootstrapSamples = 2000", source)


if __name__ == "__main__":
    unittest.main()

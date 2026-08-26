import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.csdemo.m30_post_first_kill_lightgbm_evaluation import (
    BLOCKING_CHECKS,
    assess_global_intervals,
    assess_group_robustness,
    audit_calibration_protocol,
    audit_prediction_replay,
    audit_reproduction_entrypoint,
    build_paired_prediction_table,
    decide_acceptance,
    format_source_auc_gap,
    paired_model_bootstrap,
    replay_frozen_model,
    verify_m29_prerequisite,
)
from src.csdemo.m16_first_kill_baselines import canonical_feature_names, prepare_profile_splits


class M30PostFirstKillLightGBMEvaluationTests(unittest.TestCase):
    def test_m29_prerequisite_rejects_task_or_model_drift(self) -> None:
        class Predictor:
            def predict_proba(self, values):
                return values

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "first_kill.parquet"
            model_path = root / "post_first_kill_lightgbm_tuned.joblib"
            data_path.write_bytes(b"frozen-data")
            model_path.write_bytes(b"frozen-model")
            data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            params = {"random_state": 42, "max_depth": 3}
            bundle = {
                "model": Predictor(),
                "task": "post_first_kill",
                "model_name": "lightgbm_tuned",
                "raw_features": canonical_feature_names(),
                "columns": [f"feature_{index}" for index in range(82)],
                "params": params,
                "data_sha256": data_sha,
            }
            summary = {
                "acceptance": {"status": "passed", "ready_for_m30": True},
                "data": {"sha256": data_sha},
                "features": {"raw_count": 40, "encoded_count": 82},
                "model": {
                    "params": params,
                    "model_artifact": {"sha256": model_sha},
                },
            }

            accepted = verify_m29_prerequisite(data_path, model_path, summary, bundle)
            rejected = verify_m29_prerequisite(
                data_path, model_path, summary, {**bundle, "task": "pre_round"}
            )

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])

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
                raise AssertionError("M30 must not fit the frozen LightGBM")

        rows = []
        raw_features = canonical_feature_names()
        for index, split in enumerate(("train", "train", "val", "val", "test", "test")):
            row = {
                "series_id": f"s{index}",
                "game_id": f"lan:g{index}",
                "round_id": f"r{index}",
                "split": split,
                "ct_win": index % 2,
            }
            for feature in raw_features:
                if feature == "map_name":
                    row[feature] = "de_nuke"
                elif feature == "first_kill_weapon":
                    row[feature] = "AK-47"
                else:
                    row[feature] = 0
            row["round_num"] = index + 1
            row["first_kill_advantage_ct"] = 1 if index % 2 else -1
            rows.append(row)
        data = pd.DataFrame(rows)
        encoded = prepare_profile_splits(data, raw_features)["train"][0].columns.tolist()
        predictor = FrozenPredictor()
        bundle = {
            "model": predictor,
            "raw_features": raw_features,
            "columns": encoded,
        }

        outputs, audit = replay_frozen_model(data, bundle)

        self.assertEqual(set(outputs), {"val", "test"})
        self.assertEqual(audit["lightgbm_fit_calls"], 0)
        self.assertEqual(predictor.fit_calls, 0)
        self.assertEqual(predictor.predict_calls, 2)
        self.assertIn("map_name", outputs["test"].columns)
        self.assertIn("source_subset", outputs["test"].columns)
        self.assertTrue(outputs["test"]["source_subset"].eq("lan").all())

    def test_prediction_replay_joins_complete_keys_and_detects_drift(self) -> None:
        saved = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "game_id": ["lan:g1", "online:g2"],
                "round_id": ["r1", "r2"],
                "ct_win": [0, 1],
                "lightgbm_tuned_probability": [0.2, 0.8],
            }
        )
        replayed = saved[["series_id", "game_id", "round_id"]].iloc[::-1].reset_index(drop=True)
        replayed["y_true"] = [1, 0]
        replayed["ct_win_probability"] = [0.8, 0.2]

        accepted = audit_prediction_replay(saved, replayed, tolerance=1e-12)
        replayed.loc[0, "ct_win_probability"] += 1e-5
        rejected = audit_prediction_replay(saved, replayed, tolerance=1e-12)

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])

    def test_paired_table_and_bootstrap_use_identical_series_rows(self) -> None:
        series = np.repeat([f"s{i}" for i in range(8)], 4)
        replayed = pd.DataFrame(
            {
                "series_id": series,
                "game_id": np.repeat([f"lan:g{i}" for i in range(8)], 4),
                "round_id": [f"r{i}" for i in range(32)],
                "y_true": np.tile([0, 0, 1, 1], 8),
                "ct_win_probability": np.tile([0.1, 0.2, 0.8, 0.9], 8),
            }
        )
        saved = replayed[["series_id", "game_id", "round_id"]].copy()
        saved["ct_win"] = replayed["y_true"]
        saved["xgboost_frozen_probability"] = np.tile([0.6, 0.4, 0.6, 0.4], 8)

        paired = build_paired_prediction_table(replayed, saved)
        intervals = paired_model_bootstrap(paired, n_bootstrap=40, seed=7)

        self.assertEqual(len(paired), len(replayed))
        self.assertEqual(len(intervals), 5)
        self.assertTrue(intervals["successful_bootstraps"].eq(40).all())
        self.assertTrue(intervals["bootstrap_unit"].eq("series_id_paired").all())

    def test_global_interval_assessment_uses_post_first_kill_thresholds(self) -> None:
        intervals = pd.DataFrame(
            {
                "metric": ["accuracy", "auc", "log_loss", "brier_score", "ece10"],
                "ci_lower_95": [0.72, 0.791, 0.50, 0.16, 0.01],
                "ci_upper_95": [0.76, 0.82, 0.539, 0.19, 0.03],
                "successful_bootstraps": [2000] * 5,
            }
        )

        accepted = assess_global_intervals(intervals, n_bootstrap=2000)
        intervals.loc[intervals["metric"].eq("auc"), "ci_lower_95"] = 0.779
        rejected = assess_global_intervals(intervals, n_bootstrap=2000)

        self.assertTrue(accepted["minimum_passed"])
        self.assertTrue(accepted["stage_passed"])
        self.assertFalse(rejected["minimum_passed"])

    def test_group_assessment_requires_all_eight_outputs(self) -> None:
        map_table = pd.DataFrame(
            {"map_name": ["de_nuke"], "rounds": [400], "auc": [0.78], "auc_ci_lower_95": [0.72]}
        )
        grouped = {
            "map": map_table,
            "source": pd.DataFrame({"source_subset": ["lan"], "rounds": [400]}),
            "round_stage": pd.DataFrame({"round_stage": ["early"], "rounds": [400]}),
            "equipment_band": pd.DataFrame({"equipment_band": ["even"], "rounds": [400]}),
            "first_kill_side": pd.DataFrame({"first_kill_side": ["CT"], "rounds": [400]}),
            "first_kill_time_band": pd.DataFrame({"first_kill_time_band": ["fast"], "rounds": [400]}),
            "first_kill_weapon_family": pd.DataFrame({"first_kill_weapon_family": ["rifle"], "rounds": [400]}),
            "first_kill_headshot": pd.DataFrame({"first_kill_headshot_label": ["headshot"], "rounds": [400]}),
        }

        accepted = assess_group_robustness(
            grouped, {"absolute_difference": 0.03, "ci_includes_zero": True}
        )
        rejected = assess_group_robustness(
            {name: table for name, table in grouped.items() if name != "source"},
            {"absolute_difference": 0.03, "ci_includes_zero": True},
        )

        self.assertTrue(accepted["outputs_complete"])
        self.assertTrue(accepted["large_map_minimum_passed"])
        self.assertFalse(rejected["outputs_complete"])

    def test_calibration_protocol_rejects_test_metric_columns(self) -> None:
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
            {"method": ["uncalibrated", "sigmoid", "isotonic"], "log_loss": [0.5, 0.51, 0.52], "brier_score": [0.16, 0.17, 0.18]}
        )
        oof = validation[["series_id", "game_id", "round_id", "y_true"]].copy()
        oof["fold"] = [0, 0, 1, 1]
        for method in ("uncalibrated", "sigmoid", "isotonic"):
            oof[f"probability_{method}"] = validation["ct_win_probability"]

        accepted = audit_calibration_protocol(validation, comparison, oof, "uncalibrated", n_splits=2)
        rejected = audit_calibration_protocol(
            validation, comparison.assign(test_auc=[0.8] * 3), oof, "uncalibrated", n_splits=2
        )

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])

    def test_acceptance_does_not_require_lightgbm_superiority(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        accepted = decide_acceptance(checks)
        checks["paired_comparison"] = False
        rejected = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["ready_for_m31"])
        self.assertNotIn("lightgbm_significantly_better", BLOCKING_CHECKS)
        self.assertEqual(rejected["blocking_failures"], ["paired_comparison"])

    def test_reproduction_script_freezes_m29_inputs_and_bootstrap_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "run.ps1"
            script.write_text(
                " ".join(
                    [
                        "src.csdemo.m30_post_first_kill_lightgbm_evaluation",
                        "post_first_kill_lightgbm_tuned.joblib",
                        "m29_summary.json",
                        "[int]$BootstrapSamples = 2000",
                    ]
                ),
                encoding="utf-8",
            )

            accepted = audit_reproduction_entrypoint(script)
            script.write_text("wrong", encoding="utf-8")
            rejected = audit_reproduction_entrypoint(script)

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])

    def test_source_gap_text_uses_the_frozen_signed_difference_field(self) -> None:
        text = format_source_auc_gap(
            {
                "signed_difference": -0.0123,
                "ci_lower_95": -0.03,
                "ci_upper_95": 0.01,
            }
        )

        self.assertIn("-0.012300", text)
        self.assertIn("[-0.030000, +0.010000]", text)


if __name__ == "__main__":
    unittest.main()

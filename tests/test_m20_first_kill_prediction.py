import json
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

from src.csdemo.m16_first_kill_baselines import canonical_feature_names
from src.csdemo.predict_first_kill import (
    FirstKillInputValidationError,
    FirstKillPredictor,
    load_snapshot,
    validate_calibrator_bundle,
    validate_first_kill_snapshot,
    validate_model_bundle,
)


def valid_snapshot() -> dict:
    return {
        "map_name": "de_nuke",
        "round_num": 4,
        "ct_score": 2,
        "t_score": 1,
        "ct_eq_value": 22000,
        "t_eq_value": 18500,
        "ct_cash": 3500,
        "t_cash": 6200,
        "ct_armor": 5,
        "t_armor": 4,
        "ct_helmets": 4,
        "t_helmets": 3,
        "ct_defuse_kits": 2,
        "ct_grenades": 13,
        "t_grenades": 11,
        "ct_ak47": 0,
        "t_ak47": 2,
        "ct_m4a4": 2,
        "t_m4a4": 0,
        "ct_m4a1_s": 1,
        "t_m4a1_s": 0,
        "ct_awp": 1,
        "t_awp": 1,
        "ct_rifles": 3,
        "t_rifles": 2,
        "ct_smgs": 0,
        "t_smgs": 1,
        "first_kill_advantage_ct": 1,
        "first_kill_time": 22.5,
        "first_kill_headshot": True,
        "first_kill_weapon": "AK-47",
    }


class M20FirstKillPredictionTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    known_maps = {"de_nuke", "de_inferno"}
    known_weapons = {"AK-47", "AWP", "M4A1", "MP9"}

    @classmethod
    def model_path(cls) -> Path:
        return cls.root / "models" / "esta_full_m17" / "first_kill_xgboost_tuned.joblib"

    @classmethod
    def calibrator_path(cls) -> Path:
        return cls.root / "models" / "esta_full_m18" / "first_kill_calibrator.joblib"

    def test_validation_builds_the_exact_40_feature_contract(self) -> None:
        normalized, details = validate_first_kill_snapshot(
            valid_snapshot(), self.known_maps, self.known_weapons
        )

        self.assertEqual(list(normalized), canonical_feature_names())
        self.assertEqual(normalized["score_diff_ct"], 1)
        self.assertEqual(normalized["eq_value_diff_ct"], 3500)
        self.assertEqual(normalized["first_kill_advantage_ct"], 1)
        self.assertEqual(normalized["first_kill_time"], 22.5)
        self.assertEqual(normalized["first_kill_headshot"], 1)
        self.assertEqual(details["raw_model_feature_count"], 40)
        self.assertEqual(details["derived_feature_count"], 9)
        self.assertEqual(details["first_kill_side"], "CT")

    def test_headshot_accepts_boolean_or_zero_one_integer(self) -> None:
        for supplied, expected in ((True, 1), (False, 0), (1, 1), (0, 0)):
            snapshot = valid_snapshot()
            snapshot["first_kill_headshot"] = supplied
            normalized, _ = validate_first_kill_snapshot(
                snapshot, self.known_maps, self.known_weapons
            )
            self.assertEqual(normalized["first_kill_headshot"], expected)

    def test_validation_reports_all_missing_first_kill_fields(self) -> None:
        snapshot = valid_snapshot()
        for field in (
            "first_kill_advantage_ct",
            "first_kill_time",
            "first_kill_headshot",
            "first_kill_weapon",
        ):
            snapshot.pop(field)

        with self.assertRaises(FirstKillInputValidationError) as raised:
            validate_first_kill_snapshot(snapshot, self.known_maps, self.known_weapons)

        message = str(raised.exception)
        self.assertIn("missing required first-kill fields", message)
        self.assertIn("first_kill_advantage_ct", message)
        self.assertIn("first_kill_weapon", message)

    def test_validation_rejects_invalid_event_values_together(self) -> None:
        snapshot = valid_snapshot()
        snapshot["first_kill_advantage_ct"] = 0
        snapshot["first_kill_time"] = 181
        snapshot["first_kill_headshot"] = 2
        snapshot["first_kill_weapon"] = "Unknown Blaster"

        with self.assertRaises(FirstKillInputValidationError) as raised:
            validate_first_kill_snapshot(snapshot, self.known_maps, self.known_weapons)

        message = str(raised.exception)
        self.assertIn("first_kill_advantage_ct must be -1 or 1", message)
        self.assertIn("first_kill_time must be between 0 and 180", message)
        self.assertIn("first_kill_headshot must be a boolean or 0/1", message)
        self.assertIn("first_kill_weapon", message)
        self.assertIn("was not seen during training", message)

    def test_validation_rejects_string_event_types(self) -> None:
        snapshot = valid_snapshot()
        snapshot["first_kill_advantage_ct"] = "CT"
        snapshot["first_kill_time"] = "22.5"
        snapshot["first_kill_headshot"] = "yes"
        snapshot["first_kill_weapon"] = 7

        with self.assertRaises(FirstKillInputValidationError) as raised:
            validate_first_kill_snapshot(snapshot, self.known_maps, self.known_weapons)

        message = str(raised.exception)
        self.assertIn("first_kill_advantage_ct must be an integer", message)
        self.assertIn("first_kill_time must be a finite number", message)
        self.assertIn("first_kill_headshot must be an integer", message)
        self.assertIn("first_kill_weapon must be a non-empty string", message)

    def test_validation_rejects_purchase_errors_and_future_fields(self) -> None:
        snapshot = valid_snapshot()
        snapshot["map_name"] = "de_cache"
        snapshot["score_diff_ct"] = 99
        snapshot["ct_win"] = 1
        snapshot["second_kill_weapon"] = "AWP"
        snapshot["ct_alive_after_fk"] = 5

        with self.assertRaises(FirstKillInputValidationError) as raised:
            validate_first_kill_snapshot(snapshot, self.known_maps, self.known_weapons)

        message = str(raised.exception)
        self.assertIn("map_name", message)
        self.assertIn("score_diff_ct", message)
        self.assertIn("forbidden fields", message)
        self.assertIn("ct_win", message)
        self.assertIn("second_kill_weapon", message)
        self.assertIn("ct_alive_after_fk", message)

    def test_model_bundle_contract_rejects_task_and_feature_drift(self) -> None:
        bundle = joblib.load(self.model_path())
        audit = validate_model_bundle(bundle)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["raw_feature_count"], 40)
        self.assertEqual(audit["encoded_feature_count"], 82)

        wrong_task = dict(bundle)
        wrong_task["task"] = "pre_round"
        with self.assertRaisesRegex(ValueError, "task"):
            validate_model_bundle(wrong_task)

        wrong_features = dict(bundle)
        wrong_features["raw_features"] = list(bundle["raw_features"][:-1])
        with self.assertRaisesRegex(ValueError, "raw feature contract"):
            validate_model_bundle(wrong_features)

        wrong_columns = dict(bundle)
        wrong_columns["columns"] = list(bundle["columns"][:-1])
        with self.assertRaisesRegex(ValueError, "82 encoded features"):
            validate_model_bundle(wrong_columns)

    def test_calibrator_contract_rejects_model_or_data_drift(self) -> None:
        model_bundle = joblib.load(self.model_path())
        calibrator_bundle = joblib.load(self.calibrator_path())
        model_sha = calibrator_bundle["base_model_sha256"]
        audit = validate_calibrator_bundle(
            calibrator_bundle,
            model_sha256=model_sha,
            model_data_sha256=model_bundle["data_sha256"],
        )
        self.assertTrue(audit["passed"])

        with self.assertRaisesRegex(ValueError, "base model SHA-256"):
            validate_calibrator_bundle(
                calibrator_bundle,
                model_sha256="0" * 64,
                model_data_sha256=model_bundle["data_sha256"],
            )

        with self.assertRaisesRegex(ValueError, "data SHA-256"):
            validate_calibrator_bundle(
                calibrator_bundle,
                model_sha256=model_sha,
                model_data_sha256="1" * 64,
            )

    def test_load_snapshot_supports_one_json_object_and_one_csv_row(self) -> None:
        snapshot = valid_snapshot()
        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "snapshot.json"
            csv_path = Path(directory) / "snapshot.csv"
            json_path.write_text(json.dumps(snapshot), encoding="utf-8")
            pd.DataFrame([snapshot]).to_csv(csv_path, index=False)

            self.assertEqual(load_snapshot(json_path), snapshot)
            csv_snapshot = load_snapshot(csv_path)
            self.assertEqual(csv_snapshot["first_kill_weapon"], "AK-47")
            self.assertEqual(float(csv_snapshot["first_kill_time"]), 22.5)

    def test_real_predictor_outputs_valid_complementary_probabilities(self) -> None:
        predictor = FirstKillPredictor.from_paths(
            self.model_path(), self.calibrator_path()
        )
        result = predictor.predict(valid_snapshot())

        prediction = result["prediction"]
        self.assertGreaterEqual(prediction["ct_win_probability"], 0.0)
        self.assertLessEqual(prediction["ct_win_probability"], 1.0)
        self.assertAlmostEqual(
            prediction["ct_win_probability"] + prediction["t_win_probability"],
            1.0,
            places=12,
        )
        self.assertEqual(result["task"], "first_kill")
        self.assertEqual(result["calibration_method"], "uncalibrated")
        self.assertEqual(result["validation"]["encoded_model_feature_count"], 82)

    def test_json_and_csv_predictions_match_exactly(self) -> None:
        predictor = FirstKillPredictor.from_paths(
            self.model_path(), self.calibrator_path()
        )
        snapshot = valid_snapshot()
        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "snapshot.json"
            csv_path = Path(directory) / "snapshot.csv"
            json_path.write_text(json.dumps(snapshot), encoding="utf-8")
            pd.DataFrame([snapshot]).to_csv(csv_path, index=False)

            json_probability = predictor.predict(load_snapshot(json_path))["prediction"][
                "ct_win_probability"
            ]
            csv_probability = predictor.predict(load_snapshot(csv_path))["prediction"][
                "ct_win_probability"
            ]
            self.assertEqual(json_probability, csv_probability)

    def test_cli_succeeds_and_returns_code_two_for_invalid_input(self) -> None:
        with TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.json"
            invalid_path = Path(directory) / "invalid.json"
            output_path = Path(directory) / "prediction.json"
            valid_path.write_text(json.dumps(valid_snapshot()), encoding="utf-8")
            invalid = valid_snapshot()
            invalid["first_kill_advantage_ct"] = 0
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            base_command = [
                sys.executable,
                "-m",
                "src.csdemo.predict_first_kill",
                "--model",
                str(self.model_path()),
                "--calibrator",
                str(self.calibrator_path()),
            ]

            success = subprocess.run(
                [*base_command, "--input", str(valid_path), "--output", str(output_path)],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(json.loads(success.stdout)["task"], "first_kill")
            self.assertTrue(output_path.is_file())

            failure = subprocess.run(
                [*base_command, "--input", str(invalid_path)],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(failure.returncode, 2)
            error = json.loads(failure.stderr)
            self.assertEqual(error["status"], "error")
            self.assertIn("first_kill_advantage_ct", error["message"])

    def test_acceptance_runner_writes_all_m20_artifacts(self) -> None:
        from src.csdemo.m20_first_kill_interface import run_acceptance

        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            output_dir = directory_path / "reports"
            json_path = directory_path / "snapshot.json"
            csv_path = directory_path / "snapshot.csv"
            json_path.write_text(json.dumps(valid_snapshot()), encoding="utf-8")
            pd.DataFrame([valid_snapshot()]).to_csv(csv_path, index=False)

            summary = run_acceptance(
                model_path=self.model_path(),
                calibrator_path=self.calibrator_path(),
                json_example=json_path,
                csv_example=csv_path,
                m18_summary_path=self.root / "reports" / "esta_full_m18" / "m18_summary.json",
                m19_summary_path=self.root / "reports" / "esta_full_m19" / "m19_summary.json",
                m17_comparison_path=self.root / "reports" / "esta_full_m17" / "model_comparison.csv",
                benchmarks_path=self.root / "benchmarks" / "external_first_kill_tuned_metrics.csv",
                report_dir=output_dir,
                project_root=self.root,
                run_tests=False,
            )

            self.assertEqual(summary["acceptance"]["status"], "passed")
            self.assertTrue(summary["checks"]["json_csv_prediction_match"])
            self.assertEqual(summary["checks"]["validation_cases_passed"], 10)
            self.assertEqual(summary["formal_targets"]["passed_count"], 10)
            self.assertFalse(summary["model_performance_changed"])
            for filename in (
                "m20_summary.json",
                "m20_checks.csv",
                "m20_first_kill_interface_report.md",
                "example_prediction.json",
                "validation_error_examples.json",
                "model_contract_audit.json",
                "external_benchmark_comparison.csv",
                "external_benchmark_comparison.md",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()

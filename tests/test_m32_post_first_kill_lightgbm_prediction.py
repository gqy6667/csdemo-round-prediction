import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

from src.csdemo.m15_first_kill_data import fingerprint_file
from src.csdemo.m16_first_kill_baselines import canonical_feature_names
from src.csdemo.m32_post_first_kill_lightgbm_interface import (
    BLOCKING_CHECKS,
    audit_reproduction_entrypoint,
    decide_acceptance,
    run_acceptance,
)
from src.csdemo.predict_first_kill import (
    FirstKillInputValidationError,
    load_snapshot,
)
from src.csdemo.predict_first_kill_lightgbm import (
    PostFirstKillLightGBMPredictor,
    validate_post_first_kill_calibrator_bundle,
    validate_post_first_kill_lightgbm_model_bundle,
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


class M32PostFirstKillLightGBMPredictionTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    @classmethod
    def model_path(cls) -> Path:
        return (
            cls.root
            / "models"
            / "esta_full_m29"
            / "post_first_kill_lightgbm_tuned.joblib"
        )

    @classmethod
    def calibrator_path(cls) -> Path:
        return (
            cls.root
            / "models"
            / "esta_full_m30"
            / "post_first_kill_lightgbm_calibrator.joblib"
        )

    def test_real_model_bundle_has_exact_lightgbm_deployment_contract(self) -> None:
        bundle = joblib.load(self.model_path())

        audit = validate_post_first_kill_lightgbm_model_bundle(bundle)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["task"], "post_first_kill")
        self.assertEqual(audit["model_name"], "lightgbm_tuned")
        self.assertEqual(audit["raw_feature_count"], 40)
        self.assertEqual(audit["encoded_feature_count"], 82)
        self.assertEqual(audit["known_map_count"], 8)
        self.assertEqual(audit["known_weapon_count"], 36)
        self.assertEqual(audit["deployment_tree_count"], 211)
        self.assertEqual(audit["booster_space_sanitized_count"], 9)

    def test_model_contract_rejects_task_feature_column_and_tree_drift(self) -> None:
        bundle = joblib.load(self.model_path())

        wrong_task = dict(bundle)
        wrong_task["task"] = "pre_round"
        with self.assertRaisesRegex(ValueError, "task"):
            validate_post_first_kill_lightgbm_model_bundle(wrong_task)

        wrong_raw = dict(bundle)
        wrong_raw["raw_features"] = list(bundle["raw_features"][:-1])
        with self.assertRaisesRegex(ValueError, "raw feature contract"):
            validate_post_first_kill_lightgbm_model_bundle(wrong_raw)

        wrong_columns = dict(bundle)
        wrong_columns["columns"] = list(bundle["columns"][:-1])
        with self.assertRaisesRegex(ValueError, "82 encoded"):
            validate_post_first_kill_lightgbm_model_bundle(wrong_columns)

        wrong_trees = dict(bundle)
        wrong_trees["best_iteration"] = 210
        with self.assertRaisesRegex(ValueError, "211"):
            validate_post_first_kill_lightgbm_model_bundle(wrong_trees)

    def test_calibrator_contract_binds_model_data_and_validation_selection(self) -> None:
        model = joblib.load(self.model_path())
        calibrator = joblib.load(self.calibrator_path())
        model_sha = fingerprint_file(self.model_path())["sha256"]

        audit = validate_post_first_kill_calibrator_bundle(
            calibrator,
            model_sha256=model_sha,
            model_data_sha256=model["data_sha256"],
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["method"], "uncalibrated")
        self.assertEqual(audit["selection_data"], "validation only")
        self.assertEqual(audit["validation_folds"], 5)

        wrong_selection = dict(calibrator)
        wrong_selection["selection_data"] = "test"
        with self.assertRaisesRegex(ValueError, "validation only"):
            validate_post_first_kill_calibrator_bundle(
                wrong_selection,
                model_sha256=model_sha,
                model_data_sha256=model["data_sha256"],
            )

    def test_real_predictor_outputs_strict_feature_and_probability_contract(self) -> None:
        predictor = PostFirstKillLightGBMPredictor.from_paths(
            self.model_path(), self.calibrator_path()
        )

        result = predictor.predict(valid_snapshot())

        prediction = result["prediction"]
        self.assertEqual(result["task"], "post_first_kill")
        self.assertEqual(result["model_name"], "lightgbm_tuned")
        self.assertEqual(result["calibration_method"], "uncalibrated")
        self.assertEqual(result["validation"]["raw_model_feature_count"], 40)
        self.assertEqual(result["validation"]["encoded_model_feature_count"], 82)
        self.assertEqual(result["validation"]["deployment_tree_count"], 211)
        self.assertEqual(
            prediction["base_ct_win_probability"],
            prediction["ct_win_probability"],
        )
        self.assertAlmostEqual(prediction["probability_sum"], 1.0, places=12)

    def test_predictor_builds_exact_raw_contract_and_rejects_forbidden_fields(self) -> None:
        predictor = PostFirstKillLightGBMPredictor.from_paths(
            self.model_path(), self.calibrator_path()
        )
        normalized, details = predictor.validate(valid_snapshot())

        self.assertEqual(list(normalized), canonical_feature_names())
        self.assertEqual(normalized["score_diff_ct"], 1)
        self.assertEqual(normalized["eq_value_diff_ct"], 3500)
        self.assertEqual(normalized["first_kill_advantage_ct"], 1)
        self.assertEqual(details["raw_model_feature_count"], 40)

        invalid = valid_snapshot()
        invalid["map_name"] = "de_cache"
        invalid["first_kill_weapon"] = "Unknown Blaster"
        invalid["series_id"] = "leak"
        invalid["ct_win"] = 1
        invalid["second_kill_weapon"] = "AWP"
        with self.assertRaises(FirstKillInputValidationError) as raised:
            predictor.predict(invalid)

        message = str(raised.exception)
        for field in (
            "map_name",
            "first_kill_weapon",
            "series_id",
            "ct_win",
            "second_kill_weapon",
        ):
            self.assertIn(field, message)

    def test_json_and_csv_predictions_match_exactly(self) -> None:
        predictor = PostFirstKillLightGBMPredictor.from_paths(
            self.model_path(), self.calibrator_path()
        )
        snapshot = valid_snapshot()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "snapshot.json"
            csv_path = root / "snapshot.csv"
            json_path.write_text(json.dumps(snapshot), encoding="utf-8")
            pd.DataFrame([snapshot]).to_csv(csv_path, index=False)

            json_result = predictor.predict(load_snapshot(json_path))
            csv_result = predictor.predict(load_snapshot(csv_path))

        self.assertEqual(json_result, csv_result)

    def test_cli_succeeds_and_returns_code_two_for_invalid_input(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.json"
            invalid_path = root / "invalid.json"
            output_path = root / "prediction.json"
            valid_path.write_text(json.dumps(valid_snapshot()), encoding="utf-8")
            invalid = valid_snapshot()
            invalid["first_kill_advantage_ct"] = 0
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            base = [
                sys.executable,
                "-m",
                "src.csdemo.predict_first_kill_lightgbm",
                "--model",
                str(self.model_path()),
                "--calibrator",
                str(self.calibrator_path()),
            ]

            success = subprocess.run(
                [*base, "--input", str(valid_path), "--output", str(output_path)],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            failure = subprocess.run(
                [*base, "--input", str(invalid_path)],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            output_exists = output_path.is_file()

        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(json.loads(success.stdout)["task"], "post_first_kill")
        self.assertTrue(output_exists)
        self.assertEqual(failure.returncode, 2)
        self.assertIn("first_kill_advantage_ct", json.loads(failure.stderr)["message"])

    def test_acceptance_requires_every_declared_blocker(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        passed = decide_acceptance(checks)
        checks["artifact_integrity"] = False
        failed = decide_acceptance(checks)

        self.assertEqual(passed["status"], "passed")
        self.assertTrue(passed["ready_for_m33"])
        self.assertEqual(failed["blocking_failures"], ["artifact_integrity"])

    def test_reproduction_entrypoint_freezes_expected_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            script = Path(directory) / "run.ps1"
            script.write_text(
                " ".join(
                    [
                        "src.csdemo.m32_post_first_kill_lightgbm_interface",
                        "post_first_kill_lightgbm_tuned.joblib",
                        "post_first_kill_lightgbm_calibrator.joblib",
                        "m31_summary.json",
                    ]
                ),
                encoding="utf-8",
            )

            accepted = audit_reproduction_entrypoint(script)
            script.write_text("wrong", encoding="utf-8")
            rejected = audit_reproduction_entrypoint(script)

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])

    def test_acceptance_runner_writes_all_m32_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report_dir = root / "reports"
            json_path = root / "snapshot.json"
            csv_path = root / "snapshot.csv"
            output_path = root / "example_output.json"
            json_path.write_text(json.dumps(valid_snapshot()), encoding="utf-8")
            pd.DataFrame([valid_snapshot()]).to_csv(csv_path, index=False)

            summary = run_acceptance(
                model_path=self.model_path(),
                calibrator_path=self.calibrator_path(),
                json_example=json_path,
                csv_example=csv_path,
                example_output_path=output_path,
                m30_summary_path=self.root
                / "reports"
                / "esta_full_m30"
                / "m30_summary.json",
                m31_summary_path=self.root
                / "reports"
                / "esta_full_m31"
                / "m31_summary.json",
                m31_external_path=self.root
                / "reports"
                / "esta_full_m31"
                / "external_benchmark_comparison.csv",
                m31_external_markdown_path=self.root
                / "reports"
                / "esta_full_m31"
                / "external_benchmark_comparison.md",
                report_dir=report_dir,
                project_root=self.root,
                run_verification=False,
            )

            self.assertEqual(summary["acceptance"]["status"], "passed")
            self.assertTrue(summary["checks"]["json_csv_prediction_match"])
            self.assertEqual(summary["validation_cases"]["passed"], 10)
            self.assertFalse(summary["model_performance_changed"])
            self.assertTrue(output_path.is_file())
            for filename in (
                "m32_summary.json",
                "m32_checks.csv",
                "m32_experiment_manifest.json",
                "m32_post_first_kill_lightgbm_interface_report.md",
                "example_prediction.json",
                "validation_error_examples.json",
                "model_contract_audit.json",
                "external_benchmark_comparison.csv",
                "external_benchmark_comparison.md",
            ):
                self.assertTrue((report_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()

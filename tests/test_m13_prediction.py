import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.csdemo.predict_pre_round import (
    InputValidationError,
    PreRoundPredictor,
    load_snapshot,
    validate_snapshot,
)
from src.csdemo.m13_interface import run_acceptance


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
    }


class M13PredictionTests(unittest.TestCase):
    known_maps = {"de_nuke", "de_inferno"}

    def test_validation_computes_all_difference_features(self) -> None:
        normalized, details = validate_snapshot(valid_snapshot(), self.known_maps)

        self.assertEqual(normalized["score_diff_ct"], 1)
        self.assertEqual(normalized["eq_value_diff_ct"], 3500)
        self.assertEqual(normalized["grenade_diff_ct"], 2)
        self.assertEqual(normalized["rifle_diff_ct"], 1)
        self.assertEqual(details["status"], "passed")
        self.assertEqual(len(details["derived_features"]), 9)

    def test_validation_reports_all_missing_required_fields(self) -> None:
        with self.assertRaises(InputValidationError) as raised:
            validate_snapshot({"map_name": "de_nuke"}, self.known_maps)

        self.assertIn("missing required fields", str(raised.exception))
        self.assertIn("round_num", str(raised.exception))
        self.assertGreater(len(raised.exception.errors), 1)

    def test_validation_rejects_unknown_map_and_inconsistent_difference(self) -> None:
        snapshot = valid_snapshot()
        snapshot["map_name"] = "de_cache"
        snapshot["score_diff_ct"] = 99

        with self.assertRaises(InputValidationError) as raised:
            validate_snapshot(snapshot, self.known_maps)

        self.assertIn("map_name", str(raised.exception))
        self.assertIn("score_diff_ct", str(raised.exception))

    def test_validation_rejects_round_and_inventory_inconsistency(self) -> None:
        snapshot = valid_snapshot()
        snapshot["round_num"] = 8
        snapshot["ct_helmets"] = 6
        snapshot["ct_rifles"] = 1

        with self.assertRaises(InputValidationError) as raised:
            validate_snapshot(snapshot, self.known_maps)

        message = str(raised.exception)
        self.assertIn("round_num must equal", message)
        self.assertIn("ct_helmets", message)
        self.assertIn("ct_rifles", message)

    def test_real_predictor_outputs_valid_complementary_probabilities(self) -> None:
        root = Path(__file__).resolve().parents[1]
        predictor = PreRoundPredictor.from_paths(
            root / "models" / "esta_full_m8_tuned" / "pre_round_xgb.joblib",
            root / "models" / "esta_full_m10" / "pre_round_calibrator.joblib",
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
        self.assertEqual(result["validation"]["status"], "passed")
        self.assertEqual(result["calibration_method"], "uncalibrated")

    def test_load_snapshot_supports_one_json_object_and_one_csv_row(self) -> None:
        snapshot = valid_snapshot()
        with TemporaryDirectory() as directory:
            json_path = Path(directory) / "snapshot.json"
            csv_path = Path(directory) / "snapshot.csv"
            json_path.write_text(json.dumps(snapshot), encoding="utf-8")
            pd.DataFrame([snapshot]).to_csv(csv_path, index=False)

            self.assertEqual(load_snapshot(json_path), snapshot)
            self.assertEqual(load_snapshot(csv_path), snapshot)

    def test_load_snapshot_rejects_multiple_csv_rows(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.csv"
            pd.DataFrame([valid_snapshot(), valid_snapshot()]).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "exactly one row"):
                load_snapshot(path)

    def test_acceptance_runner_writes_interface_and_benchmark_reports(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "reports"
            json_path = Path(directory) / "snapshot.json"
            csv_path = Path(directory) / "snapshot.csv"
            json_path.write_text(json.dumps(valid_snapshot()), encoding="utf-8")
            pd.DataFrame([valid_snapshot()]).to_csv(csv_path, index=False)

            summary = run_acceptance(
                model_path=root
                / "models"
                / "esta_full_m8_tuned"
                / "pre_round_xgb.joblib",
                calibrator_path=root
                / "models"
                / "esta_full_m10"
                / "pre_round_calibrator.joblib",
                json_example=json_path,
                csv_example=csv_path,
                metrics_path=root / "reports" / "esta_full_m9" / "m9_summary.json",
                benchmarks_path=root / "benchmarks" / "external_round_model_metrics.csv",
                report_dir=output_dir,
            )

            self.assertEqual(summary["status"], "passed")
            self.assertTrue(summary["checks"]["json_csv_prediction_match"])
            self.assertEqual(summary["checks"]["validation_cases_passed"], 5)
            for filename in (
                "m13_summary.json",
                "m13_interface_report.md",
                "example_prediction.json",
                "validation_error_examples.json",
                "external_benchmark_comparison.csv",
                "external_benchmark_comparison.md",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.csdemo.predict_pre_round import BASE_FEATURES
from src.csdemo.predict_pre_round import PreRoundPredictor
from src.csdemo.predict_pre_round_lightgbm import PreRoundLightGBMPredictor
from src.csdemo.predict_first_kill import FirstKillPredictor
from src.csdemo.predict_first_kill_lightgbm import PostFirstKillLightGBMPredictor
from src.csdemo.roundcast_service import DATA_FILES, MODEL_FILES, RoundcastService, RoundcastValidationError


ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_CASES = (
    ("pre_round", "xgboost", PreRoundPredictor, "M8-tuned / M10-identity", "M13:27-base+9-derived"),
    ("pre_round", "lightgbm", PreRoundLightGBMPredictor, "M23-tuned / M24-identity", "M26:27-base+9-derived"),
    ("post_first_kill", "xgboost", FirstKillPredictor, "M17-tuned / M18-identity", "M20:31-base+9-derived"),
    ("post_first_kill", "lightgbm", PostFirstKillLightGBMPredictor, "M29-tuned / M30-identity", "M32:31-base+9-derived"),
)


class RoundcastTrustTests(unittest.TestCase):
    def test_preflight_verifies_ten_frozen_artifacts_without_loading_joblib(self):
        with patch("joblib.load", side_effect=AssertionError("T01 must not load models")):
            service = RoundcastService(ROOT)
        proof = service.readiness_report()
        self.assertEqual(proof["status"], "passed")
        self.assertEqual(len(proof["frozen_artifacts"]), 10)
        self.assertFalse(proof["inference_executed"])
        for item in proof["frozen_artifacts"]:
            self.assertEqual(item["expected_sha256"], item["actual_sha256"])
            self.assertIn("manifest", item["provenance"])

    def test_four_model_routes_cannot_cross_prediction_stages(self):
        service = RoundcastService(ROOT)
        for stage in ("pre_round", "post_first_kill"):
            for algorithm, prefix in (("xgboost", "xgb"), ("lightgbm", "lgbm")):
                model = service.model_metadata(stage, algorithm)
                self.assertEqual(model["model_id"], f"{prefix}_{stage}")
                self.assertEqual(model["stage"], stage)
                self.assertEqual(model["algorithm"], algorithm)
                self.assertTrue(model["inference_ready"])
                self.assertEqual(model["available_examples"], ["A", "B", "C"])
        for stage, algorithm in (("first_kill", "xgboost"),
                                 ("pre_round", "lgbm"),
                                 ("bomb_plant", "lightgbm")):
            with self.subTest(stage=stage, algorithm=algorithm):
                with self.assertRaises(RoundcastValidationError):
                    service.model_metadata(stage, algorithm)

    def test_bad_initial_hash_is_rejected_before_any_model_load(self):
        registry = json.loads((ROOT / "examples/roundcast_v1_cases.json").read_text(encoding="utf-8"))
        bad = deepcopy(registry)
        bad["files"]["models/esta_full_m8_tuned/pre_round_xgb.joblib"]["sha256"] = "0" * 64
        with patch("src.csdemo.roundcast_service._read_registry", return_value=bad):
            with patch("joblib.load", side_effect=AssertionError("unsafe load")) as load:
                with self.assertRaisesRegex(RoundcastValidationError, "hash"):
                    RoundcastService(ROOT)
                load.assert_not_called()

    def test_missing_artifact_fails_closed(self):
        original = Path.read_bytes

        def missing(path):
            if path.name == "pre_round_xgb.joblib":
                raise FileNotFoundError(path)
            return original(path)

        with patch.object(Path, "read_bytes", missing):
            with patch("joblib.load", side_effect=AssertionError("unsafe load")):
                with self.assertRaisesRegex(RoundcastValidationError, "missing"):
                    RoundcastService(ROOT)

    def test_changed_file_bytes_and_manifest_pins_fail_before_parquet_parsing(self):
        original = Path.read_bytes
        for name in ("pre_round_xgb.joblib", "first_kill.parquet", "m14_experiment_manifest.json"):
            def corrupted(path):
                content = original(path)
                return content + b"tampered" if path.name == name else content

            with self.subTest(file=name):
                with patch.object(Path, "read_bytes", corrupted):
                    with patch("src.csdemo.roundcast_service.pd.read_parquet") as read:
                        with self.assertRaisesRegex(RoundcastValidationError, "hash"):
                            RoundcastService(ROOT)
                        read.assert_not_called()

    def test_missing_registry_entry_and_wrong_manifest_pointer_fail_closed(self):
        registry = json.loads((ROOT / "examples/roundcast_v1_cases.json").read_text(encoding="utf-8"))
        for fault in ("missing", "pointer", "path"):
            bad = deepcopy(registry)
            path = "models/esta_full_m8_tuned/pre_round_xgb.joblib"
            if fault == "missing":
                del bad["files"][path]
            elif fault == "pointer":
                bad["files"][path]["provenance"]["entry"] = ["artifacts", "calibrator"]
            else:
                bad["files"]["../outside.joblib"] = bad["files"].pop(path)
            with self.subTest(fault=fault):
                with patch("src.csdemo.roundcast_service._read_registry", return_value=bad):
                    with self.assertRaises(RoundcastValidationError):
                        RoundcastService(ROOT)

    def test_duplicate_json_registry_keys_are_rejected(self):
        with patch.object(Path, "read_text", return_value='{"schema_version": 1, "schema_version": 1}'):
            with self.assertRaisesRegex(RoundcastValidationError, "Duplicate JSON"):
                RoundcastService(ROOT)

    def test_malformed_registry_mapping_shapes_return_validation_errors(self):
        registry = json.loads((ROOT / "examples/roundcast_v1_cases.json").read_text(encoding="utf-8"))
        for bad in ([], {**registry, "files": []}, {**registry, "manifest_pins": []}):
            with self.subTest(registry=type(bad).__name__):
                with patch("src.csdemo.roundcast_service._read_registry", return_value=bad):
                    with self.assertRaises(RoundcastValidationError):
                        RoundcastService(ROOT)

    def test_reference_source_requires_its_own_trusted_file_pin(self):
        registry = json.loads((ROOT / "examples/roundcast_v1_cases.json").read_text(encoding="utf-8"))
        del registry["files"]["reports/esta_full_m10/calibrated_test_predictions.csv"]
        with patch("src.csdemo.roundcast_service._read_registry", return_value=registry):
            with self.assertRaises(RoundcastValidationError):
                RoundcastService(ROOT)


class RoundcastInferenceTests(unittest.TestCase):
    def test_nested_predictor_metadata_and_boolean_probability_sum_fail_closed(self):
        service = RoundcastService(ROOT)
        for stage, algorithm, predictor_class, _, _ in PREDICTOR_CASES:
            original = predictor_class.predict
            for fault in ('snapshot', 'validation', 'probability_sum', 'contract_flag'):
                if fault == 'contract_flag' and (stage, algorithm) == ('pre_round', 'xgboost'):
                    continue
                def malformed(predictor, snapshot):
                    result = original(predictor, snapshot)
                    if fault == 'snapshot':
                        result['snapshot_definition'] = {'local_path': 'C:/SYNTHETIC_PRIVATE/diagnostic.txt'}
                    elif fault == 'validation':
                        result['validation']['local_path'] = 'C:/SYNTHETIC_PRIVATE/diagnostic.txt'
                    elif fault == 'contract_flag':
                        result['validation']['model_contract_verified'] = 1
                    else:
                        result['prediction']['probability_sum'] = True
                    return result
                with self.subTest(stage=stage, algorithm=algorithm, fault=fault):
                    with patch.object(predictor_class, 'predict', autospec=True, side_effect=malformed):
                        with self.assertRaises(RoundcastValidationError):
                            service.predict_example('A', stage, algorithm)

    def test_same_stage_reference_source_swaps_are_rejected(self):
        for stage in ('pre_round', 'post_first_kill'):
            registry = json.loads((ROOT / 'examples/roundcast_v1_cases.json').read_text(encoding='utf-8'))
            sources = registry['reference_sources']
            sources['xgb_' + stage], sources['lgbm_' + stage] = sources['lgbm_' + stage], sources['xgb_' + stage]
            with self.subTest(stage=stage), patch('src.csdemo.roundcast_service._read_registry', return_value=registry):
                with self.assertRaises(RoundcastValidationError):
                    RoundcastService(ROOT)

    def test_real_case_a_prediction_matches_reference_without_reading_it(self):
        service = RoundcastService(ROOT)
        result = service.predict_example("A", "pre_round", "xgboost")
        self.assertAlmostEqual(result["prediction"]["ct_win_probability"], 0.6291529536247253, delta=1e-8)
        self.assertEqual(result["prediction"]["t_win_probability"], 1 - result["prediction"]["ct_win_probability"])
        self.assertEqual(result["calibration_method"], "uncalibrated")
        self.assertEqual(result["prediction"]["decision_threshold"], 0.5)
        self.assertEqual(result["model_id"], "xgb_pre_round")
        self.assertEqual(result["validation"]["required_base_feature_count"], 27)
        self.assertEqual(result["feature_version"], "M13:27-base+9-derived")
        self.assertEqual(result["identity"], service.snapshot("A", "pre_round")["identity"])
        self.assertGreaterEqual(result["inference_ms"], 0)
        self.assertEqual(result["source"]["data_sha256"], service._registry["files"]["data/processed/esta_full/pre_round.parquet"]["sha256"])
        service._registry["reference_probabilities"] = {"A": 0.001}
        with patch("pandas.read_csv", side_effect=AssertionError("reference must not feed prediction")):
            again = service.predict_example("A", "pre_round", "xgboost")
        self.assertEqual(again["prediction"], result["prediction"])
        self.assertNotEqual(again["request_id"], result["request_id"])
        self.assertNotIn("winning_side", result)

    def test_invalid_combinations_rejected(self):
        service = RoundcastService(ROOT)
        for args in (("D", "pre_round", "xgboost"), ("A", "pre_round", "lgbm"),
                     ("A", "first_kill", "xgboost"), ([], "pre_round", "xgboost"),
                     ("A", "xgboost", "pre_round"), ("A", [], "xgboost")):
            with self.subTest(args=args), self.assertRaises(RoundcastValidationError):
                service.predict_example(*args)

    def test_malformed_predictor_outputs_fail_closed(self):
        service = RoundcastService(ROOT)
        real = service.predict_example("A", "pre_round", "xgboost")
        valid = {key: real[key] for key in ("task", "snapshot_definition", "calibration_method", "validation", "prediction")}
        faults = [None, {**valid, "model_id": "lgbm_post_first_kill"},
                  {**valid, "prediction": {**valid["prediction"], "probability_sum": 999}},
                  {**valid, "prediction": {**valid["prediction"], "ct_win_probability": float("nan")}}]
        for result in faults:
            with self.subTest(result=result), patch("src.csdemo.predict_pre_round.PreRoundPredictor.predict", return_value=result):
                with self.assertRaises(RoundcastValidationError):
                    service.predict_example("A", "pre_round", "xgboost")

    def test_predictor_failure_never_returns_archived_probability(self):
        service = RoundcastService(ROOT)
        with patch("src.csdemo.predict_pre_round.PreRoundPredictor.predict", side_effect=RuntimeError("private path")):
            with self.assertRaisesRegex(RoundcastValidationError, "Inference unavailable"):
                service.predict_example("A", "pre_round", "xgboost")

    def test_changed_or_missing_artifact_after_startup_blocks_inference(self):
        service = RoundcastService(ROOT)
        original = Path.read_bytes
        for name in ("pre_round_xgb.joblib", "pre_round_calibrator.joblib", "pre_round.parquet"):
            def corrupt(path):
                return original(path) + b"changed" if path.name == name else original(path)
            with self.subTest(file=name), patch.object(Path, "read_bytes", corrupt):
                with patch("joblib.load", side_effect=AssertionError("must check first")) as load:
                    with self.assertRaises(RoundcastValidationError):
                        service.predict_example("A", "pre_round", "xgboost")
                    load.assert_not_called()


class RoundcastMatrixTests(unittest.TestCase):
    def test_wrong_native_stage_and_lightgbm_identity_or_base_are_rejected(self):
        service = RoundcastService(ROOT)
        for stage, algorithm, predictor_class, _, _ in PREDICTOR_CASES:
            actual_predict = predictor_class.predict
            faults = ["task", "encoded_features", "probability_sum"]
            if algorithm == "lightgbm":
                faults += ["model_sha256", "model_name", "base"]
            for fault in faults:
                def malformed(predictor, snapshot):
                    result = actual_predict(predictor, snapshot)
                    if fault == "task":
                        result["task"] = "pre_round" if stage == "post_first_kill" else "first_kill"
                    elif fault == "encoded_features":
                        result["validation"]["encoded_model_feature_count"] = 999
                    elif fault == "probability_sum":
                        result["prediction"]["probability_sum"] = float("nan")
                    elif fault == "model_sha256":
                        result["model_sha256"] = "0" * 64
                    elif fault == "model_name":
                        result["model_name"] = "xgboost"
                    else:
                        result["prediction"]["base_ct_win_probability"] = 0.01
                    return result
                with self.subTest(stage=stage, algorithm=algorithm, fault=fault):
                    with patch.object(predictor_class, "predict", autospec=True, side_effect=malformed):
                        with self.assertRaisesRegex(RoundcastValidationError, "Inference unavailable"):
                            service.predict_example("A", stage, algorithm)

    def test_nonidentity_or_misbound_calibrator_is_rejected_before_prediction(self):
        import joblib
        service = RoundcastService(ROOT)
        original_load = joblib.load
        for stage, algorithm, predictor_class, _, _ in PREDICTOR_CASES:
            faults = ["nonidentity"] if predictor_class is PreRoundPredictor else ["nonidentity", "model_hash"]
            for fault in faults:
                loads = []
                def altered_load(stream):
                    bundle = original_load(stream)
                    loads.append(1)
                    if len(loads) == 2:
                        if fault == "nonidentity":
                            bundle["calibrator"] = object()
                        else:
                            bundle["base_model_sha256"] = "0" * 64
                    return bundle
                with self.subTest(stage=stage, algorithm=algorithm, fault=fault):
                    with patch("joblib.load", side_effect=altered_load), patch.object(predictor_class, "predict") as predict:
                        with self.assertRaisesRegex(RoundcastValidationError, "Inference unavailable"):
                            service.predict_example("A", stage, algorithm)
                        predict.assert_not_called()

    def test_twelve_real_predictions_are_reference_independent_and_use_correct_snapshots(self):
        service = RoundcastService(ROOT)
        references = deepcopy(service._registry["reference_probabilities"])
        service._registry["reference_probabilities"] = {"invalid": "not inference data"}
        request_ids, combinations = set(), set()
        for stage, algorithm, predictor_class, model_version, feature_version in PREDICTOR_CASES:
            model_id, model_path, calibrator_path = MODEL_FILES[stage, algorithm]
            actual_predict = predictor_class.predict
            with patch.object(predictor_class, "predict", autospec=True, side_effect=actual_predict) as predict:
                with patch.object(predictor_class, "from_paths", side_effect=AssertionError("must use verified bytes")):
                    with patch("pandas.read_csv", side_effect=AssertionError("reference is not prediction")):
                        for example_id in ("A", "B", "C"):
                            with self.subTest(example=example_id, model=model_id):
                                result = service.predict_example(example_id, stage, algorithm)
                                probability = result["prediction"]
                                self.assertAlmostEqual(probability["ct_win_probability"], references[example_id][model_id], delta=1e-8)
                                self.assertEqual(probability["t_win_probability"], 1 - probability["ct_win_probability"])
                                self.assertEqual(result["base_ct_win_probability"], probability["ct_win_probability"])
                                self.assertEqual(result["stage"], stage)
                                self.assertEqual(result["task"], stage)
                                self.assertEqual(result["algorithm"], algorithm)
                                self.assertEqual(result["model_id"], model_id)
                                self.assertEqual(result["model_version"], model_version)
                                self.assertEqual(result["feature_version"], feature_version)
                                self.assertEqual(result["model_sha256"], service._registry["files"][model_path]["sha256"])
                                self.assertEqual(result["calibrator_sha256"], service._registry["files"][calibrator_path]["sha256"])
                                self.assertEqual(result["source"]["data_sha256"], service._registry["files"][DATA_FILES[stage]]["sha256"])
                                self.assertEqual(result["validation"]["required_input_field_count"], 27 if stage == "pre_round" else 31)
                                self.assertEqual(result["validation"]["encoded_model_feature_count"], 43 if stage == "pre_round" else 82)
                                snapshot = service.snapshot(example_id, stage)
                                self.assertEqual(predict.call_args.args[1], snapshot["features"])
                                self.assertEqual(result["identity"], snapshot["identity"])
                                self.assertNotIn("ct_win", predict.call_args.args[1])
                                self.assertNotIn("winning_side", result)
                                request_ids.add(result["request_id"])
                                combinations.add((example_id, stage, algorithm))
            self.assertEqual(predict.call_count, 3)
        self.assertEqual(len(combinations), 12)
        self.assertEqual(len(request_ids), 12)

    def test_all_routes_fail_if_active_source_changes_or_disappears_after_start(self):
        service = RoundcastService(ROOT)
        original = Path.read_bytes
        for stage, algorithm, _, _, _ in PREDICTOR_CASES:
            for relative in (*MODEL_FILES[stage, algorithm][1:], DATA_FILES[stage]):
                for missing in (False, True):
                    def changed(path):
                        if path == ROOT / relative:
                            if missing:
                                raise FileNotFoundError(path)
                            return original(path) + b"corrupt"
                        return original(path)
                    with self.subTest(stage=stage, algorithm=algorithm, path=relative, missing=missing):
                        with patch.object(Path, "read_bytes", changed), patch("joblib.load") as load:
                            with self.assertRaises(RoundcastValidationError):
                                service.predict_example("B", stage, algorithm)
                            load.assert_not_called()

    def test_every_predictor_exception_fails_without_fallback(self):
        service = RoundcastService(ROOT)
        for stage, algorithm, predictor_class, _, _ in PREDICTOR_CASES:
            with self.subTest(stage=stage, algorithm=algorithm):
                with patch.object(predictor_class, "predict", side_effect=RuntimeError("injected inference failure")) as predict:
                    with self.assertRaisesRegex(RoundcastValidationError, "Inference unavailable"):
                        service.predict_example("C", stage, algorithm)
                    predict.assert_called_once()


class RoundcastExampleTests(unittest.TestCase):
    def test_archived_references_match_all_four_official_csvs(self):
        registry = json.loads((ROOT / "examples/roundcast_v1_cases.json").read_text(encoding="utf-8"))
        service = RoundcastService(ROOT)
        for model_id, source in registry["reference_sources"].items():
            frame = pd.read_csv(ROOT / source["path"], float_precision="round_trip")
            keys = ["series_id", "game_id", "round_id"]
            self.assertFalse(frame.duplicated(keys).any())
            frame = frame.set_index(keys)
            self.assertEqual(len(frame), 4172 if model_id.endswith("pre_round") else 4170)
            for case in registry["cases"]:
                example_id = case["example_id"]
                row = frame.loc[tuple(case["identity"][key] for key in keys)]
                self.assertEqual(row["y_true"], service.outcome(example_id)["ct_win"])
                self.assertAlmostEqual(row[source["probability_column"]],
                                       registry["reference_probabilities"][example_id][model_id], delta=1e-15)

    def test_reference_probabilities_are_not_read_into_snapshot_or_listing(self):
        baseline = RoundcastService(ROOT)
        registry = json.loads((ROOT / "examples/roundcast_v1_cases.json").read_text(encoding="utf-8"))
        registry["reference_probabilities"] = {"invalid": "must not be used for live results"}
        with patch("src.csdemo.roundcast_service._read_registry", return_value=registry):
            changed = RoundcastService(ROOT)
        self.assertEqual(changed.examples(), baseline.examples())
        for example_id in ("A", "B", "C"):
            for stage in ("pre_round", "post_first_kill"):
                self.assertEqual(changed.snapshot(example_id, stage), baseline.snapshot(example_id, stage))

    def test_frozen_split_disagreement_and_duplicate_series_fail(self):
        original = pd.read_csv
        for fault in ("split", "duplicate"):
            def changed(*args, **kwargs):
                frame = original(*args, **kwargs)
                mask = frame["series_id"].eq("1d78bfe0-e598-4a95-b5c0-2227e976dc5d")
                if fault == "duplicate":
                    return pd.concat([frame, frame.loc[mask]], ignore_index=True)
                frame.loc[mask, "split"] = "val"
                return frame
            with self.subTest(fault=fault):
                with patch("src.csdemo.roundcast_service.pd.read_csv", side_effect=changed):
                    with self.assertRaises(RoundcastValidationError):
                        RoundcastService(ROOT)

    def test_all_three_examples_have_only_27_or_31_allowed_input_fields(self):
        service = RoundcastService(ROOT)
        self.assertEqual([e["example_id"] for e in service.examples()], ["A", "B", "C"])
        forbidden = {"ct_win", "winner", "split", "series_id", "game_id", "round_id"}
        for case_id, map_name, number in (("A", "de_ancient", 4), ("B", "de_mirage", 11), ("C", "de_nuke", 17)):
            before = service.snapshot(case_id, "pre_round")
            after = service.snapshot(case_id, "post_first_kill")
            self.assertEqual(before["identity"], after["identity"])
            self.assertEqual(before["features"]["map_name"], map_name)
            self.assertEqual(before["features"]["round_num"], number)
            self.assertEqual(len(before["features"]), 27)
            self.assertEqual(len(after["features"]), 31)
            for key in BASE_FEATURES:
                self.assertEqual(before["features"][key], after["features"][key])
            for snapshot in (before, after):
                self.assertFalse(forbidden & snapshot["features"].keys())
                self.assertNotIn("ct_win", snapshot)
                self.assertEqual(snapshot["validation"]["status"], "passed")
            self.assertEqual(service.outcome(case_id)["winning_side"], "CT" if case_id == "A" else "T")
        self.assertEqual(len(service.readiness_report()["cases"]), 3)

    def test_unknown_examples_and_stages_fail(self):
        service = RoundcastService(ROOT)
        for case_id, stage in (("D", "pre_round"), ("A", "first_kill")):
            with self.assertRaises(RoundcastValidationError):
                service.snapshot(case_id, stage)
        with self.assertRaises(RoundcastValidationError):
            service.outcome("D")

    def test_returned_snapshots_cannot_mutate_future_requests(self):
        service = RoundcastService(ROOT)
        first = service.snapshot("A", "pre_round")
        first["features"]["ct_cash"] = -1
        first["identity"]["round_id"] = "wrong"
        self.assertNotEqual(first, service.snapshot("A", "pre_round"))

    def test_duplicate_case_keys_and_duplicate_ids_are_rejected(self):
        registry = json.loads((ROOT / "examples/roundcast_v1_cases.json").read_text(encoding="utf-8"))
        for field in ("identity", "example_id"):
            bad = deepcopy(registry)
            bad["cases"][1][field] = deepcopy(bad["cases"][0][field])
            with patch("src.csdemo.roundcast_service._read_registry", return_value=bad):
                with self.assertRaisesRegex(RoundcastValidationError, "Duplicate"):
                    RoundcastService(ROOT)

    def test_changed_parquet_semantics_fail_even_if_file_layer_was_verified(self):
        original = pd.read_parquet
        for fault, message in (("duplicate", "Duplicate"), ("split", "test"),
                               ("label", "label"), ("base", "purchase"), ("invalid", "Input")):
            calls = []

            def changed(*args, **kwargs):
                frame = original(*args, **kwargs)
                calls.append(1)
                if len(calls) != 2:
                    return frame
                mask = frame["round_id"].eq("online:946b0351-728d-41c6-9964-9b20f21df71d_4")
                if fault == "duplicate":
                    return pd.concat([frame, frame.loc[mask]], ignore_index=True)
                column, value = {"split": ("split", "train"), "label": ("ct_win", 0),
                                 "base": ("ct_cash", 999), "invalid": ("first_kill_time", -1)}[fault]
                frame.loc[mask, column] = value
                return frame

            with self.subTest(fault=fault):
                with patch("src.csdemo.roundcast_service.pd.read_parquet", side_effect=changed):
                    with self.assertRaisesRegex(RoundcastValidationError, message):
                        RoundcastService(ROOT)


if __name__ == "__main__":
    unittest.main()

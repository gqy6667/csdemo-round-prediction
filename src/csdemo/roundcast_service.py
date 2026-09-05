"""Trusted ROUNDCAST snapshots and explicit, bounded frozen-model inference."""
from __future__ import annotations

import hashlib
import json
import math
from time import perf_counter
from uuid import uuid4
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pandas as pd
import joblib

from .predict_pre_round import BASE_FEATURES, InputValidationError, PreRoundPredictor, validate_snapshot
from .calibration import IdentityCalibrator
from .predict_first_kill import FirstKillInputValidationError, validate_first_kill_snapshot
from .predict_first_kill import FirstKillPredictor
from .predict_first_kill_lightgbm import PostFirstKillLightGBMPredictor
from .predict_pre_round_lightgbm import PreRoundLightGBMPredictor
from .m16_first_kill_baselines import FIRST_KILL_MODEL_FEATURES

MODEL_FILES = {
    ("pre_round", "xgboost"): (
        "xgb_pre_round", "models/esta_full_m8_tuned/pre_round_xgb.joblib",
        "models/esta_full_m10/pre_round_calibrator.joblib",
    ),
    ("pre_round", "lightgbm"): (
        "lgbm_pre_round", "models/esta_full_m23/pre_round_lightgbm_tuned.joblib",
        "models/esta_full_m24/pre_round_lightgbm_calibrator.joblib",
    ),
    ("post_first_kill", "xgboost"): (
        "xgb_post_first_kill", "models/esta_full_m17/first_kill_xgboost_tuned.joblib",
        "models/esta_full_m18/first_kill_calibrator.joblib",
    ),
    ("post_first_kill", "lightgbm"): (
        "lgbm_post_first_kill", "models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib",
        "models/esta_full_m30/post_first_kill_lightgbm_calibrator.joblib",
    ),
}
DATA_FILES = {
    "pre_round": "data/processed/esta_full/pre_round.parquet",
    "post_first_kill": "data/processed/esta_full/first_kill.parquet",
}
PREDICTOR_ROUTES = {
    ("pre_round", "xgboost"): (PreRoundPredictor, "pre_round", "M8-tuned / M10-identity", "M13:27-base+9-derived"),
    ("pre_round", "lightgbm"): (PreRoundLightGBMPredictor, "pre_round", "M23-tuned / M24-identity", "M26:27-base+9-derived"),
    ("post_first_kill", "xgboost"): (FirstKillPredictor, "first_kill", "M17-tuned / M18-identity", "M20:31-base+9-derived"),
    ("post_first_kill", "lightgbm"): (PostFirstKillLightGBMPredictor, "post_first_kill", "M29-tuned / M30-identity", "M32:31-base+9-derived"),
}
KEY_FIELDS = ("series_id", "game_id", "round_id")
SPLIT_FILES = tuple(f"reports/esta_full_m{n}/split_assignments.csv" for n in (14, 21, 27, 33))
CATEGORY_FILE = "reports/esta_full_m32/model_contract_audit.json"


class RoundcastValidationError(ValueError):
    """A trusted source or fixed example failed validation; never use fallback data."""


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RoundcastValidationError("Duplicate JSON registry key")
        result[key] = value
    return result


def _read_registry(root: Path) -> dict:
    try:
        return json.loads(
            (root / "examples/roundcast_v1_cases.json").read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except RoundcastValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise RoundcastValidationError("Case registry missing or invalid") from exc


class RoundcastService:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path(__file__).resolve().parents[2]).resolve()
        self._registry = _read_registry(self.root)
        self._verified_bytes: dict[str, bytes] = {}
        self._proof: list[dict] = []
        self._snapshots: dict[tuple[str, str], dict] = {}
        self._outcomes: dict[str, dict] = {}
        self._case_proof: list[dict] = []
        try:
            self._verify_sources()
            self._prepare_examples()
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            if isinstance(exc, RoundcastValidationError):
                raise
            raise RoundcastValidationError("Invalid trusted registry or source structure") from exc

    def _check_file(self, relative: str, expected: str) -> bytes:
        candidate = Path(relative)
        if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
            raise RoundcastValidationError("Source path outside trusted root")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise RoundcastValidationError("Source path outside trusted root")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise RoundcastValidationError(f"Trusted file missing or unreadable: {relative}") from exc
        if hashlib.sha256(content).hexdigest() != expected:
            raise RoundcastValidationError(f"Trusted file hash mismatch: {relative}")
        return content

    def _verify_sources(self) -> None:
        registry = self._registry
        if not isinstance(registry, dict) or any(
            not isinstance(registry.get(field), dict)
            for field in ("files", "manifest_pins", "reference_sources")
        ):
            raise RoundcastValidationError("Invalid registry mapping structure")
        required = set(DATA_FILES.values()) | {p for route in MODEL_FILES.values() for p in route[1:]}
        if registry.get("schema_version") != 1 or set(registry["frozen_artifacts"]) != required:
            raise RoundcastValidationError("Invalid frozen artifact registry")
        if set(registry["reference_sources"]) != {route[0] for route in MODEL_FILES.values()}:
            raise RoundcastValidationError("Missing four-model reference source mapping")
        support = set(SPLIT_FILES) | {CATEGORY_FILE} | {
            source["path"] for source in registry["reference_sources"].values()
        }
        if not (required | support).issubset(registry["files"]):
            raise RoundcastValidationError("Trusted artifact or supporting file missing from registry")
        manifests = {p: json.loads(self._check_file(p, h)) for p, h in registry["manifest_pins"].items()}
        for relative, record in registry["files"].items():
            content = self._check_file(relative, record["sha256"])
            if len(content) != record["bytes"]:
                raise RoundcastValidationError(f"Trusted file size mismatch: {relative}")
            source = record["provenance"]
            if source["kind"] == "frozen_manifest":
                entry = manifests[source["manifest"]]
                for key in source["entry"]:
                    entry = entry[key]
                recorded_path = entry["path"].replace("\\", "/")
                if not (recorded_path == relative or recorded_path.endswith("/" + relative)):
                    raise RoundcastValidationError(f"Manifest path mismatch: {relative}")
                if entry["sha256"] != record["sha256"] or entry["bytes"] != len(content):
                    raise RoundcastValidationError(f"Manifest hash or size mismatch: {relative}")
            elif source["kind"] != "initial_pin" or relative in required:
                raise RoundcastValidationError(f"Missing frozen manifest provenance: {relative}")
            self._verified_bytes[relative] = content
            self._proof.append({"path": relative, "expected_sha256": record["sha256"],
                                "actual_sha256": hashlib.sha256(content).hexdigest(), "provenance": deepcopy(source)})

    def model_metadata(self, stage: str, algorithm: str) -> dict:
        if not isinstance(stage, str) or not isinstance(algorithm, str) or (stage, algorithm) not in MODEL_FILES:
            raise RoundcastValidationError("Unknown stage/algorithm combination")
        model_id, model, calibrator = MODEL_FILES[stage, algorithm]
        return {"model_id": model_id, "stage": stage, "algorithm": algorithm,
                "model_sha256": self._registry["files"][model]["sha256"],
                "calibrator_sha256": self._registry["files"][calibrator]["sha256"],
                "inference_ready": True, "available_examples": ["A", "B", "C"]}

    def predict_example(self, example_id: str, stage: str, algorithm: str) -> dict:
        metadata = self.model_metadata(stage, algorithm)
        snapshot = self.snapshot(example_id, stage)
        predictor_class, internal_task, model_version, feature_version = PREDICTOR_ROUTES[stage, algorithm]
        started = perf_counter()
        _, model_path, calibrator_path = MODEL_FILES[stage, algorithm]
        # Recheck active inputs on every run, then deserialize these exact bytes.
        current = {p: self._check_file(p, self._registry["files"][p]["sha256"])
                   for p in (model_path, calibrator_path, DATA_FILES[stage])}
        try:
            model = joblib.load(BytesIO(current[model_path]))
            calibration = joblib.load(BytesIO(current[calibrator_path]))
            if calibration["method"] != "uncalibrated" or type(calibration["calibrator"]) is not IdentityCalibrator:
                raise ValueError("Expected frozen identity calibration")
            kwargs = {} if predictor_class is PreRoundPredictor else {"model_sha256": metadata["model_sha256"]}
            result = predictor_class(model, calibration, **kwargs).predict(snapshot["features"])
            expected_keys = {"task", "snapshot_definition", "calibration_method", "validation", "prediction"}
            if algorithm == "lightgbm":
                expected_keys |= {"model_name", "model_sha256"}
            if set(result) != expected_keys:
                raise ValueError("Unexpected predictor fields")
            probability = result["prediction"]
            ct, t = probability["ct_win_probability"], probability["t_win_probability"]
            base = probability["base_ct_win_probability"] if algorithm == "lightgbm" else ct
            if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1 for v in (ct, t, base))
                    or abs(ct + t - 1) > 1e-12 or probability["decision_threshold"] != 0.5
                    or probability["predicted_side"] != ("CT" if ct >= 0.5 else "T")
                    or result["task"] != internal_task or result["calibration_method"] != "uncalibrated"
                    or abs(base - ct) > 1e-15
                    or probability["probability_sum"] != ct + t
                    or result["validation"]["status"] != "passed"):
                raise ValueError("Invalid predictor response")
            if algorithm == "lightgbm" and (result["model_name"] != "lightgbm_tuned" or result["model_sha256"] != metadata["model_sha256"]):
                raise ValueError("Predictor model identity mismatch")
            validation = result["validation"]
            if validation["encoded_model_feature_count"] != (43 if stage == "pre_round" else 82):
                raise ValueError("Predictor feature count mismatch")
            validation["required_input_field_count"] = len(snapshot["features"])
        except Exception as exc:
            raise RoundcastValidationError("Inference unavailable; no fallback prediction was used") from exc
        return {"task": stage, "predictor_task": internal_task,
                "snapshot_definition": result["snapshot_definition"], "calibration_method": result["calibration_method"],
                "validation": validation,
                "prediction": {k: probability[k] for k in ("ct_win_probability", "t_win_probability", "predicted_side", "decision_threshold", "probability_sum")},
                **metadata, "status": "success", "example_id": example_id,
                "request_id": str(uuid4()), "identity": snapshot["identity"],
                "base_ct_win_probability": base, "model_version": model_version,
                "feature_version": feature_version,
                "inference_ms": (perf_counter() - started) * 1000,
                "source": {"snapshot": result["snapshot_definition"],
                           "data_version": "esta_full/pre_round" if stage == "pre_round" else "esta_full/first_kill",
                           "purchase_state": "purchase_end",
                           "data_sha256": self._registry["files"][DATA_FILES[stage]]["sha256"],
                           "split": "test", "exact_frame_alignment": "not_provided",
                           "timing_note": "Elapsed time includes active source rechecks, loading and prediction"}}

    def _prepare_examples(self) -> None:
        cases = self._registry["cases"]
        identifiers = [case["example_id"] for case in cases]
        identities = [tuple(case["identity"][k] for k in KEY_FIELDS) for case in cases]
        if len(set(identifiers)) != len(identifiers) or len(set(identities)) != len(identities):
            raise RoundcastValidationError("Duplicate case ID or round key")
        if identifiers != ["A", "B", "C"] or any(set(c["identity"]) != set(KEY_FIELDS) for c in cases):
            raise RoundcastValidationError("Expected exactly three fixed case identities")
        contract = json.loads(self._verified_bytes[CATEGORY_FILE])["model_contract"]
        frames = {}
        for stage, relative in DATA_FILES.items():
            fields = list(BASE_FEATURES) + (FIRST_KILL_MODEL_FEATURES if stage == "post_first_kill" else [])
            frame = pd.read_parquet(BytesIO(self._verified_bytes[relative]), columns=[*KEY_FIELDS, "ct_win", "split", *fields])
            if frame.duplicated(list(KEY_FIELDS)).any():
                raise RoundcastValidationError(f"Duplicate round key in {stage} data")
            frames[stage] = frame.set_index(list(KEY_FIELDS))
        splits = {}
        for relative in SPLIT_FILES:
            split = pd.read_csv(BytesIO(self._verified_bytes[relative]), usecols=["series_id", "split"])
            if split["series_id"].duplicated().any():
                raise RoundcastValidationError("Duplicate series key in split source")
            splits[relative] = split.set_index("series_id")["split"].to_dict()
        for case, key in zip(cases, identities):
            example_id = case["example_id"]
            if any(mapping.get(key[0]) != "test" for mapping in splits.values()):
                raise RoundcastValidationError(f"Case {example_id} is not test in all four frozen splits")
            labels, snapshots = [], {}
            for stage, frame in frames.items():
                if key not in frame.index:
                    raise RoundcastValidationError(f"Case {example_id} missing from {stage} data")
                row = frame.loc[key].to_dict()
                if row["split"] != "test":
                    raise RoundcastValidationError(f"Case {example_id} is not common test")
                labels.append(row["ct_win"])
                fields = list(BASE_FEATURES) + (FIRST_KILL_MODEL_FEATURES if stage == "post_first_kill" else [])
                raw = {f: row[f] for f in fields}
                if stage == "post_first_kill" and any(raw[f] != snapshots["pre_round"]["features"][f] for f in BASE_FEATURES):
                    raise RoundcastValidationError(f"Case {example_id} purchase fields differ between stages")
                try:
                    if stage == "pre_round":
                        normalized, validation = validate_snapshot(raw, contract["known_maps"])
                    else:
                        normalized, validation = validate_first_kill_snapshot(raw, contract["known_maps"], contract["known_weapons"])
                except (InputValidationError, FirstKillInputValidationError) as exc:
                    raise RoundcastValidationError(f"Input validation failed for case {example_id}, {stage}") from exc
                snapshots[stage] = {"example_id": example_id, "stage": stage, "identity": deepcopy(case["identity"]),
                                    "features": {f: normalized[f] for f in fields}, "validation": validation}
            outcome = self._registry["outcomes"][example_id]
            if labels[0] not in (0, 1) or labels[0] != labels[1] or labels[0] != outcome["ct_win"]:
                raise RoundcastValidationError(f"Case {example_id} label mismatch")
            if outcome["winning_side"] != ("CT" if labels[0] == 1 else "T"):
                raise RoundcastValidationError(f"Case {example_id} outcome label mismatch")
            self._outcomes[example_id] = deepcopy(outcome)
            for stage, snapshot in snapshots.items():
                self._snapshots[example_id, stage] = snapshot
            self._case_proof.append({"example_id": example_id, "identity": deepcopy(case["identity"]),
                                     "common_test": True, "four_split_sources": list(SPLIT_FILES), "labels_agree": True,
                                     "purchase_fields_agree": True, "base_feature_counts": [27, 31], "validation": "passed"})

    def snapshot(self, example_id: str, stage: str) -> dict:
        if not isinstance(example_id, str) or not isinstance(stage, str) or (example_id, stage) not in self._snapshots:
            raise RoundcastValidationError("Unknown case or stage")
        return deepcopy(self._snapshots[example_id, stage])

    def outcome(self, example_id: str) -> dict:
        if not isinstance(example_id, str) or example_id not in self._outcomes:
            raise RoundcastValidationError("Unknown case")
        return deepcopy(self._outcomes[example_id])

    def examples(self) -> list[dict]:
        return [{"example_id": e, "name": f"案例 {e}", "identity": self.snapshot(e, "pre_round")["identity"],
                 "map_name": self.snapshot(e, "pre_round")["features"]["map_name"],
                 "round_num": self.snapshot(e, "pre_round")["features"]["round_num"]} for e in ("A", "B", "C")]

    def readiness_report(self) -> dict:
        return {"status": "passed", "inference_executed": False,
                "cases": deepcopy(self._case_proof),
                "frozen_artifacts": deepcopy([p for p in self._proof if p["path"] in self._registry["frozen_artifacts"]]),
                "supporting_files": deepcopy([p for p in self._proof if p["path"] not in self._registry["frozen_artifacts"]])}

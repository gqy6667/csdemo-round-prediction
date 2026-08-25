from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .m15_first_kill_data import fingerprint_file
from .predict_pre_round import (
    InputValidationError,
    known_maps_from_columns,
    load_snapshot,
    validate_snapshot,
)
from .schema import PRE_ROUND_FEATURES
from .train_xgb import prepare_features


SNAPSHOT_DEFINITION = "freeze-time end after purchases and before combat"
EXPECTED_PROFILE = "M14_pre_round_features"
EXPECTED_MODEL_NAME = "lightgbm_tuned"
EXPECTED_RAW_FEATURE_COUNT = 36
EXPECTED_ENCODED_FEATURE_COUNT = 43
EXPECTED_DEPLOYMENT_TREE_COUNT = 115
EXPECTED_MAP_COUNT = 8


def validate_lightgbm_model_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "model",
        "task",
        "definition",
        "model_name",
        "profile",
        "raw_features",
        "columns",
        "params",
        "best_iteration",
        "data_sha256",
        "selection_metric",
        "official_seed",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError("Model bundle is missing fields: " + ", ".join(missing))
    if bundle["task"] != "pre_round":
        raise ValueError(
            f"Model bundle task must be 'pre_round'; got {bundle['task']!r}"
        )
    if bundle["model_name"] != EXPECTED_MODEL_NAME:
        raise ValueError(
            f"Model bundle name must be {EXPECTED_MODEL_NAME!r}; "
            f"got {bundle['model_name']!r}"
        )
    if bundle["profile"] != EXPECTED_PROFILE:
        raise ValueError(
            f"Model bundle profile must be {EXPECTED_PROFILE!r}; "
            f"got {bundle['profile']!r}"
        )
    if bundle["definition"] != SNAPSHOT_DEFINITION:
        raise ValueError("Model snapshot definition differs from the M26 contract")

    raw_features = list(bundle["raw_features"])
    if raw_features != list(PRE_ROUND_FEATURES):
        raise ValueError("Model bundle raw feature contract differs from M14/M25")
    columns = list(bundle["columns"])
    if len(columns) != len(set(columns)):
        raise ValueError("Model bundle encoded columns must be unique")
    if len(raw_features) != EXPECTED_RAW_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RAW_FEATURE_COUNT} raw features; got {len(raw_features)}"
        )
    if len(columns) != EXPECTED_ENCODED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded features; "
            f"got {len(columns)}"
        )

    model = bundle["model"]
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("Model bundle must contain a fitted LightGBM booster")
    booster_columns = list(booster.feature_name())
    if booster_columns != columns:
        raise ValueError("LightGBM booster feature order differs from bundle columns")
    best_iteration = int(bundle["best_iteration"])
    model_best_iteration = int(getattr(model, "best_iteration_", -1))
    booster_tree_count = int(booster.num_trees())
    if not (
        best_iteration
        == model_best_iteration
        == booster_tree_count
        == EXPECTED_DEPLOYMENT_TREE_COUNT
    ):
        raise ValueError(
            "LightGBM deployment tree contract must equal "
            f"{EXPECTED_DEPLOYMENT_TREE_COUNT}; bundle={best_iteration}, "
            f"model={model_best_iteration}, booster={booster_tree_count}"
        )
    model_feature_count = int(getattr(model, "n_features_in_", -1))
    if model_feature_count != len(columns):
        raise ValueError(
            "Model encoded feature count differs from the saved column contract"
        )
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("LightGBM model must provide predict_proba()")

    known_maps = known_maps_from_columns(columns)
    if len(known_maps) != EXPECTED_MAP_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MAP_COUNT} encoded maps; got {len(known_maps)}"
        )
    data_sha256 = str(bundle["data_sha256"])
    if len(data_sha256) != 64:
        raise ValueError("Model data SHA-256 must be a 64-character digest")
    return {
        "passed": True,
        "task": bundle["task"],
        "model_name": bundle["model_name"],
        "profile": bundle["profile"],
        "definition": bundle["definition"],
        "raw_feature_count": len(raw_features),
        "encoded_feature_count": len(columns),
        "known_map_count": len(known_maps),
        "known_maps": sorted(known_maps),
        "deployment_tree_count": booster_tree_count,
        "data_sha256": data_sha256,
        "selection_metric": str(bundle["selection_metric"]),
        "official_seed": int(bundle["official_seed"]),
    }


def validate_pre_round_calibrator_bundle(
    bundle: Mapping[str, Any],
    *,
    model_sha256: str,
    model_data_sha256: str,
) -> dict[str, Any]:
    required = {
        "calibrator",
        "method",
        "task",
        "base_model_sha256",
        "data_sha256",
        "selection_data",
        "validation_folds",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError("Calibrator bundle is missing fields: " + ", ".join(missing))
    if bundle["task"] != "pre_round":
        raise ValueError(
            f"Calibrator task must be 'pre_round'; got {bundle['task']!r}"
        )
    if bundle["method"] != "uncalibrated":
        raise ValueError(
            "M26 requires the M24 uncalibrated identity method; "
            f"got {bundle['method']!r}"
        )
    if bundle["base_model_sha256"] != model_sha256:
        raise ValueError("Calibrator base model SHA-256 differs from the loaded model")
    if bundle["data_sha256"] != model_data_sha256:
        raise ValueError("Calibrator data SHA-256 differs from the model data contract")
    if bundle["selection_data"] != "validation only":
        raise ValueError(
            "Calibrator selection_data must be 'validation only'; "
            f"got {bundle['selection_data']!r}"
        )
    if int(bundle["validation_folds"]) != 5:
        raise ValueError("Calibrator validation_folds must equal 5")
    if not callable(getattr(bundle["calibrator"], "predict", None)):
        raise ValueError("Calibrator object must provide predict()")
    return {
        "passed": True,
        "task": bundle["task"],
        "method": bundle["method"],
        "base_model_sha256": bundle["base_model_sha256"],
        "data_sha256": bundle["data_sha256"],
        "selection_data": bundle["selection_data"],
        "validation_folds": int(bundle["validation_folds"]),
    }


class PreRoundLightGBMPredictor:
    """Run one strict purchase-end prediction with frozen M23/M24 artifacts."""

    def __init__(
        self,
        model_bundle: Mapping[str, Any],
        calibrator_bundle: Mapping[str, Any],
        *,
        model_sha256: str,
    ) -> None:
        self.model_audit = validate_lightgbm_model_bundle(model_bundle)
        self.calibrator_audit = validate_pre_round_calibrator_bundle(
            calibrator_bundle,
            model_sha256=model_sha256,
            model_data_sha256=str(model_bundle["data_sha256"]),
        )
        self.model = model_bundle["model"]
        self.columns = list(model_bundle["columns"])
        self.raw_features = list(model_bundle["raw_features"])
        self.known_maps = known_maps_from_columns(self.columns)
        self.calibrator = calibrator_bundle["calibrator"]
        self.calibration_method = str(calibrator_bundle["method"])
        self.model_sha256 = model_sha256

    @classmethod
    def from_paths(
        cls,
        model_path: str | Path,
        calibrator_path: str | Path,
    ) -> "PreRoundLightGBMPredictor":
        model_path = Path(model_path)
        calibrator_path = Path(calibrator_path)
        model_bundle = joblib.load(model_path)
        calibrator_bundle = joblib.load(calibrator_path)
        if not isinstance(model_bundle, Mapping):
            raise ValueError("Model artifact must contain a mapping bundle")
        if not isinstance(calibrator_bundle, Mapping):
            raise ValueError("Calibrator artifact must contain a mapping bundle")
        return cls(
            model_bundle,
            calibrator_bundle,
            model_sha256=fingerprint_file(model_path)["sha256"],
        )

    def validate(
        self,
        snapshot: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized, details = validate_snapshot(snapshot, self.known_maps)
        if list(normalized) != self.raw_features:
            raise RuntimeError("Validated raw features differ from the model contract")
        details.update(
            {
                "raw_model_feature_count": len(self.raw_features),
                "model_contract_verified": self.model_audit["passed"],
                "calibrator_contract_verified": self.calibrator_audit["passed"],
                "deployment_tree_count": self.model_audit[
                    "deployment_tree_count"
                ],
            }
        )
        return normalized, details

    def predict(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        normalized, validation = self.validate(snapshot)
        frame = pd.DataFrame([normalized], columns=self.raw_features)
        encoded = prepare_features(frame).reindex(columns=self.columns, fill_value=0)
        if encoded.columns.tolist() != self.columns or encoded.shape != (
            1,
            EXPECTED_ENCODED_FEATURE_COUNT,
        ):
            raise RuntimeError("Encoded inference columns do not match the model contract")

        base_probability = float(self.model.predict_proba(encoded)[0, 1])
        if not math.isfinite(base_probability) or not 0.0 <= base_probability <= 1.0:
            raise ValueError(
                "Model produced an invalid base CT win probability: "
                f"{base_probability!r}"
            )
        ct_probability = float(
            np.asarray(self.calibrator.predict([base_probability])).reshape(-1)[0]
        )
        if not math.isfinite(ct_probability) or not 0.0 <= ct_probability <= 1.0:
            raise ValueError(
                "Calibrator produced an invalid CT win probability: "
                f"{ct_probability!r}"
            )
        if self.calibration_method == "uncalibrated" and not math.isclose(
            base_probability,
            ct_probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Uncalibrated identity bundle changed the model probability")

        t_probability = 1.0 - ct_probability
        validation["encoded_model_feature_count"] = len(self.columns)
        return {
            "task": "pre_round",
            "model_name": EXPECTED_MODEL_NAME,
            "snapshot_definition": SNAPSHOT_DEFINITION,
            "calibration_method": self.calibration_method,
            "model_sha256": self.model_sha256,
            "validation": validation,
            "prediction": {
                "base_ct_win_probability": base_probability,
                "ct_win_probability": ct_probability,
                "t_win_probability": t_probability,
                "predicted_side": "CT" if ct_probability >= 0.5 else "T",
                "decision_threshold": 0.5,
                "probability_sum": ct_probability + t_probability,
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict one CS round at purchase end with the frozen M23 LightGBM."
        )
    )
    parser.add_argument("--input", required=True, help="One JSON object or one CSV row")
    parser.add_argument("--model", required=True, help="Saved M23 LightGBM bundle")
    parser.add_argument(
        "--calibrator", required=True, help="Associated M24 calibration bundle"
    )
    parser.add_argument("--output", help="Optional path for the result JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        predictor = PreRoundLightGBMPredictor.from_paths(
            args.model, args.calibrator
        )
        result = predictor.predict(load_snapshot(args.input))
    except (
        InputValidationError,
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        error = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "errors": (
                list(exc.errors)
                if isinstance(exc, InputValidationError)
                else [str(exc)]
            ),
        }
        print(json.dumps(error, indent=2, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from exc

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

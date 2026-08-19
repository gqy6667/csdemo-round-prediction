from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .m15_first_kill_data import fingerprint_file
from .m16_first_kill_baselines import (
    FIRST_KILL_MODEL_FEATURES,
    REDUNDANT_FIRST_KILL_FEATURES,
    canonical_feature_names,
)
from .predict_pre_round import (
    BASE_FEATURES as PURCHASE_BASE_FEATURES,
    DIFFERENCE_FEATURES,
    InputValidationError as PreRoundInputValidationError,
    load_snapshot,
    validate_snapshot as validate_purchase_snapshot,
)
from .train_xgb import prepare_features


SNAPSHOT_DEFINITION = (
    "purchase complete, immediately after earliest valid enemy kill"
)
FIRST_KILL_TIME_RANGE = (0.0, 180.0)
EXPECTED_RAW_FEATURE_COUNT = 40
EXPECTED_ENCODED_FEATURE_COUNT = 82
EXPECTED_DEPLOYMENT_TREE_COUNT = 409

FORBIDDEN_EXACT_FIELDS = {
    "series_id",
    "game_id",
    "round_id",
    "split",
    "ct_win",
    "winner",
    "winning_side",
    "round_winner",
    *REDUNDANT_FIRST_KILL_FEATURES,
}
FORBIDDEN_PREFIXES = (
    "second_kill",
    "second_death",
    "next_kill",
    "damage",
    "health",
    "hp_",
    "bomb_",
    "plant_",
    "defuse_",
    "round_end",
)


class FirstKillInputValidationError(ValueError):
    """Raised when one post-first-kill snapshot violates the M20 contract."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        message = "Input validation failed:\n" + "\n".join(
            f"- {error}" for error in self.errors
        )
        super().__init__(message)


def _as_integer(field: str, value: Any, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append(f"{field} must be an integer; got {value!r}")
        return None
    if not math.isfinite(float(value)) or not float(value).is_integer():
        errors.append(f"{field} must be a finite integer; got {value!r}")
        return None
    return int(value)


def _as_finite_number(field: str, value: Any, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append(f"{field} must be a finite number; got {value!r}")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{field} must be a finite number; got {value!r}")
        return None
    return number


def _is_forbidden(field: str) -> bool:
    lowered = field.lower()
    return field in FORBIDDEN_EXACT_FIELDS or lowered.startswith(FORBIDDEN_PREFIXES)


def categories_from_columns(
    columns: Iterable[str], prefix: str, *, label: str
) -> set[str]:
    values = {column[len(prefix) :] for column in columns if column.startswith(prefix)}
    if not values:
        raise ValueError(f"Model bundle does not contain encoded {label} columns.")
    return values


def validate_first_kill_snapshot(
    snapshot: Mapping[str, Any],
    known_maps: Iterable[str],
    known_weapons: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one M20 snapshot and return the exact 40 raw model features."""

    if not isinstance(snapshot, Mapping):
        raise FirstKillInputValidationError(
            ["input must be one JSON object or one CSV row"]
        )

    errors: list[str] = []
    event_features = tuple(FIRST_KILL_MODEL_FEATURES)
    purchase_fields = set(PURCHASE_BASE_FEATURES) | set(DIFFERENCE_FEATURES)
    allowed_fields = purchase_fields | set(event_features)

    missing_events = [field for field in event_features if field not in snapshot]
    if missing_events:
        errors.append(
            "missing required first-kill fields: " + ", ".join(missing_events)
        )
        errors.extend(f"{field}: required field is missing" for field in missing_events)

    unexpected = sorted(set(snapshot) - allowed_fields)
    forbidden = [field for field in unexpected if _is_forbidden(field)]
    unknown = [field for field in unexpected if field not in forbidden]
    if forbidden:
        errors.append("forbidden fields: " + ", ".join(forbidden))
    if unknown:
        errors.append("unknown fields: " + ", ".join(unknown))

    purchase_snapshot = {
        field: snapshot[field] for field in snapshot if field in purchase_fields
    }
    purchase_values: dict[str, Any] | None = None
    purchase_details: dict[str, Any] | None = None
    try:
        purchase_values, purchase_details = validate_purchase_snapshot(
            purchase_snapshot, known_maps
        )
    except PreRoundInputValidationError as exc:
        errors.extend(exc.errors)

    event_values: dict[str, Any] = {}
    if "first_kill_advantage_ct" in snapshot:
        advantage = _as_integer(
            "first_kill_advantage_ct",
            snapshot["first_kill_advantage_ct"],
            errors,
        )
        if advantage is not None:
            event_values["first_kill_advantage_ct"] = advantage
            if advantage not in {-1, 1}:
                errors.append(
                    "first_kill_advantage_ct must be -1 or 1; "
                    f"got {advantage}"
                )

    if "first_kill_time" in snapshot:
        kill_time = _as_finite_number(
            "first_kill_time", snapshot["first_kill_time"], errors
        )
        if kill_time is not None:
            event_values["first_kill_time"] = kill_time
            minimum, maximum = FIRST_KILL_TIME_RANGE
            if not minimum <= kill_time <= maximum:
                errors.append(
                    "first_kill_time must be between 0 and 180 seconds; "
                    f"got {kill_time:g}"
                )

    if "first_kill_headshot" in snapshot:
        headshot = snapshot["first_kill_headshot"]
        if isinstance(headshot, bool):
            event_values["first_kill_headshot"] = int(headshot)
        else:
            normalized_headshot = _as_integer(
                "first_kill_headshot", headshot, errors
            )
            if normalized_headshot is not None:
                event_values["first_kill_headshot"] = normalized_headshot
                if normalized_headshot not in {0, 1}:
                    errors.append(
                        "first_kill_headshot must be a boolean or 0/1; "
                        f"got {normalized_headshot}"
                    )

    known_weapon_set = set(known_weapons)
    if "first_kill_weapon" in snapshot:
        weapon = snapshot["first_kill_weapon"]
        if not isinstance(weapon, str) or not weapon.strip():
            errors.append(
                "first_kill_weapon must be a non-empty string; "
                f"got {weapon!r}"
            )
        else:
            weapon = weapon.strip()
            event_values["first_kill_weapon"] = weapon
            if weapon not in known_weapon_set:
                errors.append(
                    f"first_kill_weapon {weapon!r} was not seen during training; "
                    f"choose one of {sorted(known_weapon_set)}"
                )

    if errors:
        raise FirstKillInputValidationError(errors)
    if purchase_values is None or purchase_details is None:
        raise RuntimeError("Purchase validation did not return normalized values.")

    combined = {**purchase_values, **event_values}
    raw_features = canonical_feature_names()
    normalized = {feature: combined[feature] for feature in raw_features}
    advantage = int(normalized["first_kill_advantage_ct"])
    details = {
        "status": "passed",
        "required_purchase_base_feature_count": len(PURCHASE_BASE_FEATURES),
        "required_first_kill_feature_count": len(event_features),
        "provided_field_count": len(snapshot),
        "derived_features": list(DIFFERENCE_FEATURES),
        "derived_feature_count": len(DIFFERENCE_FEATURES),
        "raw_model_feature_count": len(raw_features),
        "map_name": normalized["map_name"],
        "first_kill_side": "CT" if advantage == 1 else "T",
        "first_kill_weapon": normalized["first_kill_weapon"],
    }
    return normalized, details


def validate_model_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "model",
        "task",
        "profile",
        "raw_features",
        "columns",
        "data_sha256",
        "best_iteration",
        "best_tree_count",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError("Model bundle is missing fields: " + ", ".join(missing))
    if bundle["task"] != "first_kill":
        raise ValueError(
            f"Model bundle task must be 'first_kill'; got {bundle['task']!r}"
        )
    if bundle["profile"] != "canonical_event":
        raise ValueError(
            "Model bundle profile must be 'canonical_event'; "
            f"got {bundle['profile']!r}"
        )

    expected_raw = canonical_feature_names()
    raw_features = list(bundle["raw_features"])
    if raw_features != expected_raw:
        raise ValueError("Model bundle raw feature contract differs from M16/M19.")

    columns = list(bundle["columns"])
    if len(columns) != len(set(columns)):
        raise ValueError("Model bundle encoded columns must be unique.")
    if len(raw_features) != EXPECTED_RAW_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RAW_FEATURE_COUNT} raw features; got {len(raw_features)}."
        )
    if len(columns) != EXPECTED_ENCODED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ENCODED_FEATURE_COUNT} encoded features; got {len(columns)}."
        )

    best_iteration = int(bundle["best_iteration"])
    best_tree_count = int(bundle["best_tree_count"])
    model_best_iteration = int(getattr(bundle["model"], "best_iteration", -1))
    if not (
        best_iteration + 1
        == best_tree_count
        == EXPECTED_DEPLOYMENT_TREE_COUNT
        and model_best_iteration == best_iteration
    ):
        raise ValueError("Model bundle deployment tree contract does not match M17/M19.")

    model_feature_count = int(getattr(bundle["model"], "n_features_in_", -1))
    if model_feature_count != len(columns):
        raise ValueError(
            "Model encoded feature count differs from the saved column contract."
        )

    known_maps = categories_from_columns(columns, "map_name_", label="map_name")
    known_weapons = categories_from_columns(
        columns, "first_kill_weapon_", label="first_kill_weapon"
    )
    return {
        "passed": True,
        "task": bundle["task"],
        "profile": bundle["profile"],
        "raw_feature_count": len(raw_features),
        "encoded_feature_count": len(columns),
        "known_map_count": len(known_maps),
        "known_weapon_count": len(known_weapons),
        "deployment_tree_count": best_tree_count,
        "data_sha256": str(bundle["data_sha256"]),
    }


def validate_calibrator_bundle(
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
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError("Calibrator bundle is missing fields: " + ", ".join(missing))
    if bundle["task"] != "post_first_kill":
        raise ValueError(
            "Calibrator task must be 'post_first_kill'; "
            f"got {bundle['task']!r}"
        )
    if bundle["method"] != "uncalibrated":
        raise ValueError(
            "M20 requires the M18 uncalibrated identity method; "
            f"got {bundle['method']!r}"
        )
    if bundle["base_model_sha256"] != model_sha256:
        raise ValueError("Calibrator base model SHA-256 differs from the loaded model.")
    if bundle["data_sha256"] != model_data_sha256:
        raise ValueError("Calibrator data SHA-256 differs from the model data contract.")
    if not callable(getattr(bundle["calibrator"], "predict", None)):
        raise ValueError("Calibrator object must provide predict().")
    return {
        "passed": True,
        "task": bundle["task"],
        "method": bundle["method"],
        "base_model_sha256": bundle["base_model_sha256"],
        "data_sha256": bundle["data_sha256"],
    }


class FirstKillPredictor:
    """Load the frozen M17 model and associated M18 calibrator for inference."""

    def __init__(
        self,
        model_bundle: Mapping[str, Any],
        calibrator_bundle: Mapping[str, Any],
        *,
        model_sha256: str,
    ) -> None:
        self.model_audit = validate_model_bundle(model_bundle)
        self.calibrator_audit = validate_calibrator_bundle(
            calibrator_bundle,
            model_sha256=model_sha256,
            model_data_sha256=str(model_bundle["data_sha256"]),
        )
        self.model = model_bundle["model"]
        self.columns = list(model_bundle["columns"])
        self.raw_features = list(model_bundle["raw_features"])
        self.known_maps = categories_from_columns(
            self.columns, "map_name_", label="map_name"
        )
        self.known_weapons = categories_from_columns(
            self.columns, "first_kill_weapon_", label="first_kill_weapon"
        )
        self.calibrator = calibrator_bundle["calibrator"]
        self.calibration_method = str(calibrator_bundle["method"])
        self.model_sha256 = model_sha256

    @classmethod
    def from_paths(
        cls, model_path: str | Path, calibrator_path: str | Path
    ) -> "FirstKillPredictor":
        model_path = Path(model_path)
        calibrator_path = Path(calibrator_path)
        model_bundle = joblib.load(model_path)
        calibrator_bundle = joblib.load(calibrator_path)
        model_sha256 = fingerprint_file(model_path)["sha256"]
        return cls(
            model_bundle,
            calibrator_bundle,
            model_sha256=model_sha256,
        )

    def predict(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        normalized, validation = validate_first_kill_snapshot(
            snapshot, self.known_maps, self.known_weapons
        )
        frame = pd.DataFrame([normalized], columns=self.raw_features)
        encoded = prepare_features(frame).reindex(columns=self.columns, fill_value=0)
        if encoded.columns.tolist() != self.columns:
            raise RuntimeError("Encoded inference columns do not match the model contract.")

        base_probability = float(self.model.predict_proba(encoded)[0, 1])
        ct_probability = float(
            np.asarray(self.calibrator.predict([base_probability])).reshape(-1)[0]
        )
        if not math.isfinite(ct_probability) or not 0.0 <= ct_probability <= 1.0:
            raise ValueError(
                "Model produced an invalid CT win probability: "
                f"{ct_probability!r}"
            )

        t_probability = 1.0 - ct_probability
        validation.update(
            {
                "encoded_model_feature_count": len(self.columns),
                "model_contract_verified": self.model_audit["passed"],
                "calibrator_contract_verified": self.calibrator_audit["passed"],
                "deployment_tree_count": self.model_audit["deployment_tree_count"],
            }
        )
        return {
            "task": "first_kill",
            "snapshot_definition": SNAPSHOT_DEFINITION,
            "calibration_method": self.calibration_method,
            "validation": validation,
            "prediction": {
                "ct_win_probability": ct_probability,
                "t_win_probability": t_probability,
                "predicted_side": "CT" if ct_probability >= 0.5 else "T",
                "decision_threshold": 0.5,
                "probability_sum": ct_probability + t_probability,
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict one CS round immediately after the first valid enemy kill."
    )
    parser.add_argument("--input", required=True, help="One JSON object or one CSV row.")
    parser.add_argument("--model", required=True, help="Saved M17 XGBoost bundle.")
    parser.add_argument(
        "--calibrator", required=True, help="Associated M18 calibration bundle."
    )
    parser.add_argument("--output", help="Optional path for the result JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        predictor = FirstKillPredictor.from_paths(args.model, args.calibrator)
        result = predictor.predict(load_snapshot(args.input))
    except (
        FirstKillInputValidationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        error = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "errors": (
                list(exc.errors)
                if isinstance(exc, FirstKillInputValidationError)
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

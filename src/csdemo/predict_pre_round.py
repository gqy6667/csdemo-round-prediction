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

from .schema import PRE_ROUND_FEATURES
from .train_xgb import prepare_features


DIFFERENCE_FEATURES = {
    "score_diff_ct": ("ct_score", "t_score"),
    "eq_value_diff_ct": ("ct_eq_value", "t_eq_value"),
    "cash_diff_ct": ("ct_cash", "t_cash"),
    "armor_diff_ct": ("ct_armor", "t_armor"),
    "helmet_diff_ct": ("ct_helmets", "t_helmets"),
    "grenade_diff_ct": ("ct_grenades", "t_grenades"),
    "awp_diff_ct": ("ct_awp", "t_awp"),
    "rifle_diff_ct": ("ct_rifles", "t_rifles"),
    "smg_diff_ct": ("ct_smgs", "t_smgs"),
}

BASE_FEATURES = tuple(
    feature for feature in PRE_ROUND_FEATURES if feature not in DIFFERENCE_FEATURES
)

# These limits reject impossible interface inputs; training data is never clipped here.
INTEGER_RANGES = {
    "round_num": (1, 100),
    "ct_score": (0, 99),
    "t_score": (0, 99),
    "ct_eq_value": (0, 50_000),
    "t_eq_value": (0, 50_000),
    "ct_cash": (0, 80_000),
    "t_cash": (0, 80_000),
    "ct_armor": (0, 5),
    "t_armor": (0, 5),
    "ct_helmets": (0, 5),
    "t_helmets": (0, 5),
    "ct_defuse_kits": (0, 5),
    "ct_grenades": (0, 20),
    "t_grenades": (0, 20),
    "ct_ak47": (0, 5),
    "t_ak47": (0, 5),
    "ct_m4a4": (0, 5),
    "t_m4a4": (0, 5),
    "ct_m4a1_s": (0, 5),
    "t_m4a1_s": (0, 5),
    "ct_awp": (0, 5),
    "t_awp": (0, 5),
    "ct_rifles": (0, 5),
    "t_rifles": (0, 5),
    "ct_smgs": (0, 5),
    "t_smgs": (0, 5),
}


class InputValidationError(ValueError):
    """Raised when a pre-round snapshot violates the M13 input contract."""

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


def _check_inventory_consistency(values: Mapping[str, int], errors: list[str]) -> None:
    for side in ("ct", "t"):
        armor = values.get(f"{side}_armor")
        helmets = values.get(f"{side}_helmets")
        if armor is not None and helmets is not None and helmets > armor:
            errors.append(
                f"{side}_helmets ({helmets}) cannot exceed {side}_armor ({armor})"
            )

        rifles = values.get(f"{side}_rifles")
        named_rifles = [
            values.get(f"{side}_ak47"),
            values.get(f"{side}_m4a4"),
            values.get(f"{side}_m4a1_s"),
        ]
        if rifles is not None and all(value is not None for value in named_rifles):
            named_total = sum(named_rifles)
            if named_total > rifles:
                errors.append(
                    f"{side}_rifles ({rifles}) cannot be smaller than the named "
                    f"rifle total ({named_total})"
                )

        primary_counts = [
            values.get(f"{side}_rifles"),
            values.get(f"{side}_awp"),
            values.get(f"{side}_smgs"),
        ]
        if all(value is not None for value in primary_counts):
            primary_total = sum(primary_counts)
            if primary_total > 5:
                errors.append(
                    f"{side} primary weapon total (rifles + awp + smgs) "
                    f"cannot exceed 5; got {primary_total}"
                )


def validate_snapshot(
    snapshot: Mapping[str, Any], known_maps: Iterable[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one purchase-end snapshot and derive CT-minus-T features."""

    if not isinstance(snapshot, Mapping):
        raise InputValidationError(["input must be one JSON object or one CSV row"])

    errors: list[str] = []
    allowed_fields = set(BASE_FEATURES) | set(DIFFERENCE_FEATURES)
    missing = [field for field in BASE_FEATURES if field not in snapshot]
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
        errors.extend(f"{field}: required field is missing" for field in missing)

    unknown = sorted(set(snapshot) - allowed_fields)
    if unknown:
        errors.append("unknown fields: " + ", ".join(unknown))

    normalized_values: dict[str, Any] = {}
    known_map_set = set(known_maps)
    if "map_name" in snapshot:
        map_name = snapshot["map_name"]
        if not isinstance(map_name, str) or not map_name.strip():
            errors.append(f"map_name must be a non-empty string; got {map_name!r}")
        else:
            map_name = map_name.strip()
            normalized_values["map_name"] = map_name
            if map_name not in known_map_set:
                errors.append(
                    f"map_name {map_name!r} was not seen during training; "
                    f"choose one of {sorted(known_map_set)}"
                )

    for field, (minimum, maximum) in INTEGER_RANGES.items():
        if field not in snapshot:
            continue
        value = _as_integer(field, snapshot[field], errors)
        if value is None:
            continue
        normalized_values[field] = value
        if not minimum <= value <= maximum:
            errors.append(
                f"{field} must be between {minimum} and {maximum}; got {value}"
            )

    round_num = normalized_values.get("round_num")
    ct_score = normalized_values.get("ct_score")
    t_score = normalized_values.get("t_score")
    if round_num is not None and ct_score is not None and t_score is not None:
        expected_round = ct_score + t_score + 1
        if round_num != expected_round:
            errors.append(
                "round_num must equal ct_score + t_score + 1; "
                f"expected {expected_round}, got {round_num}"
            )

    _check_inventory_consistency(normalized_values, errors)

    for derived, (ct_field, t_field) in DIFFERENCE_FEATURES.items():
        ct_value = normalized_values.get(ct_field)
        t_value = normalized_values.get(t_field)
        if ct_value is None or t_value is None:
            continue
        expected = ct_value - t_value
        normalized_values[derived] = expected
        if derived in snapshot:
            supplied = _as_integer(derived, snapshot[derived], errors)
            if supplied is not None and supplied != expected:
                errors.append(
                    f"{derived} must equal {ct_field} - {t_field}; "
                    f"expected {expected}, got {supplied}"
                )

    if errors:
        raise InputValidationError(errors)

    normalized = {feature: normalized_values[feature] for feature in PRE_ROUND_FEATURES}
    details = {
        "status": "passed",
        "required_base_feature_count": len(BASE_FEATURES),
        "provided_field_count": len(snapshot),
        "derived_features": list(DIFFERENCE_FEATURES),
        "model_feature_count_before_encoding": len(PRE_ROUND_FEATURES),
        "map_name": normalized["map_name"],
    }
    return normalized, details


def known_maps_from_columns(columns: Iterable[str]) -> set[str]:
    prefix = "map_name_"
    known_maps = {column[len(prefix) :] for column in columns if column.startswith(prefix)}
    if not known_maps:
        raise ValueError("Model bundle does not contain any encoded map_name columns.")
    return known_maps


class PreRoundPredictor:
    """Load the saved M8 model and M10 calibrator for one-row predictions."""

    def __init__(
        self,
        model_bundle: Mapping[str, Any],
        calibrator_bundle: Mapping[str, Any] | None = None,
    ) -> None:
        if "model" not in model_bundle or "columns" not in model_bundle:
            raise ValueError("Model bundle must contain 'model' and 'columns'.")
        self.model = model_bundle["model"]
        self.columns = list(model_bundle["columns"])
        self.known_maps = known_maps_from_columns(self.columns)

        self.calibrator = None
        self.calibration_method = "not_applied"
        if calibrator_bundle is not None:
            if "calibrator" not in calibrator_bundle or "method" not in calibrator_bundle:
                raise ValueError("Calibrator bundle must contain 'calibrator' and 'method'.")
            self.calibrator = calibrator_bundle["calibrator"]
            self.calibration_method = str(calibrator_bundle["method"])

    @classmethod
    def from_paths(
        cls, model_path: str | Path, calibrator_path: str | Path | None = None
    ) -> "PreRoundPredictor":
        model_bundle = joblib.load(Path(model_path))
        calibrator_bundle = (
            joblib.load(Path(calibrator_path)) if calibrator_path is not None else None
        )
        return cls(model_bundle, calibrator_bundle)

    def predict(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        normalized, validation = validate_snapshot(snapshot, self.known_maps)
        frame = pd.DataFrame([normalized], columns=PRE_ROUND_FEATURES)
        encoded = prepare_features(frame).reindex(columns=self.columns, fill_value=0)

        base_probability = float(self.model.predict_proba(encoded)[0, 1])
        ct_probability = base_probability
        if self.calibrator is not None:
            ct_probability = float(
                np.asarray(self.calibrator.predict([base_probability])).reshape(-1)[0]
            )
        if not math.isfinite(ct_probability) or not 0.0 <= ct_probability <= 1.0:
            raise ValueError(
                "Model produced an invalid CT win probability: "
                f"{ct_probability!r}"
            )

        t_probability = 1.0 - ct_probability
        validation["encoded_model_feature_count"] = len(self.columns)
        return {
            "task": "pre_round",
            "snapshot_definition": "purchase end, before combat",
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


def load_snapshot(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        if not isinstance(snapshot, dict):
            raise ValueError("JSON input must contain exactly one object.")
        return snapshot
    if suffix == ".csv":
        frame = pd.read_csv(input_path)
        if len(frame) != 1:
            raise ValueError(f"CSV input must contain exactly one row; got {len(frame)}.")
        return frame.iloc[0].to_dict()
    raise ValueError("Input file must use the .json or .csv extension.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a CS round from one purchase-end, pre-combat snapshot."
    )
    parser.add_argument("--input", required=True, help="One JSON object or one CSV row.")
    parser.add_argument("--model", required=True, help="Saved XGBoost joblib bundle.")
    parser.add_argument("--calibrator", help="Optional saved calibration bundle.")
    parser.add_argument("--output", help="Optional path for the result JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        predictor = PreRoundPredictor.from_paths(args.model, args.calibrator)
        result = predictor.predict(load_snapshot(args.input))
    except (InputValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        error = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "errors": list(exc.errors) if isinstance(exc, InputValidationError) else [str(exc)],
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

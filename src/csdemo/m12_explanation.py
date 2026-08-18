from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from .schema import ID_COLUMNS, PRE_ROUND_FEATURES


FUTURE_INFORMATION_TERMS = (
    "kill",
    "death",
    "damage",
    "bomb_planted",
    "plant_tick",
    "round_end",
    "winner",
)


def deployment_tree_count(bundle: dict) -> int:
    model = bundle.get("model")
    if model is None:
        raise KeyError("Model bundle must contain model.")
    available = len(model.get_booster().get_dump())
    best_iteration = bundle.get("best_iteration")
    tree_count = available if best_iteration is None else int(best_iteration) + 1
    if tree_count < 1 or tree_count > available:
        raise ValueError(
            f"Deployment tree count {tree_count} is outside available range 1..{available}."
        )
    return tree_count


def tree_shap_contributions(
    bundle: dict, x: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    expected_columns = list(bundle.get("columns", []))
    if not expected_columns:
        raise KeyError("Model bundle must contain non-empty columns.")
    if list(x.columns) != expected_columns:
        raise ValueError("SHAP input columns must exactly match the model bundle order.")
    if x.empty:
        raise ValueError("SHAP input must contain at least one row.")

    tree_count = deployment_tree_count(bundle)
    matrix = xgb.DMatrix(x, feature_names=expected_columns)
    contributions = bundle["model"].get_booster().predict(
        matrix,
        pred_contribs=True,
        iteration_range=(0, tree_count),
    )
    if contributions.shape != (len(x), len(expected_columns) + 1):
        raise ValueError(
            "Expected binary TreeSHAP output with one contribution per feature plus bias."
        )
    values = pd.DataFrame(
        contributions[:, :-1], index=x.index, columns=expected_columns
    )
    return values, contributions[:, -1]


def shap_importance(shap_values: pd.DataFrame) -> pd.DataFrame:
    if shap_values.empty:
        raise ValueError("SHAP values must not be empty.")
    importance = pd.DataFrame(
        {
            "feature": shap_values.columns,
            "mean_abs_shap": shap_values.abs().mean().to_numpy(),
            "mean_signed_shap": shap_values.mean().to_numpy(),
            "max_abs_shap": shap_values.abs().max().to_numpy(),
        }
    ).sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    importance.insert(0, "shap_rank", range(1, len(importance) + 1))
    return importance.reset_index(drop=True)


def audit_model_features(feature_names: Sequence[str]) -> pd.DataFrame:
    allowed_numeric = set(PRE_ROUND_FEATURES) - {"map_name"}
    rows = []
    for rank, feature in enumerate(feature_names, start=1):
        source_feature = "map_name" if feature.startswith("map_name_") else feature
        allowed = feature in allowed_numeric or feature.startswith("map_name_")
        if allowed:
            reason = "allowed_pre_round"
        elif feature in ID_COLUMNS or feature == "match_id" or feature.endswith("_id"):
            reason = "identifier"
        elif any(term in feature.lower() for term in FUTURE_INFORMATION_TERMS):
            reason = "future_information"
        else:
            reason = "not_in_pre_round_schema"
        rows.append(
            {
                "importance_rank": rank,
                "feature": feature,
                "source_feature": source_feature,
                "audit_result": "pass" if allowed else "fail",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def select_explanation_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"y_true", "ct_win_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Case predictions are missing columns: {sorted(missing)}")
    working = predictions.copy().reset_index(drop=True)
    working["row_position"] = np.arange(len(working))
    working["predicted_label"] = (
        working["ct_win_probability"].to_numpy() >= 0.5
    ).astype(int)
    working["prediction_confidence"] = np.maximum(
        working["ct_win_probability"], 1.0 - working["ct_win_probability"]
    )

    candidates = {
        "ct_high_probability": working[
            working["y_true"].eq(1) & working["predicted_label"].eq(1)
        ].sort_values("ct_win_probability", ascending=False),
        "t_high_probability": working[
            working["y_true"].eq(0) & working["predicted_label"].eq(0)
        ].sort_values("ct_win_probability", ascending=True),
        "high_confidence_error": working[
            working["y_true"].ne(working["predicted_label"])
        ].sort_values("prediction_confidence", ascending=False),
    }
    empty = [name for name, frame in candidates.items() if frame.empty]
    if empty:
        raise ValueError(f"No eligible rows for explanation cases: {empty}")

    selected = []
    for case_type, frame in candidates.items():
        row = frame.iloc[0].to_dict()
        row["case_type"] = case_type
        selected.append(row)
    return pd.DataFrame(selected)

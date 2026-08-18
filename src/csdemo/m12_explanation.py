from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance

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


def gain_importance(bundle: dict) -> pd.DataFrame:
    expected_columns = list(bundle.get("columns", []))
    if not expected_columns:
        raise KeyError("Model bundle must contain non-empty columns.")
    full_booster = bundle["model"].get_booster()
    available_tree_count = len(full_booster.get_dump())
    tree_count = deployment_tree_count(bundle)
    booster = full_booster[:tree_count] if tree_count < available_tree_count else full_booster
    gain = booster.get_score(importance_type="gain")
    split_count = booster.get_score(importance_type="weight")
    result = pd.DataFrame(
        {
            "feature": expected_columns,
            "gain": [float(gain.get(feature, 0.0)) for feature in expected_columns],
            "split_count": [int(split_count.get(feature, 0)) for feature in expected_columns],
        }
    )
    total_gain = float(result["gain"].sum())
    result["gain_normalized"] = result["gain"] / total_gain if total_gain else 0.0
    result["deployment_tree_count"] = tree_count
    result["available_tree_count"] = available_tree_count
    result = result.sort_values(["gain", "feature"], ascending=[False, True])
    result.insert(0, "gain_rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def permutation_auc_importance(
    model,
    x: pd.DataFrame,
    y,
    *,
    n_repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    if x.empty or len(x) != len(y):
        raise ValueError("Permutation inputs must have the same non-zero length.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    result = permutation_importance(
        model,
        x,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": x.columns,
            "auc_decrease_mean": result.importances_mean,
            "auc_decrease_std": result.importances_std,
            "n_repeats": n_repeats,
        }
    ).sort_values(["auc_decrease_mean", "feature"], ascending=[False, True])
    importance.insert(0, "permutation_rank", range(1, len(importance) + 1))
    return importance.reset_index(drop=True)


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


def build_case_explanations(
    cases: pd.DataFrame,
    x: pd.DataFrame,
    shap_values: pd.DataFrame,
    base_values: np.ndarray,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if len(x) != len(shap_values) or len(x) != len(base_values):
        raise ValueError("Feature, SHAP, and base-value rows must have equal length.")
    if list(x.columns) != list(shap_values.columns):
        raise ValueError("Feature and SHAP columns must match in the same order.")
    if "row_position" not in cases.columns or "case_type" not in cases.columns:
        raise KeyError("Cases must contain row_position and case_type.")

    rows = []
    for _, case in cases.iterrows():
        position = int(case["row_position"])
        if position < 0 or position >= len(x):
            raise IndexError(f"Case row_position {position} is outside the test data.")
        contributions = shap_values.iloc[position]
        ordered_features = contributions.abs().sort_values(ascending=False).index[:top_n]
        model_log_odds = float(base_values[position] + contributions.sum())
        reconstructed_probability = float(
            1.0 / (1.0 + np.exp(-np.clip(model_log_odds, -709, 709)))
        )
        common = {
            column: case[column]
            for column in cases.columns
            if column != "row_position"
        }
        common.update(
            {
                "row_position": position,
                "base_value_log_odds": float(base_values[position]),
                "model_log_odds": model_log_odds,
                "reconstructed_ct_probability": reconstructed_probability,
            }
        )
        for rank, feature in enumerate(ordered_features, start=1):
            contribution = float(contributions[feature])
            rows.append(
                {
                    **common,
                    "contribution_rank": rank,
                    "feature": feature,
                    "feature_value": x.iloc[position][feature],
                    "shap_value_log_odds": contribution,
                    "direction": (
                        "toward_ct"
                        if contribution > 0
                        else "toward_t" if contribution < 0 else "neutral"
                    ),
                }
            )
    return pd.DataFrame(rows)

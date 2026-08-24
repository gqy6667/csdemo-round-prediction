from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .io import read_table
from .m12_explanation import (
    build_case_explanations,
    build_importance_comparison,
    select_explanation_cases,
    shap_importance,
)
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m19_first_kill_explanation import (
    build_source_importance_summary,
    grouped_permutation_auc_importance,
)
from .m22_pre_round_lightgbm_baseline import (
    REPORT_METRICS,
    audit_data_contract,
    prepare_pre_round_splits,
    write_json,
)
from .m24_pre_round_lightgbm_evaluation import run_compile_check
from .metrics import probability_metrics
from .schema import ID_COLUMNS, PRE_ROUND_FEATURE_GROUPS, PRE_ROUND_FEATURES


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLOCKING_CHECKS = (
    "m24_prerequisite",
    "model_replay",
    "model_unchanged",
    "importance_methods",
    "feature_mapping_and_leakage",
    "shap_reconstruction",
    "source_and_macro_groups",
    "xgboost_explanation_comparison",
    "case_explanations",
    "external_report",
    "automated_tests",
    "source_compile",
    "reproduction_entrypoint",
    "artifact_manifest",
)

SOURCE_TO_MACRO_GROUP = {
    feature: group
    for group, features in PRE_ROUND_FEATURE_GROUPS.items()
    for feature in features
}


def map_encoded_feature_to_source(
    encoded_feature: str,
    raw_features: Sequence[str],
) -> str:
    raw = list(raw_features)
    if encoded_feature in raw:
        return encoded_feature
    if "map_name" in raw and encoded_feature.startswith("map_name_"):
        return "map_name"
    raise ValueError(
        f"Encoded feature {encoded_feature!r} does not map to one raw feature"
    )


def _failed_feature_reason(feature: str) -> str:
    lower = feature.lower()
    if feature in ID_COLUMNS or feature == "match_id" or feature.endswith("_id"):
        return "identifier"
    if feature in {
        "ct_win",
        "y_true",
        "label",
        "target",
        "split",
        "predicted_label",
    }:
        return "label_or_split"
    if any(token in lower for token in ("team", "player", "steam", "roster")):
        return "identity_feature"
    if any(
        token in lower
        for token in (
            "kill",
            "death",
            "damage",
            "health",
            "alive",
            "bomb",
            "plant",
            "defuse_event",
            "round_end",
            "winner",
        )
    ):
        return "future_information"
    return "not_in_purchase_end_contract"


def audit_pre_round_features(
    encoded_features: Sequence[str],
    raw_features: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for position, encoded in enumerate(encoded_features, start=1):
        try:
            source = map_encoded_feature_to_source(encoded, raw_features)
            macro_group = SOURCE_TO_MACRO_GROUP.get(source)
            allowed = source in PRE_ROUND_FEATURES and macro_group is not None
            reason = (
                "accepted_m14_purchase_end_feature"
                if allowed
                else "not_in_purchase_end_contract"
            )
            availability = (
                "purchase_end_pre_combat" if allowed else "forbidden_or_unknown"
            )
        except ValueError:
            source = None
            macro_group = None
            allowed = False
            reason = _failed_feature_reason(encoded)
            availability = "forbidden_or_unknown"
        rows.append(
            {
                "encoded_position": position,
                "encoded_feature": encoded,
                "source_feature": source,
                "feature_group": macro_group if macro_group else "forbidden",
                "availability": availability,
                "audit_result": "pass" if allowed else "fail",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_source_feature_groups(
    encoded_features: Sequence[str],
    raw_features: Sequence[str],
) -> dict[str, list[str]]:
    groups = {feature: [] for feature in raw_features}
    for encoded in encoded_features:
        source = map_encoded_feature_to_source(encoded, raw_features)
        groups[source].append(encoded)
    missing = [feature for feature, columns in groups.items() if not columns]
    if missing:
        raise ValueError(f"Raw features have no encoded columns: {missing}")
    flattened = [column for columns in groups.values() for column in columns]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(
        encoded_features
    ):
        raise ValueError("Encoded columns must map to exactly one source feature")
    return groups


def build_macro_feature_groups(
    encoded_features: Sequence[str],
    raw_features: Sequence[str],
) -> dict[str, list[str]]:
    source_groups = build_source_feature_groups(encoded_features, raw_features)
    unknown = sorted(set(raw_features) - set(SOURCE_TO_MACRO_GROUP))
    if unknown:
        raise ValueError(f"Raw features are outside the M14 feature groups: {unknown}")
    groups: dict[str, list[str]] = {}
    for macro_group, contract_features in PRE_ROUND_FEATURE_GROUPS.items():
        columns = [
            encoded
            for source in contract_features
            if source in source_groups
            for encoded in source_groups[source]
        ]
        if columns:
            groups[macro_group] = columns
    return groups


def encoded_permutation_auc_importance(
    model: Any,
    x: pd.DataFrame,
    y: Sequence[int],
    *,
    n_repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    if x.empty or len(x) != len(y):
        raise ValueError("Permutation inputs must have equal non-zero length")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    labels = np.asarray(y, dtype=int)
    baseline_probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    baseline_auc = float(roc_auc_score(labels, baseline_probability))
    rng = np.random.default_rng(seed)
    rows = []
    for feature in x.columns:
        decreases = []
        for _ in range(n_repeats):
            order = rng.permutation(len(x))
            permuted = x.copy()
            permuted.loc[:, feature] = x.iloc[order][feature].to_numpy()
            probability = np.asarray(
                model.predict_proba(permuted)[:, 1], dtype=float
            )
            decreases.append(baseline_auc - roc_auc_score(labels, probability))
        values = np.asarray(decreases, dtype=float)
        rows.append(
            {
                "feature": feature,
                "baseline_auc": baseline_auc,
                "auc_decrease_mean": float(values.mean()),
                "auc_decrease_std": float(values.std(ddof=0)),
                "auc_decrease_min": float(values.min()),
                "auc_decrease_max": float(values.max()),
                "n_repeats": n_repeats,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["auc_decrease_mean", "feature"], ascending=[False, True]
    )
    result.insert(0, "permutation_rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def audit_frozen_prediction_replay(
    saved: pd.DataFrame,
    replayed: pd.DataFrame,
    expected_metrics: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    required = set(ID_COLUMNS) | {"y_true", "ct_win_probability"}
    for name, table in (("saved", saved), ("replayed", replayed)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise KeyError(f"M25 {name} predictions are missing columns: {missing}")
    saved_duplicates = int(saved.duplicated(ID_COLUMNS).sum())
    replayed_duplicates = int(replayed.duplicated(ID_COLUMNS).sum())
    saved_keys = pd.MultiIndex.from_frame(saved[ID_COLUMNS].astype("string"))
    replayed_keys = pd.MultiIndex.from_frame(replayed[ID_COLUMNS].astype("string"))
    key_mismatches = int(
        (~saved_keys.isin(replayed_keys)).sum()
        + (~replayed_keys.isin(saved_keys)).sum()
    )
    label_mismatches = 0
    invalid_probability_cells = 0
    max_probability_difference: float | None = None
    metric_difference: float | None = None
    current_metrics: dict[str, float] = {}
    if not saved_duplicates and not replayed_duplicates and not key_mismatches:
        joined = saved[ID_COLUMNS + ["y_true", "ct_win_probability"]].merge(
            replayed[ID_COLUMNS + ["y_true", "ct_win_probability"]],
            on=ID_COLUMNS,
            how="inner",
            validate="one_to_one",
            suffixes=("_saved", "_replayed"),
        )
        label_mismatches = int(
            joined["y_true_saved"].astype(int).ne(
                joined["y_true_replayed"].astype(int)
            ).sum()
        )
        probabilities = joined[
            ["ct_win_probability_saved", "ct_win_probability_replayed"]
        ].to_numpy(dtype=float)
        invalid_probability_cells = int(
            (~np.isfinite(probabilities)).sum()
            + ((probabilities < 0) | (probabilities > 1)).sum()
        )
        if not invalid_probability_cells and not label_mismatches:
            max_probability_difference = float(
                np.max(
                    np.abs(probabilities[:, 0] - probabilities[:, 1]),
                    initial=0.0,
                )
            )
            current_metrics = probability_metrics(
                joined["y_true_replayed"].to_numpy(dtype=int),
                probabilities[:, 1],
                n_bins=10,
            )
            metric_names = tuple(expected_metrics)
            missing_metrics = sorted(set(metric_names) - set(current_metrics))
            if missing_metrics:
                raise KeyError(
                    f"Expected replay metrics are unknown: {missing_metrics}"
                )
            metric_difference = max(
                abs(float(current_metrics[name]) - float(expected_metrics[name]))
                for name in metric_names
            )
    passed = bool(
        not saved_duplicates
        and not replayed_duplicates
        and not key_mismatches
        and not label_mismatches
        and not invalid_probability_cells
        and max_probability_difference is not None
        and max_probability_difference <= tolerance
        and metric_difference is not None
        and metric_difference <= tolerance
    )
    return {
        "passed": passed,
        "tolerance": tolerance,
        "saved_rows": int(len(saved)),
        "replayed_rows": int(len(replayed)),
        "saved_duplicate_key_rows": saved_duplicates,
        "replayed_duplicate_key_rows": replayed_duplicates,
        "key_mismatch_count": key_mismatches,
        "label_mismatch_count": label_mismatches,
        "invalid_probability_cells": invalid_probability_cells,
        "max_absolute_probability_difference": max_probability_difference,
        "metric_max_absolute_difference": metric_difference,
        "metrics": current_metrics,
        "lightgbm_fit_calls": 0,
    }


def lightgbm_deployment_tree_count(bundle: Mapping[str, Any]) -> int:
    model = bundle.get("model")
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise KeyError("LightGBM model bundle must contain a fitted booster")
    available = int(booster.num_trees())
    best_iteration = bundle.get("best_iteration")
    tree_count = available if best_iteration is None else int(best_iteration)
    if tree_count < 1 or tree_count > available:
        raise ValueError(
            f"Deployment tree count {tree_count} is outside available range 1..{available}"
        )
    return tree_count


def lightgbm_gain_importance(bundle: Mapping[str, Any]) -> pd.DataFrame:
    expected_columns = list(bundle.get("columns", []))
    if not expected_columns:
        raise KeyError("Model bundle must contain non-empty columns")
    booster = bundle["model"].booster_
    booster_columns = list(booster.feature_name())
    if booster_columns != expected_columns:
        raise ValueError("LightGBM booster feature names differ from bundle columns")
    tree_count = lightgbm_deployment_tree_count(bundle)
    available = int(booster.num_trees())
    gain = np.asarray(
        booster.feature_importance(importance_type="gain", iteration=tree_count),
        dtype=float,
    )
    split_count = np.asarray(
        booster.feature_importance(importance_type="split", iteration=tree_count),
        dtype=int,
    )
    if len(gain) != len(expected_columns) or len(split_count) != len(expected_columns):
        raise ValueError("LightGBM importance length differs from encoded columns")
    result = pd.DataFrame(
        {
            "feature": expected_columns,
            "gain": gain,
            "split_count": split_count,
        }
    )
    total_gain = float(result["gain"].sum())
    result["gain_normalized"] = (
        result["gain"] / total_gain if total_gain > 0 else 0.0
    )
    result["deployment_tree_count"] = tree_count
    result["available_tree_count"] = available
    result = result.sort_values(["gain", "feature"], ascending=[False, True])
    result.insert(0, "gain_rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def lightgbm_tree_shap_contributions(
    bundle: Mapping[str, Any],
    x: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    expected_columns = list(bundle.get("columns", []))
    if not expected_columns:
        raise KeyError("Model bundle must contain non-empty columns")
    if x.columns.tolist() != expected_columns:
        raise ValueError("SHAP input columns must exactly match the model bundle order")
    if x.empty:
        raise ValueError("SHAP input must contain at least one row")
    tree_count = lightgbm_deployment_tree_count(bundle)
    contributions = np.asarray(
        bundle["model"].booster_.predict(
            x,
            pred_contrib=True,
            num_iteration=tree_count,
        ),
        dtype=float,
    )
    expected_shape = (len(x), len(expected_columns) + 1)
    if contributions.shape != expected_shape:
        raise ValueError(
            "Expected binary TreeSHAP output with one contribution per feature plus bias"
        )
    if not np.isfinite(contributions).all():
        raise ValueError("TreeSHAP contributions must all be finite")
    values = pd.DataFrame(
        contributions[:, :-1],
        index=x.index,
        columns=expected_columns,
    )
    return values, contributions[:, -1]


def build_model_importance_comparison(
    lightgbm: pd.DataFrame,
    xgboost: pd.DataFrame,
    *,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    rank_columns = ("gain_rank", "permutation_rank", "shap_rank", "mean_rank")
    required = {"feature", *rank_columns}
    for name, table in (("LightGBM", lightgbm), ("XGBoost", xgboost)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise KeyError(f"{name} importance table is missing columns: {missing}")
        if table["feature"].duplicated().any():
            raise ValueError(f"{name} importance table contains duplicate features")
    lightgbm_features = set(lightgbm["feature"])
    xgboost_features = set(xgboost["feature"])
    if lightgbm_features != xgboost_features:
        raise ValueError(
            "LightGBM and XGBoost feature sets differ: "
            f"only_lightgbm={sorted(lightgbm_features - xgboost_features)}, "
            f"only_xgboost={sorted(xgboost_features - lightgbm_features)}"
        )

    left = lightgbm[["feature", *rank_columns]].rename(
        columns={column: f"lightgbm_{column}" for column in rank_columns}
    )
    right = xgboost[["feature", *rank_columns]].rename(
        columns={column: f"xgboost_{column}" for column in rank_columns}
    )
    detail = left.merge(right, on="feature", validate="one_to_one")
    method_columns = {
        "gain": "gain_rank",
        "permutation_auc": "permutation_rank",
        "tree_shap": "shap_rank",
        "mean_rank": "mean_rank",
    }
    rows = []
    effective_top_n = min(top_n, len(detail))
    for method, rank_column in method_columns.items():
        lightgbm_column = f"lightgbm_{rank_column}"
        xgboost_column = f"xgboost_{rank_column}"
        difference_column = (
            f"{rank_column}_difference_lgbm_minus_xgb"
        )
        detail[difference_column] = (
            detail[lightgbm_column] - detail[xgboost_column]
        )
        lightgbm_top = set(
            detail.nsmallest(effective_top_n, lightgbm_column)["feature"]
        )
        xgboost_top = set(
            detail.nsmallest(effective_top_n, xgboost_column)["feature"]
        )
        intersection = lightgbm_top & xgboost_top
        union = lightgbm_top | xgboost_top
        rows.append(
            {
                "method": method,
                "feature_count": int(len(detail)),
                "spearman_rank": float(
                    detail[lightgbm_column].corr(
                        detail[xgboost_column], method="spearman"
                    )
                ),
                "top_n": effective_top_n,
                "top_overlap_count": len(intersection),
                "top_union_count": len(union),
                "top_jaccard": len(intersection) / len(union),
            }
        )
    detail = detail.sort_values(
        ["lightgbm_mean_rank", "feature"], ascending=[True, True]
    ).reset_index(drop=True)
    return detail, pd.DataFrame(rows)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sigmoid(log_odds: Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(log_odds, dtype=float), -709, 709)
    return 1.0 / (1.0 + np.exp(-clipped))


def verify_m24_prerequisite(
    data_path: str | Path,
    model_path: str | Path,
    m24_summary: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    data_artifact = fingerprint_file(data_path)
    model_artifact = fingerprint_file(model_path)
    expected_data_sha = m24_summary.get("data", {}).get("sha256")
    expected_model_sha = (
        m24_summary.get("prerequisite", {})
        .get("model_artifact", {})
        .get("sha256")
    )
    raw_features = list(bundle.get("raw_features", []))
    encoded_columns = list(bundle.get("columns", []))
    model_replay = m24_summary.get("model_replay", {})
    expected_raw_count = int(model_replay.get("raw_feature_count", -1))
    expected_encoded_count = int(model_replay.get("encoded_feature_count", -1))
    deployment_trees = lightgbm_deployment_tree_count(bundle)
    available_trees = int(bundle["model"].booster_.num_trees())
    checks = {
        "m24_accepted": bool(
            m24_summary.get("acceptance", {}).get("status") == "passed"
            and m24_summary.get("acceptance", {}).get("ready_for_m25") is True
        ),
        "m24_task": m24_summary.get("task") == "pre_round",
        "data_sha256": bool(expected_data_sha)
        and data_artifact["sha256"] == expected_data_sha
        and bundle.get("data_sha256") == expected_data_sha,
        "model_sha256": bool(expected_model_sha)
        and model_artifact["sha256"] == expected_model_sha,
        "bundle_task": bundle.get("task") == "pre_round",
        "bundle_model_name": bundle.get("model_name") == "lightgbm_tuned",
        "raw_feature_contract": raw_features == list(PRE_ROUND_FEATURES)
        and len(raw_features) == expected_raw_count,
        "encoded_feature_contract": bool(encoded_columns)
        and len(encoded_columns) == len(set(encoded_columns))
        and len(encoded_columns) == expected_encoded_count,
        "deployment_tree_contract": deployment_trees == available_trees
        and deployment_trees == int(bundle.get("best_iteration", -1)),
        "predictor_available": hasattr(bundle.get("model"), "predict_proba"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "data_artifact": data_artifact,
        "model_artifact": model_artifact,
        "raw_feature_count": len(raw_features),
        "encoded_feature_count": len(encoded_columns),
        "deployment_tree_count": deployment_trees,
        "available_tree_count": available_trees,
    }


def prepare_explanation_inputs(
    data: pd.DataFrame,
    bundle: Mapping[str, Any],
    m24_summary: Mapping[str, Any],
    m24_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    data_audit = audit_data_contract(data)
    expected_split_rows = {
        name: int(count)
        for name, count in m24_summary.get("data", {})
        .get("split_rows", {})
        .items()
    }
    expected_split_series = {
        name: int(count)
        for name, count in m24_summary.get("data", {})
        .get("split_series", {})
        .items()
    }
    if (
        not data_audit["passed"]
        or data_audit["split_rows"] != expected_split_rows
        or data_audit["split_series"] != expected_split_series
    ):
        raise RuntimeError("M25 data identity or split contract differs from M24")

    prepared = prepare_pre_round_splits(data)
    x_test, y_test, identity = prepared["test"]
    encoded_columns = list(bundle.get("columns", []))
    if x_test.columns.tolist() != encoded_columns:
        raise RuntimeError("M25 encoded test columns differ from the frozen model")
    probability = np.asarray(
        bundle["model"].predict_proba(x_test)[:, 1], dtype=float
    )
    if (
        len(probability) != len(x_test)
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise RuntimeError("M25 frozen LightGBM produced invalid probabilities")

    predictions = identity[ID_COLUMNS].copy()
    predictions["y_true"] = y_test.to_numpy(dtype=int)
    predictions["ct_win_probability"] = probability
    predictions["t_win_probability"] = 1.0 - probability
    predictions["predicted_label"] = (probability >= 0.5).astype(int)
    predictions["correct"] = predictions["predicted_label"].eq(
        predictions["y_true"]
    )
    metadata = data.loc[
        data["split"].eq("test"),
        ID_COLUMNS + ["map_name", "round_num"],
    ].copy()
    if metadata.duplicated(ID_COLUMNS).any():
        raise RuntimeError("M25 test metadata contains duplicate complete keys")
    predictions = predictions.merge(
        metadata,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if predictions[["map_name", "round_num"]].isna().any().any():
        raise RuntimeError("M25 could not attach test metadata by complete key")

    saved_columns = ID_COLUMNS + ["y_true", "ct_win_probability"]
    missing_saved = sorted(set(saved_columns) - set(m24_predictions.columns))
    if missing_saved:
        raise KeyError(f"M24 predictions are missing columns: {missing_saved}")
    replay = audit_frozen_prediction_replay(
        m24_predictions[saved_columns],
        predictions[saved_columns],
        m24_summary["metrics"],
        tolerance=1e-12,
    )
    replay.update(
        {
            "data_contract_passed": data_audit["passed"],
            "split_rows": data_audit["split_rows"],
            "split_series": data_audit["split_series"],
            "duplicate_key_rows": data_audit["duplicate_key_rows"],
            "cross_split_series": data_audit["cross_split_series"],
            "cross_split_games": data_audit["cross_split_games"],
            "cross_split_rounds": data_audit["cross_split_rounds"],
        }
    )
    return x_test, y_test, predictions, replay


def run_explanation_core(
    data_path: str | Path,
    model_path: str | Path,
    m24_summary_path: str | Path,
    m24_predictions_path: str | Path,
    *,
    permutation_repeats: int = 20,
    seed: int = 42,
    case_features: int = 10,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    m24_summary = _read_json(m24_summary_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("M25 expected the M23 model artifact to contain a bundle")
    prerequisite = verify_m24_prerequisite(
        data_path,
        model_path,
        m24_summary,
        bundle,
    )
    if not prerequisite["passed"]:
        raise RuntimeError("M25 input does not match accepted M24 artifacts")

    data = read_table(data_path)
    m24_predictions = read_table(m24_predictions_path)
    x_test, y_test, predictions, model_replay = prepare_explanation_inputs(
        data,
        bundle,
        m24_summary,
        m24_predictions,
    )
    if not model_replay["passed"]:
        raise RuntimeError("M25 could not exactly replay the frozen M24 predictions")

    gain = lightgbm_gain_importance(bundle)
    encoded_permutation = encoded_permutation_auc_importance(
        bundle["model"],
        x_test,
        y_test,
        n_repeats=permutation_repeats,
        seed=seed,
    )
    shap_values, base_values = lightgbm_tree_shap_contributions(bundle, x_test)
    shap = shap_importance(shap_values)
    encoded_comparison = build_importance_comparison(
        gain,
        encoded_permutation,
        shap,
    )

    raw_features = list(bundle["raw_features"])
    encoded_columns = list(bundle["columns"])
    encoded_contract = audit_pre_round_features(encoded_columns, raw_features)
    leakage_audit = audit_pre_round_features(
        shap["feature"].tolist(), raw_features
    ).merge(
        shap[["feature", "shap_rank", "mean_abs_shap"]],
        left_on="encoded_feature",
        right_on="feature",
        how="left",
        validate="one_to_one",
    ).drop(columns="feature")
    top20_audit = leakage_audit.head(20).copy()

    source_groups = build_source_feature_groups(encoded_columns, raw_features)
    grouped_permutation = grouped_permutation_auc_importance(
        bundle["model"],
        x_test,
        y_test,
        source_groups,
        n_repeats=permutation_repeats,
        seed=seed,
    )
    macro_groups = build_macro_feature_groups(encoded_columns, raw_features)
    macro_permutation = grouped_permutation_auc_importance(
        bundle["model"],
        x_test,
        y_test,
        macro_groups,
        n_repeats=permutation_repeats,
        seed=seed + 1,
    )
    source_importance = build_source_importance_summary(
        gain,
        shap,
        grouped_permutation,
        encoded_contract,
    )

    probability = predictions["ct_win_probability"].to_numpy(dtype=float)
    reconstructed = _sigmoid(
        base_values + shap_values.sum(axis=1).to_numpy(dtype=float)
    )
    reconstruction_error = np.abs(reconstructed - probability)
    cases = select_explanation_cases(predictions)
    case_explanations = build_case_explanations(
        cases,
        x_test,
        shap_values,
        base_values,
        top_n=case_features,
    )
    rank_correlations = encoded_comparison[
        ["gain_rank", "permutation_rank", "shap_rank"]
    ].corr(method="spearman")
    summary = {
        "prerequisite": prerequisite,
        "model_replay": model_replay,
        "test_rounds": int(len(x_test)),
        "raw_features": len(raw_features),
        "encoded_features": len(encoded_columns),
        "available_tree_count": prerequisite["available_tree_count"],
        "deployment_tree_count": prerequisite["deployment_tree_count"],
        "permutation_repeats": permutation_repeats,
        "shap_reconstruction_max_abs_error": float(reconstruction_error.max()),
        "shap_reconstruction_mean_abs_error": float(reconstruction_error.mean()),
        "feature_audit": {
            "all_feature_failures": int(
                leakage_audit["audit_result"].eq("fail").sum()
            ),
            "top20_failures": int(
                top20_audit["audit_result"].eq("fail").sum()
            ),
            "mapped_source_features": int(
                encoded_contract["source_feature"].nunique()
            ),
            "mapped_macro_groups": int(
                encoded_contract["feature_group"].nunique()
            ),
        },
        "top_features": {
            "encoded_gain": gain.head(10)["feature"].tolist(),
            "encoded_permutation": encoded_permutation.head(10)["feature"].tolist(),
            "encoded_shap": shap.head(10)["feature"].tolist(),
            "source_mean_rank": source_importance.head(10)[
                "source_feature"
            ].tolist(),
            "source_grouped_permutation": grouped_permutation.head(10)[
                "feature_group"
            ].tolist(),
        },
        "importance_rank_spearman": rank_correlations.to_dict(),
        "selected_cases": cases.to_dict(orient="records"),
    }
    tables = {
        "gain_importance": gain,
        "permutation_importance_auc": encoded_permutation,
        "shap_importance": shap,
        "importance_comparison": encoded_comparison,
        "encoded_feature_contract": encoded_contract,
        "all_feature_leakage_audit": leakage_audit,
        "top20_feature_audit": top20_audit,
        "grouped_permutation_importance_auc": grouped_permutation,
        "macro_group_permutation_auc": macro_permutation,
        "source_feature_importance": source_importance,
        "selected_cases": cases,
        "case_explanations": case_explanations,
        "test_predictions": predictions,
        "x_test": x_test,
        "shap_values": shap_values,
    }
    return summary, tables


def load_m12_xgboost_importance(
    m12_report_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    report_dir = Path(m12_report_dir)
    tables = {
        "gain": read_table(report_dir / "gain_importance.csv"),
        "permutation": read_table(
            report_dir / "permutation_importance_auc.csv"
        ),
        "shap": read_table(report_dir / "shap_importance.csv"),
        "saved_comparison": read_table(
            report_dir / "importance_comparison.csv"
        ),
    }
    rebuilt = build_importance_comparison(
        tables["gain"],
        tables["permutation"],
        tables["shap"],
    )
    saved = tables["saved_comparison"].sort_values("feature").reset_index(drop=True)
    rebuilt_sorted = rebuilt.sort_values("feature").reset_index(drop=True)
    exact_columns = [
        "feature",
        "gain_rank",
        "permutation_rank",
        "shap_rank",
    ]
    exact_match = saved[exact_columns].to_dict(orient="records") == rebuilt_sorted[
        exact_columns
    ].to_dict(orient="records")
    mean_rank_match = np.allclose(
        saved["mean_rank"].to_numpy(dtype=float),
        rebuilt_sorted["mean_rank"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )
    if not exact_match or not mean_rank_match:
        raise RuntimeError("M12 saved explanation ranks do not match source tables")
    return rebuilt, tables


def validate_external_comparison(
    external: pd.DataFrame,
    m24_metrics: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    required = {
        "benchmark_id",
        "source_title",
        "metric",
        "reported_value",
        "comparability",
        "current_value",
        "performance_advantage_ours",
    }
    missing = sorted(required - set(external.columns))
    if missing:
        raise KeyError(f"M24 external comparison is missing columns: {missing}")
    duplicate_ids = int(external["benchmark_id"].duplicated().sum())
    unknown_metrics = sorted(set(external["metric"]) - set(m24_metrics))
    current_differences = []
    if not unknown_metrics:
        current_differences = [
            abs(float(row["current_value"]) - float(m24_metrics[row["metric"]]))
            for _, row in external.iterrows()
        ]
    max_difference = max(current_differences, default=float("inf"))
    invalid_numeric = int(
        (~np.isfinite(external["reported_value"].to_numpy(dtype=float))).sum()
        + (~np.isfinite(external["current_value"].to_numpy(dtype=float))).sum()
        + (
            ~np.isfinite(
                external["performance_advantage_ours"].to_numpy(dtype=float)
            )
        ).sum()
    )
    passed = bool(
        len(external) > 0
        and duplicate_ids == 0
        and not unknown_metrics
        and invalid_numeric == 0
        and max_difference <= tolerance
    )
    return {
        "passed": passed,
        "rows": int(len(external)),
        "duplicate_benchmark_ids": duplicate_ids,
        "unknown_metrics": unknown_metrics,
        "invalid_numeric_cells": invalid_numeric,
        "current_metric_max_absolute_difference_vs_m24": max_difference,
        "tolerance": tolerance,
    }


def render_external_report(external: pd.DataFrame) -> str:
    lines = [
        "# M25 外部模型指标差距",
        "",
        "M25 不改变 M24 概率，因此本表逐行复用 M24 的外部比较。差值统一为本项目减",
        "外部报告；不同数据、预测时点和 split 不能用于证明算法排名。",
        "",
        "| 来源 | 指标 | 本项目 | 外部报告 | 性能优势 | 可比性 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, row in external.iterrows():
        title = str(row["source_title"])
        url = str(row.get("source_url", ""))
        source = f"[{title}]({url})" if url else title
        lines.append(
            f"| {source} | {row['metric']} | {float(row['current_value']):.6f} | "
            f"{float(row['reported_value']):.6f} | "
            f"{float(row['performance_advantage_ours']):+.6f} | "
            f"{row['comparability']} |"
        )
    lines.extend(
        [
            "",
            "`closest_task` 只表示预测时点最接近；`not_comparable` 表示输入包含回合内",
            "信息或任务明显更容易。性能优势为正表示按该指标方向本项目更好。",
            "",
        ]
    )
    return "\n".join(lines)


def _save_importance_bar(
    table: pd.DataFrame,
    *,
    label_column: str,
    value_column: str,
    title: str,
    x_label: str,
    path: Path,
    color: str,
    error_column: str | None = None,
    top_n: int = 20,
) -> None:
    part = table.head(top_n).sort_values(value_column, ascending=True)
    errors = part[error_column] if error_column else None
    figure, axis = plt.subplots(figsize=(8.6, 6.4))
    axis.barh(
        part[label_column],
        part[value_column],
        xerr=errors,
        color=color,
        alpha=0.92,
        capsize=2,
    )
    axis.axvline(0, color="#4B5563", linewidth=1)
    axis.set(title=title, xlabel=x_label, ylabel="Feature")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_m25_plots(
    tables: Mapping[str, pd.DataFrame],
    report_dir: str | Path,
    *,
    seed: int,
    shap_plot_rows: int = 1500,
) -> None:
    output_dir = Path(report_dir)
    plt.style.use("seaborn-v0_8-whitegrid")
    _save_importance_bar(
        tables["gain_importance"],
        label_column="feature",
        value_column="gain_normalized",
        title="M25 LightGBM gain importance",
        x_label="Normalized total gain",
        path=output_dir / "gain_importance.png",
        color="#176B87",
    )
    _save_importance_bar(
        tables["permutation_importance_auc"],
        label_column="feature",
        value_column="auc_decrease_mean",
        error_column="auc_decrease_std",
        title="M25 encoded-feature permutation importance",
        x_label="Mean decrease in fixed-test AUC",
        path=output_dir / "permutation_importance_auc.png",
        color="#D17A22",
    )
    _save_importance_bar(
        tables["shap_importance"],
        label_column="feature",
        value_column="mean_abs_shap",
        title="M25 LightGBM TreeSHAP importance",
        x_label="Mean absolute contribution (log-odds)",
        path=output_dir / "shap_importance.png",
        color="#4C956C",
    )
    _save_importance_bar(
        tables["source_feature_importance"].sort_values(
            "grouped_auc_decrease_mean", ascending=False
        ),
        label_column="source_feature",
        value_column="grouped_auc_decrease_mean",
        error_column="grouped_auc_decrease_std",
        title="M25 grouped permutation by raw feature",
        x_label="Mean decrease in fixed-test AUC",
        path=output_dir / "source_feature_grouped_permutation.png",
        color="#2A9D8F",
    )
    _save_importance_bar(
        tables["macro_group_permutation_auc"],
        label_column="feature_group",
        value_column="auc_decrease_mean",
        error_column="auc_decrease_std",
        title="M25 permutation importance by M14 feature group",
        x_label="Mean decrease in fixed-test AUC",
        path=output_dir / "macro_group_permutation_auc.png",
        color="#C44E52",
        top_n=5,
    )

    x_test = tables["x_test"]
    shap_values = tables["shap_values"]
    top_features = tables["shap_importance"].head(15)["feature"].tolist()[::-1]
    rng = np.random.default_rng(seed)
    sample_size = min(shap_plot_rows, len(x_test))
    positions = np.sort(rng.choice(len(x_test), size=sample_size, replace=False))
    figure, axis = plt.subplots(figsize=(9.0, 6.8))
    last_scatter = None
    for y_position, feature in enumerate(top_features):
        feature_values = pd.to_numeric(
            x_test.iloc[positions][feature], errors="coerce"
        ).fillna(0.0)
        relative_values = feature_values.rank(method="average", pct=True).to_numpy()
        jitter = rng.uniform(-0.24, 0.24, size=sample_size)
        last_scatter = axis.scatter(
            shap_values.iloc[positions][feature],
            y_position + jitter,
            c=relative_values,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=9,
            alpha=0.62,
            linewidths=0,
        )
    axis.axvline(0, color="#4B5563", linewidth=1)
    axis.set_yticks(range(len(top_features)), labels=top_features)
    axis.set(
        xlabel="SHAP contribution to CT win log-odds",
        ylabel="Feature",
        title=f"M25 TreeSHAP summary ({sample_size:,} fixed-test rows)",
    )
    if last_scatter is not None:
        colorbar = figure.colorbar(last_scatter, ax=axis, pad=0.02)
        colorbar.set_label("Relative feature value (low to high)")
    figure.tight_layout()
    figure.savefig(output_dir / "shap_summary.png", dpi=170)
    plt.close(figure)

    cases = tables["selected_cases"]
    explanations = tables["case_explanations"]
    case_order = (
        "ct_high_probability",
        "t_high_probability",
        "high_confidence_error",
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.8))
    for axis, case_type in zip(axes, case_order):
        part = explanations.loc[
            explanations["case_type"].eq(case_type)
        ].sort_values("contribution_rank", ascending=False)
        colors = np.where(
            part["shap_value_log_odds"].gt(0), "#176B87", "#C44E52"
        )
        axis.barh(part["feature"], part["shap_value_log_odds"], color=colors)
        axis.axvline(0, color="#4B5563", linewidth=1)
        probability = float(
            cases.loc[
                cases["case_type"].eq(case_type), "ct_win_probability"
            ].iloc[0]
        )
        axis.set(
            title=f"{case_type}\nCT probability = {probability:.3f}",
            xlabel="SHAP value (log-odds)",
        )
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "case_explanations.png", dpi=170)
    plt.close(figure)

    agreement = tables["model_importance_agreement"].sort_values(
        "spearman_rank", ascending=True
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.barh(agreement["method"], agreement["spearman_rank"], color="#6C5B7B")
    axis.axvline(0, color="#111827", linewidth=1)
    axis.set_xlim(-1, 1)
    axis.set(
        title="M25 LightGBM vs XGBoost importance rank agreement",
        xlabel="Spearman rank correlation",
        ylabel="Explanation method",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "model_importance_agreement.png", dpi=170)
    plt.close(figure)


def render_m25_report(
    summary: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> str:
    metrics = summary["metrics"]
    source = tables["source_feature_importance"]
    macro = tables["macro_group_permutation_auc"]
    agreement = tables["model_importance_agreement"]
    audit = tables["top20_feature_audit"]
    cases = tables["selected_cases"]
    external = tables["external_benchmark_comparison"]
    acceptance = summary["acceptance"]
    lines = [
        "# M25 开局前 LightGBM 模型解释与泄漏审计",
        "",
        "## 结论",
        "",
        f"M25 阻断检查 {acceptance['blocking_passed']}/{acceptance['blocking_total']} "
        f"通过，状态为 `{acceptance['status']}`。本阶段没有训练、调参、删特征或修改",
        "阈值；解释对象仍是 M23/M24 冻结的购买结束、交火前 LightGBM。",
        "",
        f"测试集仍为 {summary['test_rounds']:,} 回合。Accuracy "
        f"{metrics['accuracy']:.6f}、AUC {metrics['auc']:.6f}、Log Loss "
        f"{metrics['log_loss']:.6f}、Brier {metrics['brier_score']:.6f}、ECE10 "
        f"{metrics['ece10']:.6f}，与 M24 最大差 "
        f"{summary['model_replay']['metric_max_absolute_difference']:.3e}。",
        "",
        "## 解释完整性",
        "",
        f"冻结模型包含并部署 {summary['deployment_tree_count']} 棵树。LightGBM 原生",
        f"TreeSHAP 重建概率的最大绝对误差为 "
        f"{summary['shap_reconstruction_max_abs_error']:.3e}；模型文件运行前后 SHA-256 "
        f"一致为 `{summary['model_integrity']['sha256_before']}`。",
        "",
        f"43 个编码列全部映射到 36 个 M14 购买结束特征和五个宏观组。完整审计失败 "
        f"{summary['feature_audit']['all_feature_failures']}，SHAP 前 20 失败 "
        f"{summary['feature_audit']['top20_failures']}。首杀、伤害、血量、下包、标签、",
        "ID、战队和选手身份均未进入模型。",
        "",
        "## 原始特征重要性",
        "",
        "| 特征 | 组 | Gain 排名 | 分组置换排名 | SHAP 排名 | 测试 AUC 平均下降 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in source.head(15).iterrows():
        lines.append(
            f"| {row['source_feature']} | {row['feature_group']} | "
            f"{int(row['source_gain_rank'])} | "
            f"{int(row['source_permutation_rank'])} | "
            f"{int(row['source_shap_rank'])} | "
            f"{row['grouped_auc_decrease_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "三种方法回答不同问题：Gain 统计分裂带来的训练损失下降，Permutation",
            "衡量固定测试 AUC 对打乱输入的敏感度，TreeSHAP 衡量单回合 log-odds",
            "贡献。相关经济列和差值列会分摊信号，因此不应只看一种排名。",
            "",
            "## 宏观特征组",
            "",
            "| M14 特征组 | 编码列数 | AUC 平均下降 | 标准差 |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in macro.iterrows():
        lines.append(
            f"| {row['feature_group']} | {int(row['encoded_column_count'])} | "
            f"{row['auc_decrease_mean']:.6f} | {row['auc_decrease_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 与 M12 XGBoost 的解释差异",
            "",
            "| 方法 | 43 列 Spearman | Top 10 交集 | Top 10 Jaccard |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in agreement.iterrows():
        lines.append(
            f"| {row['method']} | {row['spearman_rank']:.3f} | "
            f"{int(row['top_overlap_count'])}/10 | {row['top_jaccard']:.3f} |"
        )
    lines.extend(
        [
            "",
            "排名差异不构成验收失败。LightGBM 和 XGBoost 的树生长策略、分裂方式及",
            "相关特征归因方式不同；本对照只说明两个冻结模型如何使用同一批输入，不能",
            "证明某个特征对胜负具有因果作用。",
            "",
            "## SHAP 前 20 泄漏检查",
            "",
            "| 排名 | 编码列 | 原始特征 | 组 | 结果 |",
            "|---:|---|---|---|---|",
        ]
    )
    for _, row in audit.iterrows():
        lines.append(
            f"| {int(row['shap_rank'])} | {row['encoded_feature']} | "
            f"{row['source_feature']} | {row['feature_group']} | "
            f"{row['audit_result']} |"
        )
    lines.extend(
        [
            "",
            "## 三个固定案例",
            "",
            "| 类型 | series_id | game_id | round_id | 地图 | 真实标签 | CT 概率 |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for _, row in cases.iterrows():
        lines.append(
            f"| {row['case_type']} | {row['series_id']} | {row['game_id']} | "
            f"{row['round_id']} | {row['map_name']} | {int(row['y_true'])} | "
            f"{row['ct_win_probability']:.6f} |"
        )
    lines.extend(
        [
            "",
            "案例中的正 SHAP 推向 CT，负 SHAP 推向 T。它解释模型为什么输出该概率，",
            "不解释该回合后来发生的交火原因。",
            "",
            "## 外部指标",
            "",
            f"外部比较共 {len(external)} 行，与 M24 完全一致。最接近的购买结束 DNN",
            "仍使用不同数据和随机行级切分，其 Accuracy 和 Log Loss 只能作为背景参照。",
            "完整逐行差值见 `external_benchmark_comparison.csv`。",
            "",
            "## 验收与下一步",
            "",
            f"自动化测试 {summary['automated_tests']['test_count']} 项通过，源码编译通过。",
            "M25 的结论是模型解释链路完整、无特征时间泄漏、冻结预测未漂移。解释结果",
            "不会触发 test 驱动的重训。下一阶段 M26 建立购买结束 LightGBM 的单条",
            "JSON/CSV 预测接口，并复用 M24 validation-only 选择的 identity 校准器。",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(
    data_path: str | Path,
    model_path: str | Path,
    m24_summary_path: str | Path,
    m24_predictions_path: str | Path,
    m12_report_dir: str | Path,
    m24_external_path: str | Path,
    report_dir: str | Path,
    project_root: str | Path,
    *,
    permutation_repeats: int = 20,
    seed: int = 42,
    case_features: int = 10,
    shap_plot_rows: int = 1500,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = Path(project_root).resolve()
    data_path = _resolve(root, data_path).resolve()
    model_path = _resolve(root, model_path).resolve()
    m24_summary_path = _resolve(root, m24_summary_path).resolve()
    m24_predictions_path = _resolve(root, m24_predictions_path).resolve()
    m12_report_dir = _resolve(root, m12_report_dir).resolve()
    m24_external_path = _resolve(root, m24_external_path).resolve()
    report_dir = _resolve(root, report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    model_before = fingerprint_file(model_path)
    core_summary, tables = run_explanation_core(
        data_path,
        model_path,
        m24_summary_path,
        m24_predictions_path,
        permutation_repeats=permutation_repeats,
        seed=seed,
        case_features=case_features,
    )
    m24_summary = _read_json(m24_summary_path)
    xgboost_importance, xgboost_tables = load_m12_xgboost_importance(
        m12_report_dir
    )
    comparison_detail, agreement = build_model_importance_comparison(
        tables["importance_comparison"],
        xgboost_importance,
        top_n=10,
    )
    external = read_table(m24_external_path)
    external_audit = validate_external_comparison(external, m24_summary["metrics"])
    external_report = render_external_report(external)
    tables.update(
        {
            "xgboost_lightgbm_importance_comparison": comparison_detail,
            "model_importance_agreement": agreement,
            "external_benchmark_comparison": external,
        }
    )

    runtime_tables = {"x_test", "shap_values"}
    for name, table in tables.items():
        if name not in runtime_tables and name != "external_benchmark_comparison":
            table.to_csv(report_dir / f"{name}.csv", index=False)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    (report_dir / "external_benchmark_comparison.md").write_text(
        external_report,
        encoding="utf-8",
    )
    save_m25_plots(
        tables,
        report_dir,
        seed=seed,
        shap_plot_rows=shap_plot_rows,
    )

    automated_tests = run_automated_tests(root)
    compile_check = run_compile_check(root)
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests["output"], encoding="utf-8"
    )
    (report_dir / "source_compile_output.txt").write_text(
        compile_check["output"], encoding="utf-8"
    )
    test_count_match = re.search(r"Ran (\d+) tests?", automated_tests["output"])
    automated_test_count = int(test_count_match.group(1)) if test_count_match else None
    model_after = fingerprint_file(model_path)

    encoded_features = core_summary["encoded_features"]
    raw_features = core_summary["raw_features"]
    expected_case_types = {
        "ct_high_probability",
        "t_high_probability",
        "high_confidence_error",
    }
    importance_complete = bool(
        len(tables["gain_importance"]) == encoded_features
        and len(tables["permutation_importance_auc"]) == encoded_features
        and len(tables["shap_importance"]) == encoded_features
        and tables["permutation_importance_auc"]["n_repeats"]
        .eq(permutation_repeats)
        .all()
        and np.isclose(tables["gain_importance"]["gain_normalized"].sum(), 1.0)
    )
    feature_contract_passed = bool(
        core_summary["feature_audit"]["all_feature_failures"] == 0
        and core_summary["feature_audit"]["top20_failures"] == 0
        and core_summary["feature_audit"]["mapped_source_features"] == raw_features
        and len(tables["encoded_feature_contract"]) == encoded_features
    )
    group_outputs_complete = bool(
        len(tables["grouped_permutation_importance_auc"]) == raw_features
        and len(tables["macro_group_permutation_auc"])
        == len(PRE_ROUND_FEATURE_GROUPS)
        and tables["grouped_permutation_importance_auc"]["n_repeats"]
        .eq(permutation_repeats)
        .all()
        and tables["macro_group_permutation_auc"]["n_repeats"]
        .eq(permutation_repeats)
        .all()
        and set(tables["macro_group_permutation_auc"]["feature_group"])
        == set(PRE_ROUND_FEATURE_GROUPS)
    )
    comparison_complete = bool(
        len(xgboost_importance) == encoded_features
        and len(comparison_detail) == encoded_features
        and len(agreement) == 4
        and agreement["feature_count"].eq(encoded_features).all()
        and set(comparison_detail["feature"])
        == set(tables["importance_comparison"]["feature"])
        and len(xgboost_tables["gain"]) == encoded_features
        and len(xgboost_tables["permutation"]) == encoded_features
        and len(xgboost_tables["shap"]) == encoded_features
    )
    cases_complete = bool(
        len(tables["selected_cases"]) == 3
        and set(tables["selected_cases"]["case_type"]) == expected_case_types
        and len(tables["case_explanations"]) == 3 * case_features
        and tables["case_explanations"]
        .groupby("case_type")["contribution_rank"]
        .max()
        .eq(case_features)
        .all()
    )
    entrypoint = root / "scripts" / "run_pre_round_lightgbm_explanation.ps1"
    manifest_inputs = [
        data_path,
        model_path,
        m24_summary_path,
        m24_predictions_path,
        m12_report_dir / "gain_importance.csv",
        m12_report_dir / "permutation_importance_auc.csv",
        m12_report_dir / "shap_importance.csv",
        m12_report_dir / "importance_comparison.csv",
        m24_external_path,
        root / "docs" / "m25_pre_round_lightgbm_explanation_spec.md",
        root / "src" / "csdemo" / "m25_pre_round_lightgbm_explanation.py",
        entrypoint,
    ]
    manifest_ready = all(path.is_file() for path in manifest_inputs)
    checks = {
        "m24_prerequisite": core_summary["prerequisite"]["passed"],
        "model_replay": core_summary["model_replay"]["passed"],
        "model_unchanged": model_before["sha256"] == model_after["sha256"],
        "importance_methods": importance_complete,
        "feature_mapping_and_leakage": feature_contract_passed,
        "shap_reconstruction": core_summary[
            "shap_reconstruction_max_abs_error"
        ]
        <= 1e-10,
        "source_and_macro_groups": group_outputs_complete,
        "xgboost_explanation_comparison": comparison_complete,
        "case_explanations": cases_complete,
        "external_report": external_audit["passed"] and bool(external_report),
        "automated_tests": automated_tests["passed"],
        "source_compile": compile_check["passed"],
        "reproduction_entrypoint": entrypoint.is_file(),
        "artifact_manifest": manifest_ready,
    }
    acceptance = decide_acceptance(checks)
    summary = {
        "stage": "M25",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "model_policy": (
            "M23/M24 LightGBM frozen; no fit, tuning, feature selection, "
            "threshold change, or calibration change"
        ),
        "acceptance": acceptance,
        "checks": checks,
        "prerequisite": core_summary["prerequisite"],
        "model_replay": core_summary["model_replay"],
        "model_integrity": {
            "unchanged": checks["model_unchanged"],
            "sha256_before": model_before["sha256"],
            "sha256_after": model_after["sha256"],
            "lightgbm_fit_calls": 0,
        },
        "metrics": core_summary["model_replay"]["metrics"],
        "test_rounds": core_summary["test_rounds"],
        "raw_features": raw_features,
        "encoded_features": encoded_features,
        "available_tree_count": core_summary["available_tree_count"],
        "deployment_tree_count": core_summary["deployment_tree_count"],
        "permutation_repeats": permutation_repeats,
        "seed": seed,
        "shap_method": "lightgbm_native_tree_shap_pred_contrib",
        "shap_units": "log_odds",
        "shap_reconstruction_max_abs_error": core_summary[
            "shap_reconstruction_max_abs_error"
        ],
        "shap_reconstruction_mean_abs_error": core_summary[
            "shap_reconstruction_mean_abs_error"
        ],
        "feature_audit": core_summary["feature_audit"],
        "top_features": core_summary["top_features"],
        "importance_rank_spearman": core_summary["importance_rank_spearman"],
        "macro_group_permutation": json.loads(
            tables["macro_group_permutation_auc"].to_json(orient="records")
        ),
        "xgboost_explanation_agreement": json.loads(
            agreement.to_json(orient="records")
        ),
        "selected_cases": core_summary["selected_cases"],
        "external_comparison": external_audit,
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_test_count,
        },
        "source_compile": compile_check,
        "environment": {
            "python": sys.version.split()[0],
            "lightgbm": _package_version("lightgbm"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
        },
        "roadmap": {
            "pre_round_xgboost": "complete_through_M14",
            "first_kill_xgboost": "complete_through_M21",
            "pre_round_lightgbm_current": "M25_explanation_complete",
            "next_stage": "M26 pre-round LightGBM JSON/CSV prediction interface",
            "later_tracks": [
                "pre-round LightGBM final acceptance",
                "post-first-kill LightGBM controlled comparison",
                "real-time win probability data and model",
            ],
        },
        "next_stage": "M26 pre-round LightGBM prediction interface",
    }

    pd.DataFrame(
        [
            {"check": name, "passed": passed, "blocking": True}
            for name, passed in checks.items()
        ]
    ).to_csv(report_dir / "m25_checks.csv", index=False)
    write_json(summary, report_dir / "m25_summary.json")
    (report_dir / "m25_pre_round_lightgbm_explanation_report.md").write_text(
        render_m25_report(summary, tables),
        encoding="utf-8",
    )

    output_names = [
        path.name
        for path in report_dir.iterdir()
        if path.is_file() and path.name != "m25_experiment_manifest.json"
    ]
    output_artifacts = [
        fingerprint_file(report_dir / name) for name in sorted(output_names)
    ]
    manifest = {
        "stage": "M25",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            "python -m src.csdemo.m25_pre_round_lightgbm_explanation "
            "--permutation-repeats 20 --seed 42"
        ),
        "policy": "frozen model explanation; no training or test-driven selection",
        "parameters": {
            "permutation_repeats": permutation_repeats,
            "seed": seed,
            "case_features": case_features,
            "shap_plot_rows": shap_plot_rows,
        },
        "inputs": [fingerprint_file(path) for path in manifest_inputs],
        "model_sha256_before": model_before["sha256"],
        "model_sha256_after": model_after["sha256"],
        "outputs": output_artifacts,
        "acceptance": acceptance,
        "environment": summary["environment"],
    }
    write_json(manifest, report_dir / "m25_experiment_manifest.json")
    return summary, tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run M25 frozen pre-round LightGBM Gain, permutation, TreeSHAP, "
            "leakage audit, and XGBoost explanation comparison."
        )
    )
    parser.add_argument(
        "--data", default="data/processed/esta_full/pre_round.parquet"
    )
    parser.add_argument(
        "--model",
        default="models/esta_full_m23/pre_round_lightgbm_tuned.joblib",
    )
    parser.add_argument(
        "--m24-summary", default="reports/esta_full_m24/m24_summary.json"
    )
    parser.add_argument(
        "--m24-predictions",
        default="reports/esta_full_m24/test_predictions_enriched.csv",
    )
    parser.add_argument("--m12-report-dir", default="reports/esta_full_m12")
    parser.add_argument(
        "--m24-external",
        default="reports/esta_full_m24/external_benchmark_comparison.csv",
    )
    parser.add_argument("--report-dir", default="reports/esta_full_m25")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-features", type=int, default=10)
    parser.add_argument("--shap-plot-rows", type=int, default=1500)
    args = parser.parse_args()

    summary, tables = run(
        data_path=args.data,
        model_path=args.model,
        m24_summary_path=args.m24_summary,
        m24_predictions_path=args.m24_predictions,
        m12_report_dir=args.m12_report_dir,
        m24_external_path=args.m24_external,
        report_dir=args.report_dir,
        project_root=args.project_root,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
        case_features=args.case_features,
        shap_plot_rows=args.shap_plot_rows,
    )
    print(
        tables["source_feature_importance"][
            [
                "source_feature",
                "feature_group",
                "source_gain_rank",
                "source_permutation_rank",
                "source_shap_rank",
                "grouped_auc_decrease_mean",
            ]
        ]
        .head(12)
        .round(6)
        .to_string(index=False)
    )
    print(
        tables["model_importance_agreement"]
        .round(6)
        .to_string(index=False)
    )
    print(
        f"M25 {summary['acceptance']['status']}; "
        f"ready_for_m26={summary['acceptance']['ready_for_m26']}"
    )
    if not summary["acceptance"]["ready_for_m26"]:
        raise SystemExit(1)


def decide_acceptance(checks: Mapping[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not checks.get(name, False)]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "m25_lightgbm_explanation_complete": not failures,
        "ready_for_m26": not failures,
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

from .io import read_table
from .m9_evaluation import METRIC_ORDER, bootstrap_metric_intervals
from .m10_calibration import (
    CALIBRATION_METHODS,
    calibration_curves,
    cross_validated_comparison,
    evaluate_test_calibrators,
    fit_full_calibrators,
    select_calibration_method,
)
from .m11_robustness import (
    assign_equipment_band,
    assign_round_stage,
    group_metrics_with_intervals,
    select_high_confidence_errors,
)
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m16_first_kill_baselines import (
    audit_training_data,
    canonical_feature_names,
    compare_external_models,
    prepare_profile_splits,
    write_json,
)
from .metrics import probability_metrics
from .schema import ID_COLUMNS

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEY_COLUMNS = tuple(ID_COLUMNS)
TIME_BAND_LABELS = (
    "fast_00_15",
    "normal_15_30",
    "late_30_60",
    "very_late_60_plus",
)
WEAPON_FAMILY_LABELS = (
    "rifle",
    "sniper",
    "pistol",
    "smg_shotgun",
    "utility_other",
)

RIFLES = {"AK-47", "AUG", "FAMAS", "Galil AR", "M4A1", "M4A4", "SG 553"}
SNIPERS = {"AWP", "G3SG1", "SCAR-20", "SSG 08"}
PISTOLS = {
    "CZ75 Auto",
    "Desert Eagle",
    "Dual Berettas",
    "Five-SeveN",
    "Glock-18",
    "P2000",
    "P250",
    "Tec-9",
    "USP-S",
}
SMGS_AND_SHOTGUNS = {
    "MAC-10",
    "MAG-7",
    "MP5-SD",
    "MP7",
    "MP9",
    "Nova",
    "P90",
    "PP-Bizon",
    "UMP-45",
    "XM1014",
}

ANALYSIS_CONTEXT_COLUMNS = (
    "map_name",
    "round_num",
    "eq_value_diff_ct",
    "first_kill_time",
    "first_kill_advantage_ct",
    "first_kill_weapon",
    "first_kill_headshot",
)

BLOCKING_CHECKS = (
    "m17_prerequisite",
    "split_and_key_contract",
    "prediction_replay",
    "global_bootstrap",
    "global_metric_minimum",
    "group_outputs",
    "source_stability",
    "large_map_minimum",
    "calibration_protocol",
    "calibration_no_material_harm",
    "error_review",
    "external_report",
    "automated_tests",
)


def parse_source_subset(game_ids: pd.Series) -> pd.Series:
    values = game_ids.astype("string")
    parsed = values.str.extract(r"^(lan|online):", expand=False)
    if parsed.isna().any():
        examples = values.loc[parsed.isna()].head(3).tolist()
        raise ValueError(f"Could not parse LAN/online source from game_id: {examples}")
    return pd.Series(parsed, index=game_ids.index, dtype="string")


def assign_first_kill_time_band(first_kill_time: pd.Series) -> pd.Series:
    values = pd.to_numeric(first_kill_time, errors="raise")
    invalid = values.isna() | ~np.isfinite(values) | values.lt(0)
    if invalid.any():
        raise ValueError("First-kill time must be finite and non-negative")
    return pd.Series(
        pd.cut(
            values,
            bins=[0, 15, 30, 60, np.inf],
            labels=TIME_BAND_LABELS,
            right=False,
            include_lowest=True,
        ).astype("string"),
        index=first_kill_time.index,
    )


def assign_weapon_family(first_kill_weapon: pd.Series) -> pd.Series:
    values = first_kill_weapon.astype("string")
    labels = np.select(
        [
            values.isin(RIFLES),
            values.isin(SNIPERS),
            values.isin(PISTOLS),
            values.isin(SMGS_AND_SHOTGUNS),
        ],
        WEAPON_FAMILY_LABELS[:-1],
        default=WEAPON_FAMILY_LABELS[-1],
    )
    return pd.Series(labels, index=first_kill_weapon.index, dtype="string")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[list(KEY_COLUMNS)].astype("string"))


def _validate_unique_keys(frame: pd.DataFrame, label: str) -> None:
    null_cells = int(frame[list(KEY_COLUMNS)].isna().sum().sum())
    duplicate_rows = int(frame.duplicated(list(KEY_COLUMNS)).sum())
    if null_cells or duplicate_rows:
        raise ValueError(
            f"{label} has invalid complete keys: null_cells={null_cells}, "
            f"duplicate_rows={duplicate_rows}"
        )


def prepare_analysis_table(
    predictions: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    prediction_required = set(KEY_COLUMNS) | {
        "y_true",
        "ct_win_probability",
        "predicted_label",
    }
    _require_columns(predictions, prediction_required, "M18 predictions")
    _require_columns(features, set(KEY_COLUMNS) | {"split", "ct_win"}, "M18 data")
    _validate_unique_keys(predictions, "M18 predictions")

    test_features = features.loc[features["split"].eq("test")].copy()
    _validate_unique_keys(test_features, "M18 test data")
    prediction_keys = _key_index(predictions)
    feature_keys = _key_index(test_features)
    missing_from_data = int((~prediction_keys.isin(feature_keys)).sum())
    missing_from_predictions = int((~feature_keys.isin(prediction_keys)).sum())
    if missing_from_data or missing_from_predictions:
        raise ValueError(
            "M18 predictions and test data do not have the same complete key set: "
            f"missing_from_data={missing_from_data}, "
            f"missing_from_predictions={missing_from_predictions}"
        )

    _require_columns(
        test_features,
        set(ANALYSIS_CONTEXT_COLUMNS),
        "M18 test analysis data",
    )
    base_columns = list(KEY_COLUMNS) + [
        "y_true",
        "ct_win_probability",
        "predicted_label",
    ]
    context_columns = ["ct_win", *ANALYSIS_CONTEXT_COLUMNS]
    analysis = predictions[base_columns].merge(
        test_features[list(KEY_COLUMNS) + context_columns],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    label_mismatches = int(
        analysis["y_true"].astype(int).ne(analysis["ct_win"].astype(int)).sum()
    )
    if label_mismatches:
        raise ValueError(f"M18 joined labels disagree in {label_mismatches} rows")

    probability = analysis["ct_win_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("M18 prediction probabilities must be finite and in [0, 1]")
    expected_label = (probability >= 0.5).astype(int)
    if not np.array_equal(expected_label, analysis["predicted_label"].to_numpy(dtype=int)):
        raise ValueError("M18 predicted labels do not match the 0.5 threshold")

    advantage = pd.to_numeric(analysis["first_kill_advantage_ct"], errors="raise")
    invalid_advantage = ~advantage.isin([-1, 1])
    if invalid_advantage.any():
        raise ValueError("M18 requires a valid CT or T first-kill advantage in every row")

    analysis["source_subset"] = parse_source_subset(analysis["game_id"])
    analysis["round_stage"] = assign_round_stage(analysis["round_num"])
    analysis["equipment_band"] = assign_equipment_band(
        analysis["eq_value_diff_ct"]
    )
    analysis["first_kill_time_band"] = assign_first_kill_time_band(
        analysis["first_kill_time"]
    )
    analysis["first_kill_weapon_family"] = assign_weapon_family(
        analysis["first_kill_weapon"]
    )
    analysis["first_kill_side"] = np.where(advantage.eq(1), "CT", "T")
    analysis["first_kill_headshot_label"] = np.where(
        pd.to_numeric(analysis["first_kill_headshot"], errors="raise").eq(1),
        "headshot",
        "non_headshot",
    )
    analysis["t_win_probability"] = 1.0 - probability
    analysis["correct"] = analysis["predicted_label"].eq(analysis["y_true"])
    return analysis


def audit_prediction_replay(
    saved: pd.DataFrame,
    replayed: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    _require_columns(
        saved,
        set(KEY_COLUMNS) | {"xgboost_tuned_probability"},
        "M17 saved predictions",
    )
    _require_columns(
        replayed,
        set(KEY_COLUMNS) | {"ct_win_probability"},
        "M18 replayed predictions",
    )
    saved_duplicate_rows = int(saved.duplicated(list(KEY_COLUMNS)).sum())
    replayed_duplicate_rows = int(replayed.duplicated(list(KEY_COLUMNS)).sum())

    saved_keys = _key_index(saved)
    replayed_keys = _key_index(replayed)
    key_mismatch_count = int(
        (~saved_keys.isin(replayed_keys)).sum()
        + (~replayed_keys.isin(saved_keys)).sum()
    )
    max_difference: float | None = None
    invalid_probability_cells = 0
    if not saved_duplicate_rows and not replayed_duplicate_rows and not key_mismatch_count:
        joined = saved[list(KEY_COLUMNS) + ["xgboost_tuned_probability"]].merge(
            replayed[list(KEY_COLUMNS) + ["ct_win_probability"]],
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        values = joined[
            ["xgboost_tuned_probability", "ct_win_probability"]
        ].to_numpy(dtype=float)
        invalid_probability_cells = int(
            (~np.isfinite(values)).sum() + ((values < 0) | (values > 1)).sum()
        )
        if not invalid_probability_cells:
            max_difference = float(
                np.max(np.abs(values[:, 0] - values[:, 1]), initial=0.0)
            )

    passed = bool(
        not saved_duplicate_rows
        and not replayed_duplicate_rows
        and not key_mismatch_count
        and not invalid_probability_cells
        and max_difference is not None
        and max_difference <= tolerance
    )
    return {
        "passed": passed,
        "tolerance": tolerance,
        "saved_rows": int(len(saved)),
        "replayed_rows": int(len(replayed)),
        "saved_duplicate_key_rows": saved_duplicate_rows,
        "replayed_duplicate_key_rows": replayed_duplicate_rows,
        "key_mismatch_count": key_mismatch_count,
        "invalid_probability_cells": invalid_probability_cells,
        "max_absolute_probability_difference": max_difference,
    }


def post_first_kill_error_pattern(row: pd.Series) -> str:
    predicted_ct = int(row["predicted_label"]) == 1
    expected_advantage = 1 if predicted_ct else -1
    first_kill_support = int(row["first_kill_advantage_ct"]) == expected_advantage
    equipment_difference = float(row["eq_value_diff_ct"])
    favored_equipment = equipment_difference if predicted_ct else -equipment_difference
    equipment_support = favored_equipment >= 1500
    if first_kill_support and equipment_support:
        return "first_kill_and_equipment_agree"
    if first_kill_support:
        return "first_kill_only"
    if equipment_support:
        return "equipment_only"
    return "neither"


def bootstrap_source_auc_gap(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    required = {"series_id", "source_subset", "y_true", "ct_win_probability"}
    _require_columns(predictions, required, "M18 source comparison")
    sources = set(predictions["source_subset"].dropna().astype(str))
    if sources != {"lan", "online"}:
        raise ValueError(f"M18 source comparison requires LAN and online; found {sources}")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    def auc_for(frame: pd.DataFrame, source: str) -> float:
        part = frame.loc[frame["source_subset"].eq(source)]
        return float(
            probability_metrics(part["y_true"], part["ct_win_probability"])["auc"]
        )

    lan_auc = auc_for(predictions, "lan")
    online_auc = auc_for(predictions, "online")
    series = predictions["series_id"].astype(str).to_numpy()
    unique_series = np.unique(series)
    positions = [np.flatnonzero(series == value) for value in unique_series]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(positions), size=len(positions))
        sampled_positions = np.concatenate([positions[index] for index in selected])
        sampled = predictions.iloc[sampled_positions]
        if set(sampled["source_subset"].astype(str)) != {"lan", "online"}:
            continue
        sampled_lan = auc_for(sampled, "lan")
        sampled_online = auc_for(sampled, "online")
        difference = sampled_lan - sampled_online
        if np.isfinite(difference):
            differences.append(difference)
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise RuntimeError("M18 source bootstrap produced no valid AUC differences")
    return {
        "comparison": "lan_minus_online_auc",
        "lan_auc": lan_auc,
        "online_auc": online_auc,
        "signed_difference": lan_auc - online_auc,
        "absolute_difference": abs(lan_auc - online_auc),
        "ci_lower_95": float(np.quantile(values, 0.025)),
        "ci_upper_95": float(np.quantile(values, 0.975)),
        "ci_includes_zero": bool(np.quantile(values, 0.025) <= 0 <= np.quantile(values, 0.975)),
        "successful_bootstraps": int(values.size),
        "bootstrap_unit": "series_id_global",
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def verify_m17_prerequisite(
    data_path: str | Path,
    model_path: str | Path,
    m17_summary: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    data_artifact = fingerprint_file(data_path)
    model_artifact = fingerprint_file(model_path)
    expected_data_sha = m17_summary.get("data", {}).get("sha256")
    expected_model_sha = (
        m17_summary.get("frozen_model", {}).get("artifact", {}).get("sha256")
    )
    expected_features = m17_summary.get("features", {}).get("raw_features", [])
    actual_features = bundle.get("raw_features", [])
    encoded_columns = bundle.get("columns", [])
    checks = {
        "m17_accepted": bool(
            m17_summary.get("acceptance", {}).get("ready_for_m18", False)
        ),
        "m17_task": m17_summary.get("task") == "post_first_kill",
        "data_sha256": bool(expected_data_sha)
        and data_artifact["sha256"] == expected_data_sha
        and bundle.get("data_sha256") == expected_data_sha,
        "model_sha256": bool(expected_model_sha)
        and model_artifact["sha256"] == expected_model_sha,
        "bundle_task": bundle.get("task") == "first_kill",
        "raw_feature_contract": bool(expected_features)
        and actual_features == expected_features
        and actual_features == canonical_feature_names(),
        "encoded_feature_contract": bool(encoded_columns)
        and len(encoded_columns) == len(set(encoded_columns))
        and len(encoded_columns)
        == int(m17_summary.get("features", {}).get("encoded_count", -1)),
        "predictor_available": hasattr(bundle.get("model"), "predict_proba"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "data_artifact": data_artifact,
        "model_artifact": model_artifact,
        "expected_data_sha256": expected_data_sha,
        "expected_model_sha256": expected_model_sha,
    }


def replay_model_predictions(
    data: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_features = list(bundle["raw_features"])
    prepared = prepare_profile_splits(data, raw_features)
    encoded_columns = prepared["train"][0].columns.tolist()
    expected_columns = list(bundle["columns"])
    if encoded_columns != expected_columns:
        raise ValueError("M18 encoded feature columns do not match the M17 model")

    outputs: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        x, y, identity = prepared[split]
        probability = np.asarray(
            bundle["model"].predict_proba(x)[:, 1], dtype=float
        )
        metadata = data.loc[
            data["split"].eq(split), list(KEY_COLUMNS) + ["map_name"]
        ].copy()
        _validate_unique_keys(metadata, f"M18 {split} metadata")
        output = identity.copy()
        output["y_true"] = y.to_numpy(dtype=int)
        output["ct_win_probability"] = probability
        output["t_win_probability"] = 1.0 - probability
        output["predicted_label"] = (probability >= 0.5).astype(int)
        output["correct"] = output["predicted_label"].eq(output["y_true"])
        output = output.merge(
            metadata,
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        output["source_subset"] = parse_source_subset(output["game_id"])
        outputs[split] = output

    audit = {
        "passed": True,
        "raw_feature_count": len(raw_features),
        "encoded_feature_count": len(encoded_columns),
        "encoded_columns_match_m17": encoded_columns == expected_columns,
        "split_rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "xgboost_fit_calls": 0,
    }
    return outputs, audit


def audit_calibration_protocol(
    validation_predictions: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    validation_oof: pd.DataFrame,
    selected_method: str,
    *,
    n_splits: int,
) -> dict[str, Any]:
    expected_probability_columns = {
        f"probability_{method}" for method in CALIBRATION_METHODS
    }
    probability_columns_present = expected_probability_columns.issubset(
        validation_oof.columns
    )
    unique_keys = not validation_oof.duplicated(list(KEY_COLUMNS)).any()
    exact_rows = len(validation_oof) == len(validation_predictions)
    series_one_fold = bool(
        validation_oof.groupby("series_id")["fold"].nunique().eq(1).all()
    )
    fold_count = int(validation_oof["fold"].nunique())
    finite_probabilities = False
    if probability_columns_present:
        probability = validation_oof[
            sorted(expected_probability_columns)
        ].to_numpy(dtype=float)
        finite_probabilities = bool(
            np.isfinite(probability).all()
            and ((probability >= 0) & (probability <= 1)).all()
        )
    selected_from_oof = selected_method == select_calibration_method(
        validation_comparison
    )
    methods_complete = set(validation_comparison["method"].astype(str)) == set(
        CALIBRATION_METHODS
    )
    checks = {
        "exact_rows": exact_rows,
        "unique_complete_keys": unique_keys,
        "series_one_fold": series_one_fold,
        "fold_count": fold_count == n_splits,
        "probability_columns": probability_columns_present,
        "finite_probabilities": finite_probabilities,
        "methods_complete": methods_complete,
        "selected_from_oof": selected_from_oof,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selection_data": "validation only",
        "selection_rule": "lowest grouped OOF log_loss, then brier_score, then method",
        "validation_folds": n_splits,
        "selected_method": selected_method,
    }


def enrich_high_confidence_errors(analysis: pd.DataFrame) -> pd.DataFrame:
    errors = select_high_confidence_errors(
        analysis,
        minimum_confidence=0.8,
        max_cases=None,
    )
    errors["predicted_side"] = np.where(
        errors["predicted_label"].eq(1), "CT", "T"
    )
    errors["actual_winner"] = np.where(errors["y_true"].eq(1), "CT", "T")
    errors["signal_pattern"] = errors.apply(
        post_first_kill_error_pattern, axis=1
    )
    return errors


def summarize_high_confidence_errors(errors: pd.DataFrame) -> pd.DataFrame:
    dimensions = (
        "signal_pattern",
        "predicted_side",
        "first_kill_side",
        "first_kill_time_band",
        "first_kill_weapon_family",
        "first_kill_headshot_label",
    )
    rows = []
    denominator = len(errors)
    for dimension in dimensions:
        for value, count in errors[dimension].value_counts(dropna=False).items():
            rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "cases": int(count),
                    "share": float(count / denominator) if denominator else 0.0,
                }
            )
    return pd.DataFrame(rows)


def assess_global_intervals(
    intervals: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> dict[str, Any]:
    indexed = intervals.set_index("metric")
    successful = bool(
        set(METRIC_ORDER).issubset(indexed.index)
        and indexed.loc[list(METRIC_ORDER), "successful_bootstraps"]
        .eq(n_bootstrap)
        .all()
    )
    auc_lower = float(indexed.loc["auc", "ci_lower_95"])
    log_loss_upper = float(indexed.loc["log_loss", "ci_upper_95"])
    return {
        "bootstrap_complete": successful,
        "auc_ci_lower_95": auc_lower,
        "log_loss_ci_upper_95": log_loss_upper,
        "minimum_passed": bool(auc_lower >= 0.780 and log_loss_upper <= 0.550),
        "stage_passed": bool(auc_lower >= 0.790 and log_loss_upper <= 0.540),
        "minimum_thresholds": {
            "auc_ci_lower_95": 0.780,
            "log_loss_ci_upper_95": 0.550,
        },
        "stage_thresholds": {
            "auc_ci_lower_95": 0.790,
            "log_loss_ci_upper_95": 0.540,
        },
    }


def assess_group_robustness(
    grouped: dict[str, pd.DataFrame],
    source_gap: dict[str, Any],
) -> dict[str, Any]:
    expected_groups = {
        "map",
        "source",
        "round_stage",
        "equipment_band",
        "first_kill_side",
        "first_kill_time_band",
        "first_kill_weapon_family",
        "first_kill_headshot",
    }
    outputs_complete = set(grouped) == expected_groups and all(
        not table.empty for table in grouped.values()
    )
    large_maps = grouped["map"].loc[grouped["map"]["rounds"].ge(300)].copy()
    if large_maps.empty:
        raise RuntimeError("M18 found no map with at least 300 test rounds")
    minimum_auc = float(large_maps["auc"].min())
    minimum_auc_ci_lower = float(large_maps["auc_ci_lower_95"].min())
    return {
        "outputs_complete": outputs_complete,
        "large_map_count": int(len(large_maps)),
        "large_map_min_auc": minimum_auc,
        "large_map_min_auc_ci_lower": minimum_auc_ci_lower,
        "large_map_minimum_passed": bool(minimum_auc >= 0.740),
        "large_map_stage_passed": bool(minimum_auc >= 0.770),
        "large_map_ci_stage_passed": bool(minimum_auc_ci_lower >= 0.700),
        "source_gap_passed": bool(source_gap["absolute_difference"] <= 0.040),
        "source_gap_ci_includes_zero": bool(source_gap["ci_includes_zero"]),
    }


def assess_calibration(
    test_comparison: pd.DataFrame,
    selected_method: str,
) -> dict[str, Any]:
    selected = test_comparison.loc[
        test_comparison["method"].eq(selected_method)
    ].iloc[0]
    raw = test_comparison.loc[
        test_comparison["method"].eq("uncalibrated")
    ].iloc[0]
    log_loss_change = float(selected["log_loss"] - raw["log_loss"])
    brier_change = float(selected["brier_score"] - raw["brier_score"])
    ece_change = float(selected["ece10"] - raw["ece10"])
    return {
        "selected_method": selected_method,
        "selected_test_metrics": {
            metric: float(selected[metric]) for metric in METRIC_ORDER
        },
        "changes_vs_uncalibrated": {
            "log_loss": log_loss_change,
            "brier_score": brier_change,
            "ece10": ece_change,
        },
        "no_material_probability_metric_harm": bool(
            log_loss_change <= 0.002 and brier_change <= 0.001
        ),
        "test_ece_stage_passed": bool(float(selected["ece10"]) <= 0.030),
    }


def decide_acceptance(checks: dict[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not checks.get(name, False)]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "ready_for_m19": not failures,
    }


def write_case_review(cases: pd.DataFrame, path: Path) -> None:
    lines = [
        "# M18 首杀后高置信度错误复核",
        "",
        "定义：预测错误且预测方概率不低于 0.80。以下模式是事后描述，不是因果结论。",
        "",
    ]
    for index, row in cases.reset_index(drop=True).iterrows():
        lines.append(
            f"{index + 1}. `{row['series_id']} / {row['game_id']} / "
            f"{row['round_id']}`：预测 {row['predicted_side']} "
            f"({row['assigned_side_probability']:.3f})，实际 {row['actual_winner']}；"
            f"首杀 {row['first_kill_side']}，时间 {row['first_kill_time']:.2f}s，"
            f"武器 {row['first_kill_weapon']}，装备差 CT "
            f"{row['eq_value_diff_ct']:+.0f}；`{row['signal_pattern']}`。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_external_report(external: pd.DataFrame) -> str:
    lines = [
        "# M18 外部模型指标差距",
        "",
        "差值统一为“当前项目指标 - 外部报告指标”。Accuracy/AUC 同时换算成百分点。",
        "不同数据集、切分、特征和预测时点使这些差值不能解释为纯算法优劣。",
        "",
        "| 可比性 | 当前模型 | 外部来源 | 指标 | 当前 | 外部 | 差值 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in external.to_dict(orient="records"):
        title = row.get("source_title", row["benchmark_id"])
        url = row.get("source_url", "")
        source = f"[{title}]({url})" if url else title
        difference = float(row["raw_difference_ours_minus_reported"])
        difference_text = (
            f"{difference * 100:+.2f} 百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| {row.get('comparability', '')} | `{row['current_model']}` | "
            f"{source} | {row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {difference_text} |"
        )
    return "\n".join(lines) + "\n"


def save_plots(
    map_metrics: pd.DataFrame,
    calibration_curve: pd.DataFrame,
    selected_method: str,
    error_summary: pd.DataFrame,
    report_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    maps = map_metrics.sort_values("auc").copy()
    maps["display_name"] = maps.apply(
        lambda row: f"{row['map_name']} (n={row['rounds']}, series={row['series']})",
        axis=1,
    )
    lower = np.maximum(maps["auc"] - maps["auc_ci_lower_95"], 0)
    upper = np.maximum(maps["auc_ci_upper_95"] - maps["auc"], 0)
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.errorbar(
        maps["auc"],
        maps["display_name"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#176B87",
        ecolor="#6B7280",
        capsize=3,
    )
    ax.axvline(0.740, color="#C44E52", linestyle="--", linewidth=1, label="Minimum 0.740")
    ax.axvline(0.770, color="#2A9D8F", linestyle=":", linewidth=1.5, label="Target 0.770")
    ax.set(
        xlabel="Test AUC with series-level 95% CI",
        ylabel="Map",
        title="M18 post-first-kill map robustness",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / "map_auc_with_ci.png", dpi=160)
    plt.close(fig)

    colors = {
        "uncalibrated": "#6B7280",
        "sigmoid": "#176B87",
        "isotonic": "#C44E52",
    }
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1)
    for method in CALIBRATION_METHODS:
        part = calibration_curve.loc[
            calibration_curve["method"].eq(method)
            & calibration_curve["count"].gt(0)
        ]
        selected = method == selected_method
        ax.plot(
            part["mean_probability"],
            part["observed_ct_win_rate"],
            marker="o",
            color=colors[method],
            linewidth=2.5 if selected else 1.5,
            alpha=1.0 if selected else 0.75,
            label=f"{method}{' (selected)' if selected else ''}",
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted CT probability",
        ylabel="Observed CT win rate",
        title="M18 test reliability comparison",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(report_dir / "reliability_comparison.png", dpi=160)
    plt.close(fig)

    patterns = error_summary.loc[
        error_summary["dimension"].eq("signal_pattern")
    ].sort_values("cases")
    if not patterns.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.barh(
            patterns["value"].astype(str).str.replace("_", " "),
            patterns["cases"],
            color="#C44E52",
        )
        ax.set(
            xlabel="High-confidence wrong rounds",
            ylabel="Signal pattern",
            title="M18 high-confidence error patterns",
        )
        fig.tight_layout()
        fig.savefig(report_dir / "error_pattern_counts.png", dpi=160)
        plt.close(fig)


def render_m18_report(
    summary: dict[str, Any],
    global_intervals: pd.DataFrame,
    map_metrics: pd.DataFrame,
    source_metrics: pd.DataFrame,
    validation_calibration: pd.DataFrame,
    test_calibration: pd.DataFrame,
    external: pd.DataFrame,
) -> str:
    acceptance = summary["acceptance"]
    global_assessment = summary["global_assessment"]
    robustness = summary["robustness"]
    source_gap = summary["source_auc_gap"]
    calibration = summary["calibration"]
    lines = [
        "# M18 首杀后固定模型评估报告",
        "",
        "## 阶段结论",
        "",
        f"阻断验收状态：**{acceptance['status']}**；可进入 M19："
        f"**{acceptance['ready_for_m19']}**。",
        "本阶段没有重训或调参 XGBoost；只回放 M17 冻结模型并做统计评估。",
        f"完整主键为 `{'+'.join(KEY_COLUMNS)}`，测试概率回放最大绝对误差为 "
        f"`{summary['prediction_replay']['max_absolute_probability_difference']:.3e}`。",
        "",
        "## 70/20/10 与主键",
        "",
        f"总样本 {summary['data']['rows']:,}，系列赛 {summary['data']['series']:,}；"
        f"train/validation/test 行数为 "
        f"{summary['data']['split_rows']['train']:,}/"
        f"{summary['data']['split_rows']['val']:,}/"
        f"{summary['data']['split_rows']['test']:,}。",
        f"实际行比例为 {summary['data']['split_percentages']['train']:.2f}% / "
        f"{summary['data']['split_percentages']['val']:.2f}% / "
        f"{summary['data']['split_percentages']['test']:.2f}%；"
        "同一 series_id 没有跨 split。",
        "",
        "## 整体置信区间",
        "",
        "| 指标 | 点估计 | 95% CI | 成功次数 |",
        "|---|---:|---:|---:|",
    ]
    for row in global_intervals.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['point_estimate']:.6f} | "
            f"[{row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}] | "
            f"{int(row['successful_bootstraps'])} |"
        )
    lines.extend(
        [
            "",
            f"AUC CI 下界最低线 0.780：{global_assessment['auc_ci_lower_95']:.6f}，"
            f"通过 `{global_assessment['auc_ci_lower_95'] >= 0.780}`。",
            f"Log Loss CI 上界最低线 0.550："
            f"{global_assessment['log_loss_ci_upper_95']:.6f}，"
            f"通过 `{global_assessment['log_loss_ci_upper_95'] <= 0.550}`。",
            "",
            "## 地图与来源",
            "",
            "| 地图 | 回合 | 系列 | AUC | AUC 95% CI | Log Loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in map_metrics.to_dict(orient="records"):
        lines.append(
            f"| {row['map_name']} | {int(row['rounds'])} | {int(row['series'])} | "
            f"{row['auc']:.6f} | [{row['auc_ci_lower_95']:.6f}, "
            f"{row['auc_ci_upper_95']:.6f}] | {row['log_loss']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| 来源 | 回合 | 系列 | AUC | AUC 95% CI | Log Loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in source_metrics.to_dict(orient="records"):
        lines.append(
            f"| {row['source_subset']} | {int(row['rounds'])} | "
            f"{int(row['series'])} | {row['auc']:.6f} | "
            f"[{row['auc_ci_lower_95']:.6f}, {row['auc_ci_upper_95']:.6f}] | "
            f"{row['log_loss']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"LAN-online AUC 点差为 `{source_gap['signed_difference']:+.6f}`，"
            f"绝对差 `{source_gap['absolute_difference']:.6f}`，95% CI "
            f"[{source_gap['ci_lower_95']:.6f}, {source_gap['ci_upper_95']:.6f}]。",
            f"主要地图最低 AUC 为 `{robustness['large_map_min_auc']:.6f}`；"
            f"最低 CI 下界为 `{robustness['large_map_min_auc_ci_lower']:.6f}`。",
            "",
            "## 校准",
            "",
            f"只根据 validation 的 5 折 GroupKFold OOF 结果选择 "
            f"`{calibration['selected_method']}`。",
            "",
            "| 数据 | 方法 | Log Loss | Brier | ECE10 | AUC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split, table in (
        ("validation_oof", validation_calibration),
        ("test", test_calibration),
    ):
        for row in table.to_dict(orient="records"):
            lines.append(
                f"| {split} | {row['method']} | {row['log_loss']:.6f} | "
                f"{row['brier_score']:.6f} | {row['ece10']:.6f} | "
                f"{row['auc']:.6f} |"
            )
    changes = calibration["changes_vs_uncalibrated"]
    lines.extend(
        [
            "",
            f"所选方法相对原始概率：Log Loss `{changes['log_loss']:+.6f}`，"
            f"Brier `{changes['brier_score']:+.6f}`，ECE10 "
            f"`{changes['ece10']:+.6f}`。",
            f"概率指标无明显伤害："
            f"`{calibration['no_material_probability_metric_harm']}`。",
            "",
            "## 高置信度错误",
            "",
            f"共有 {summary['errors']['available']} 个概率至少 0.80 的错误回合，"
            f"已复核前 {summary['errors']['reviewed']} 个。",
            "错误模式表和完整案例保存在本阶段 CSV；这些模式不代表因果关系。",
            "",
            "## 与外部模型差多少",
            "",
            "| 可比性 | 当前模型 | 外部来源 | 指标 | 当前 | 外部 | 差值 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in external.to_dict(orient="records"):
        difference = float(row["raw_difference_ours_minus_reported"])
        difference_text = (
            f"{difference * 100:+.2f} 百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| {row.get('comparability', '')} | `{row['current_model']}` | "
            f"{row.get('source_title', row['benchmark_id'])} | {row['metric']} | "
            f"{row['current_value']:.6f} | {row['reported_value']:.6f} | "
            f"{difference_text} |"
        )
    lines.extend(
        [
            "",
            "外部结果使用不同数据、切分和预测时点，只能作为参照，不能作为同场排名。",
            "M18 模型与 M17 相同，因此点指标差值不变；本阶段新增的是统计不确定性和"
            "分组证据。",
            "",
            "## 下一阶段",
            "",
            "M19 将对首杀后模型执行 gain、Permutation Importance、SHAP 和特征泄漏审计，"
            "重点解释首杀方、首杀时间、武器与购买状态怎样影响预测。",
            "",
            "运行命令：",
            "",
            "```powershell",
            ".\\scripts\\run_first_kill_evaluation.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    data_path: str | Path,
    model_path: str | Path,
    m17_summary_path: str | Path,
    m17_predictions_path: str | Path,
    m17_comparison_path: str | Path,
    benchmarks_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path,
    project_root: str | Path,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
    n_splits: int = 5,
    review_cases: int = 30,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    data_path = Path(data_path)
    model_path = Path(model_path)
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    project_root = Path(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m17_summary = _read_json(m17_summary_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("M18 expected the M17 model artifact to contain a bundle")
    prerequisite = verify_m17_prerequisite(
        data_path, model_path, m17_summary, bundle
    )
    if not prerequisite["passed"]:
        raise RuntimeError("M18 input does not match the accepted M17 artifacts")

    data = read_table(data_path)
    data_audit = audit_training_data(data)
    expected_split_rows = {
        name: int(value)
        for name, value in m17_summary.get("data", {})
        .get("split_rows", {})
        .items()
    }
    split_counts_match = data_audit.get("split_rows", {}) == expected_split_rows
    split_and_key_contract = bool(data_audit["passed"] and split_counts_match)
    if not split_and_key_contract:
        raise RuntimeError("M18 data identity or split membership differs from M17")

    replayed, model_replay_audit = replay_model_predictions(data, bundle)
    saved_predictions = read_table(m17_predictions_path)
    prediction_replay = audit_prediction_replay(
        saved_predictions,
        replayed["test"],
        tolerance=1e-12,
    )
    if not prediction_replay["passed"]:
        raise RuntimeError("M18 could not exactly replay the M17 test probabilities")

    analysis = prepare_analysis_table(replayed["test"], data)
    global_intervals = bootstrap_metric_intervals(
        analysis,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    global_assessment = assess_global_intervals(
        global_intervals,
        n_bootstrap=n_bootstrap,
    )

    group_columns = {
        "map": "map_name",
        "source": "source_subset",
        "round_stage": "round_stage",
        "equipment_band": "equipment_band",
        "first_kill_side": "first_kill_side",
        "first_kill_time_band": "first_kill_time_band",
        "first_kill_weapon_family": "first_kill_weapon_family",
        "first_kill_headshot": "first_kill_headshot_label",
    }
    grouped = {
        output_name: group_metrics_with_intervals(
            analysis,
            column,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for output_name, column in group_columns.items()
    }
    source_gap = bootstrap_source_auc_gap(
        analysis,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    robustness = assess_group_robustness(grouped, source_gap)

    all_errors = enrich_high_confidence_errors(analysis)
    reviewed = all_errors.head(review_cases).copy()
    error_summary = summarize_high_confidence_errors(all_errors)

    validation_comparison, validation_oof = cross_validated_comparison(
        replayed["val"],
        n_splits=n_splits,
    )
    selected_method = select_calibration_method(validation_comparison)
    calibrators = fit_full_calibrators(replayed["val"])
    test_calibration, calibrated_test = evaluate_test_calibrators(
        replayed["test"],
        calibrators,
        selected_method,
    )
    curves = calibration_curves(calibrated_test)
    calibration_protocol = audit_calibration_protocol(
        replayed["val"],
        validation_comparison,
        validation_oof,
        selected_method,
        n_splits=n_splits,
    )
    calibration_assessment = assess_calibration(
        test_calibration,
        selected_method,
    )

    calibrator_path = model_dir / "first_kill_calibrator.joblib"
    joblib.dump(
        {
            "calibrator": calibrators[selected_method],
            "method": selected_method,
            "task": "post_first_kill",
            "base_model_path": model_path.as_posix(),
            "base_model_sha256": prerequisite["model_artifact"]["sha256"],
            "data_sha256": prerequisite["data_artifact"]["sha256"],
            "selection_data": "validation only",
            "validation_folds": n_splits,
        },
        calibrator_path,
    )
    calibrator_artifact = fingerprint_file(calibrator_path)

    m17_comparison = read_table(m17_comparison_path)
    external_benchmarks = read_table(benchmarks_path)
    external = compare_external_models(m17_comparison, external_benchmarks)
    external_report = render_external_report(external)
    external_report_passed = bool(not external.empty and external_report.strip())

    current_metrics = probability_metrics(
        analysis["y_true"],
        analysis["ct_win_probability"],
        n_bins=10,
    )
    m17_metrics = m17_summary.get("metrics", {})
    metric_replay_max_difference = max(
        abs(float(current_metrics[name]) - float(m17_metrics[name]))
        for name in METRIC_ORDER
    )

    automated_tests = run_automated_tests(project_root)
    test_count_match = re.search(r"Ran (\d+) tests?", automated_tests["output"])
    automated_test_count = (
        int(test_count_match.group(1)) if test_count_match else None
    )

    checks = {
        "m17_prerequisite": prerequisite["passed"],
        "split_and_key_contract": split_and_key_contract,
        "prediction_replay": prediction_replay["passed"]
        and metric_replay_max_difference <= 1e-12,
        "global_bootstrap": global_assessment["bootstrap_complete"],
        "global_metric_minimum": global_assessment["minimum_passed"],
        "group_outputs": robustness["outputs_complete"],
        "source_stability": robustness["source_gap_passed"],
        "large_map_minimum": robustness["large_map_minimum_passed"],
        "calibration_protocol": calibration_protocol["passed"],
        "calibration_no_material_harm": calibration_assessment[
            "no_material_probability_metric_harm"
        ],
        "error_review": len(reviewed) >= 30,
        "external_report": external_report_passed,
        "automated_tests": automated_tests["passed"],
    }
    acceptance = decide_acceptance(checks)
    stage_targets = {
        "global_interval_target": global_assessment["stage_passed"],
        "large_map_auc_target": robustness["large_map_stage_passed"],
        "large_map_ci_target": robustness["large_map_ci_stage_passed"],
        "source_gap_ci_includes_zero": robustness[
            "source_gap_ci_includes_zero"
        ],
        "calibration_ece_target": calibration_assessment[
            "test_ece_stage_passed"
        ],
    }
    stage_targets["all_passed"] = all(stage_targets.values())

    split_rows = data_audit["split_rows"]
    split_percentages = {
        name: float(count / len(data) * 100) for name, count in split_rows.items()
    }
    split_series = {
        name: int(data.loc[data["split"].eq(name), "series_id"].nunique())
        for name in ("train", "val", "test")
    }
    summary = {
        "stage": "M18",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "model_policy": "M17 XGBoost frozen; no training or tuning in M18",
        "acceptance": acceptance,
        "checks": checks,
        "stage_targets": stage_targets,
        "prerequisite": prerequisite,
        "data": {
            "path": data_path.as_posix(),
            "sha256": prerequisite["data_artifact"]["sha256"],
            "rows": int(len(data)),
            "series": int(data["series_id"].nunique()),
            "games": int(data["game_id"].nunique()),
            "split_rows": split_rows,
            "split_percentages": split_percentages,
            "split_series": split_series,
            "cross_split_series": data_audit["cross_split_series"],
            "duplicate_key_rows": data_audit["duplicate_key_rows"],
            "key_columns": list(KEY_COLUMNS),
        },
        "model_replay": model_replay_audit,
        "prediction_replay": prediction_replay,
        "metric_replay_max_absolute_difference": metric_replay_max_difference,
        "metrics": {name: float(current_metrics[name]) for name in METRIC_ORDER},
        "bootstrap": {
            "unit": "series_id",
            "samples": n_bootstrap,
            "seed": seed,
        },
        "global_assessment": global_assessment,
        "robustness": robustness,
        "source_auc_gap": source_gap,
        "group_table_rows": {
            name: int(len(table)) for name, table in grouped.items()
        },
        "calibration_protocol": calibration_protocol,
        "calibration": {
            **calibration_assessment,
            "calibrator_artifact": calibrator_artifact,
            "validation_oof_comparison": json.loads(
                validation_comparison.to_json(orient="records")
            ),
            "test_comparison": json.loads(
                test_calibration.to_json(orient="records")
            ),
        },
        "errors": {
            "definition": "wrong and assigned predicted-side probability >= 0.80",
            "available": int(len(all_errors)),
            "reviewed": int(len(reviewed)),
            "review_target": 30,
            "signal_pattern_counts": {
                str(name): int(count)
                for name, count in all_errors["signal_pattern"].value_counts().items()
            },
        },
        "external_comparison_rows": int(len(external)),
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_test_count,
        },
        "next_stage": "M19 first-kill model explanation and leakage audit",
    }

    global_intervals.to_csv(
        report_dir / "global_bootstrap_95ci.csv", index=False
    )
    analysis.to_csv(report_dir / "test_predictions_enriched.csv", index=False)
    for name, table in grouped.items():
        table.to_csv(
            report_dir / f"metrics_by_{name}_with_ci.csv",
            index=False,
        )
    pd.DataFrame([source_gap]).to_csv(
        report_dir / "source_auc_gap.csv", index=False
    )
    validation_comparison.to_csv(
        report_dir / "validation_oof_calibration.csv", index=False
    )
    validation_oof.to_csv(
        report_dir / "validation_oof_predictions.csv", index=False
    )
    test_calibration.to_csv(
        report_dir / "test_calibration_comparison.csv", index=False
    )
    calibrated_test.to_csv(
        report_dir / "calibrated_test_predictions.csv", index=False
    )
    curves.to_csv(report_dir / "calibration_curves.csv", index=False)
    all_errors.to_csv(
        report_dir / "all_high_confidence_errors.csv", index=False
    )
    reviewed.to_csv(report_dir / "reviewed_top30_errors.csv", index=False)
    error_summary.to_csv(report_dir / "error_pattern_summary.csv", index=False)
    write_case_review(reviewed, report_dir / "top30_error_review.md")
    external.to_csv(
        report_dir / "external_benchmark_comparison.csv", index=False
    )
    (report_dir / "external_benchmark_comparison.md").write_text(
        external_report,
        encoding="utf-8",
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests["output"],
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"check": name, "passed": passed, "blocking": True}
            for name, passed in checks.items()
        ]
    ).to_csv(report_dir / "m18_checks.csv", index=False)
    write_json(summary, report_dir / "m18_summary.json")
    (report_dir / "m18_first_kill_evaluation_report.md").write_text(
        render_m18_report(
            summary,
            global_intervals,
            grouped["map"],
            grouped["source"],
            validation_comparison,
            test_calibration,
            external,
        ),
        encoding="utf-8",
    )
    save_plots(
        grouped["map"],
        curves,
        selected_method,
        error_summary,
        report_dir,
    )
    return grouped, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M18 fixed post-first-kill model evaluation."
    )
    parser.add_argument(
        "--data", default="data/processed/esta_full/first_kill.parquet"
    )
    parser.add_argument(
        "--model",
        default="models/esta_full_m17/first_kill_xgboost_tuned.joblib",
    )
    parser.add_argument(
        "--m17-summary", default="reports/esta_full_m17/m17_summary.json"
    )
    parser.add_argument(
        "--m17-predictions",
        default="reports/esta_full_m17/test_predictions.csv",
    )
    parser.add_argument(
        "--m17-comparison",
        default="reports/esta_full_m17/model_comparison.csv",
    )
    parser.add_argument(
        "--benchmarks",
        default="benchmarks/external_first_kill_tuned_metrics.csv",
    )
    parser.add_argument("--model-dir", default="models/esta_full_m18")
    parser.add_argument("--report-dir", default="reports/esta_full_m18")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--review-cases", type=int, default=30)
    args = parser.parse_args()

    grouped, summary = run(
        data_path=args.data,
        model_path=args.model,
        m17_summary_path=args.m17_summary,
        m17_predictions_path=args.m17_predictions,
        m17_comparison_path=args.m17_comparison,
        benchmarks_path=args.benchmarks,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        project_root=args.project_root,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        n_splits=args.folds,
        review_cases=args.review_cases,
    )
    print(
        grouped["source"][
            [
                "source_subset",
                "rounds",
                "series",
                "auc",
                "auc_ci_lower_95",
                "auc_ci_upper_95",
                "log_loss",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(
        f"M18 {summary['acceptance']['status']}; "
        f"ready_for_m19={summary['acceptance']['ready_for_m19']}"
    )
    if not summary["acceptance"]["ready_for_m19"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib
import numpy as np
import pandas as pd

from .benchmark_comparison import compare_benchmarks, write_markdown_report
from .io import read_table
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
    outcome_error_pattern,
    pre_round_error_pattern,
    select_high_confidence_errors,
)
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m18_first_kill_evaluation import (
    assess_calibration,
    bootstrap_source_auc_gap,
    parse_source_subset,
)
from .m22_pre_round_lightgbm_baseline import (
    audit_data_contract,
    prepare_pre_round_splits,
    write_json,
)
from .m9_evaluation import METRIC_ORDER, bootstrap_metric_intervals
from .metrics import probability_metrics
from .schema import ID_COLUMNS, PRE_ROUND_FEATURES


matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY_COLUMNS = tuple(ID_COLUMNS)
HIGHER_IS_BETTER = {"accuracy", "auc"}
BLOCKING_CHECKS = (
    "m23_prerequisite",
    "split_and_key_contract",
    "prediction_replay",
    "global_bootstrap",
    "global_metric_minimum",
    "paired_comparison",
    "group_outputs",
    "source_stability",
    "large_map_minimum",
    "calibration_protocol",
    "calibration_no_material_harm",
    "error_review",
    "external_report",
    "automated_tests",
    "source_compile",
    "reproduction_entrypoint",
)

ANALYSIS_FEATURE_COLUMNS = (
    "map_name",
    "round_num",
    "eq_value_diff_ct",
    "ct_eq_value",
    "t_eq_value",
    "rifle_diff_ct",
    "awp_diff_ct",
)
FIRST_KILL_COLUMNS = (
    "killer_side",
    "victim_side",
    "weapon",
    "headshot",
    "time",
)


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


def _assert_same_key_set(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str,
) -> None:
    left_keys = _key_index(left)
    right_keys = _key_index(right)
    missing_from_right = int((~left_keys.isin(right_keys)).sum())
    missing_from_left = int((~right_keys.isin(left_keys)).sum())
    if missing_from_right or missing_from_left:
        raise ValueError(
            f"{label} do not have the same complete key set: "
            f"missing_from_right={missing_from_right}, "
            f"missing_from_left={missing_from_left}"
        )


def prepare_analysis_table(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    kills: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        predictions,
        set(KEY_COLUMNS) | {"y_true", "ct_win_probability", "predicted_label"},
        "M24 predictions",
    )
    _require_columns(
        features,
        set(KEY_COLUMNS)
        | {"split", "ct_win"}
        | set(ANALYSIS_FEATURE_COLUMNS),
        "M24 data",
    )
    _validate_unique_keys(predictions, "M24 predictions")
    test_features = features.loc[features["split"].eq("test")].copy()
    _validate_unique_keys(test_features, "M24 test data")
    _assert_same_key_set(
        predictions,
        test_features,
        label="M24 predictions and test data",
    )

    prediction_columns = list(KEY_COLUMNS) + [
        "y_true",
        "ct_win_probability",
        "predicted_label",
    ]
    analysis = predictions[prediction_columns].merge(
        test_features[
            list(KEY_COLUMNS) + ["ct_win", *ANALYSIS_FEATURE_COLUMNS]
        ],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    label_mismatches = int(
        analysis["y_true"].astype(int).ne(analysis["ct_win"].astype(int)).sum()
    )
    if label_mismatches:
        raise ValueError(f"M24 joined labels disagree in {label_mismatches} rows")

    probability = analysis["ct_win_probability"].to_numpy(dtype=float)
    invalid = (~np.isfinite(probability)) | (probability < 0) | (probability > 1)
    if invalid.any():
        raise ValueError("M24 prediction probabilities must be finite and in [0, 1]")
    expected_label = (probability >= 0.5).astype(int)
    if not np.array_equal(
        expected_label,
        analysis["predicted_label"].to_numpy(dtype=int),
    ):
        raise ValueError("M24 predicted labels do not match the 0.5 threshold")

    _require_columns(
        kills,
        set(KEY_COLUMNS) | {"is_first_kill", *FIRST_KILL_COLUMNS},
        "M24 kill diagnostics",
    )
    first_kills = kills.loc[
        kills["is_first_kill"].eq(1),
        list(KEY_COLUMNS) + list(FIRST_KILL_COLUMNS),
    ].copy()
    _validate_unique_keys(first_kills, "M24 first-kill diagnostics")
    first_kills = first_kills.rename(
        columns={
            "killer_side": "first_kill_side",
            "victim_side": "first_death_side",
            "weapon": "first_kill_weapon",
            "headshot": "first_kill_headshot",
            "time": "first_kill_time",
        }
    )
    analysis = analysis.merge(
        first_kills,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    analysis["source_subset"] = parse_source_subset(analysis["game_id"])
    analysis["round_stage"] = assign_round_stage(analysis["round_num"])
    analysis["equipment_band"] = assign_equipment_band(
        analysis["eq_value_diff_ct"]
    )
    analysis["t_win_probability"] = 1.0 - probability
    analysis["correct"] = analysis["predicted_label"].eq(analysis["y_true"])
    return analysis


def verify_m23_prerequisite(
    data_path: str | Path,
    model_path: str | Path,
    m23_summary: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    data_artifact = fingerprint_file(data_path)
    model_artifact = fingerprint_file(model_path)
    expected_data_sha = m23_summary.get("data", {}).get("sha256")
    expected_model_sha = (
        m23_summary.get("model", {}).get("model_artifact", {}).get("sha256")
    )
    expected_params = m23_summary.get("model", {}).get("params", {})
    encoded_columns = list(bundle.get("columns", []))
    raw_features = list(bundle.get("raw_features", []))
    checks = {
        "m23_accepted": bool(
            m23_summary.get("acceptance", {}).get("status") == "passed"
            and m23_summary.get("acceptance", {}).get("ready_for_m24") is True
        ),
        "data_sha256": bool(expected_data_sha)
        and data_artifact["sha256"] == expected_data_sha
        and bundle.get("data_sha256") == expected_data_sha,
        "model_sha256": bool(expected_model_sha)
        and model_artifact["sha256"] == expected_model_sha,
        "bundle_task": bundle.get("task") == "pre_round",
        "bundle_model_name": bundle.get("model_name") == "lightgbm_tuned",
        "raw_feature_contract": raw_features == list(PRE_ROUND_FEATURES)
        and len(raw_features)
        == int(m23_summary.get("features", {}).get("raw_count", -1)),
        "encoded_feature_contract": bool(encoded_columns)
        and len(encoded_columns) == len(set(encoded_columns))
        and len(encoded_columns)
        == int(m23_summary.get("features", {}).get("encoded_count", -1)),
        "parameter_contract": bool(expected_params)
        and dict(bundle.get("params", {})) == dict(expected_params),
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


def replay_frozen_model(
    data: pd.DataFrame,
    bundle: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    prepared = prepare_pre_round_splits(data)
    encoded_columns = prepared["train"][0].columns.tolist()
    expected_columns = list(bundle.get("columns", []))
    if encoded_columns != expected_columns:
        raise ValueError("M24 encoded feature columns do not match the M23 model")
    model = bundle.get("model")
    if not hasattr(model, "predict_proba"):
        raise ValueError("M24 frozen model does not provide predict_proba")

    outputs: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        x, y, identity = prepared[split]
        probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        if (
            len(probability) != len(identity)
            or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise ValueError(f"M24 {split} replay produced invalid probabilities")
        metadata = data.loc[
            data["split"].eq(split),
            list(KEY_COLUMNS) + ["map_name", "round_num"],
        ].copy()
        _validate_unique_keys(metadata, f"M24 {split} metadata")
        output = identity[list(KEY_COLUMNS)].copy()
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

    return outputs, {
        "passed": True,
        "raw_feature_count": len(bundle.get("raw_features", [])),
        "encoded_feature_count": len(encoded_columns),
        "encoded_columns_match_m23": encoded_columns == expected_columns,
        "split_rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "lightgbm_fit_calls": 0,
    }


def build_paired_prediction_table(
    replayed: pd.DataFrame,
    saved_m23: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        replayed,
        set(KEY_COLUMNS) | {"y_true", "ct_win_probability"},
        "M24 replayed predictions",
    )
    _require_columns(
        saved_m23,
        set(KEY_COLUMNS) | {"ct_win", "xgboost_frozen_probability"},
        "M23 comparison predictions",
    )
    _validate_unique_keys(replayed, "M24 replayed predictions")
    _validate_unique_keys(saved_m23, "M23 comparison predictions")
    _assert_same_key_set(
        replayed,
        saved_m23,
        label="M24 replayed and M23 comparison predictions",
    )
    paired = replayed[
        list(KEY_COLUMNS) + ["y_true", "ct_win_probability"]
    ].merge(
        saved_m23[
            list(KEY_COLUMNS) + ["ct_win", "xgboost_frozen_probability"]
        ],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    label_mismatch_count = int(
        paired["y_true"].astype(int).ne(paired["ct_win"].astype(int)).sum()
    )
    if label_mismatch_count:
        raise ValueError(
            f"M24 paired model labels disagree in {label_mismatch_count} rows"
        )
    return paired.rename(
        columns={
            "ct_win_probability": "lightgbm_probability",
            "xgboost_frozen_probability": "xgboost_probability",
        }
    ).drop(columns=["ct_win"])


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
    errors["pre_round_pattern"] = errors.apply(
        pre_round_error_pattern,
        axis=1,
    )
    errors["outcome_pattern"] = errors.apply(outcome_error_pattern, axis=1)
    errors["predicted_side_won_first_kill"] = (
        errors["predicted_side"].eq(errors["first_kill_side"])
    ).where(errors["first_kill_side"].notna())
    return errors


def summarize_high_confidence_errors(errors: pd.DataFrame) -> pd.DataFrame:
    dimensions = (
        "pre_round_pattern",
        "outcome_pattern",
        "predicted_side",
        "first_kill_side",
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


def write_case_review(cases: pd.DataFrame, path: Path) -> None:
    lines = [
        "# M24 开局前 LightGBM 高置信错误复核",
        "",
        "定义：预测错误且预测方概率不低于 0.80。首杀字段只用于预测后的事后诊断。",
        "以下模式是描述性结果，不是因果结论。",
        "",
    ]
    for index, row in cases.reset_index(drop=True).iterrows():
        first_kill = (
            str(row["first_kill_side"])
            if pd.notna(row["first_kill_side"])
            else "none"
        )
        lines.append(
            f"{index + 1}. `{row['series_id']} / {row['game_id']} / "
            f"{row['round_id']}`：预测 {row['predicted_side']} "
            f"({row['assigned_side_probability']:.3f})，实际 "
            f"{row['actual_winner']}；装备差 CT "
            f"{row['eq_value_diff_ct']:+.0f}，首杀 {first_kill}；"
            f"`{row['pre_round_pattern']}` / `{row['outcome_pattern']}`。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def run_compile_check(project_root: str | Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "compileall", "src", "tests"]
    completed = subprocess.run(
        command,
        cwd=Path(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "command": command,
        "output": completed.stdout + completed.stderr,
    }


def _collect_git_state(project_root: Path, report_dir: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip()

    status = [line for line in git("status", "--short").splitlines() if line]
    ignored_report = []
    retained_status = []
    try:
        report_prefix = report_dir.relative_to(project_root).as_posix()
    except ValueError:
        report_prefix = None
    for line in status:
        if report_prefix and report_prefix in line.replace("\\", "/"):
            ignored_report.append(line)
        else:
            retained_status.append(line)
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "remote": git("remote", "get-url", "origin"),
        "working_tree_clean_before_report_generation": not retained_status,
        "working_tree_status_before_report_generation": retained_status,
        "ignored_report_status": ignored_report,
    }


def audit_prediction_replay(
    saved: pd.DataFrame,
    replayed: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    _require_columns(
        saved,
        set(KEY_COLUMNS) | {"ct_win", "lightgbm_tuned_probability"},
        "M23 saved predictions",
    )
    _require_columns(
        replayed,
        set(KEY_COLUMNS) | {"y_true", "ct_win_probability"},
        "M24 replayed predictions",
    )
    saved_duplicate_rows = int(saved.duplicated(list(KEY_COLUMNS)).sum())
    replayed_duplicate_rows = int(replayed.duplicated(list(KEY_COLUMNS)).sum())
    saved_keys = _key_index(saved)
    replayed_keys = _key_index(replayed)
    key_mismatch_count = int(
        (~saved_keys.isin(replayed_keys)).sum()
        + (~replayed_keys.isin(saved_keys)).sum()
    )

    label_mismatch_count = 0
    invalid_probability_cells = 0
    max_difference: float | None = None
    if not saved_duplicate_rows and not replayed_duplicate_rows and not key_mismatch_count:
        joined = saved[
            list(KEY_COLUMNS) + ["ct_win", "lightgbm_tuned_probability"]
        ].merge(
            replayed[
                list(KEY_COLUMNS) + ["y_true", "ct_win_probability"]
            ],
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        label_mismatch_count = int(
            joined["ct_win"].astype(int).ne(joined["y_true"].astype(int)).sum()
        )
        values = joined[
            ["lightgbm_tuned_probability", "ct_win_probability"]
        ].to_numpy(dtype=float)
        invalid_probability_cells = int(
            (~np.isfinite(values)).sum()
            + ((values < 0) | (values > 1)).sum()
        )
        if not invalid_probability_cells:
            max_difference = float(
                np.max(np.abs(values[:, 0] - values[:, 1]), initial=0.0)
            )

    passed = bool(
        not saved_duplicate_rows
        and not replayed_duplicate_rows
        and not key_mismatch_count
        and not label_mismatch_count
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
        "label_mismatch_count": label_mismatch_count,
        "invalid_probability_cells": invalid_probability_cells,
        "max_absolute_probability_difference": max_difference,
    }


def paired_model_bootstrap(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    required = set(KEY_COLUMNS) | {
        "y_true",
        "lightgbm_probability",
        "xgboost_probability",
    }
    _require_columns(predictions, required, "M24 paired predictions")
    _validate_unique_keys(predictions, "M24 paired predictions")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    y = predictions["y_true"].to_numpy(dtype=int)
    lightgbm = predictions["lightgbm_probability"].to_numpy(dtype=float)
    xgboost = predictions["xgboost_probability"].to_numpy(dtype=float)
    probability = np.column_stack([lightgbm, xgboost])
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("M24 paired probabilities must be finite and in [0, 1]")

    series = predictions["series_id"].astype(str).to_numpy()
    unique_series = np.unique(series)
    positions = [np.flatnonzero(series == value) for value in unique_series]
    point_lightgbm = probability_metrics(y, lightgbm, n_bins=10)
    point_xgboost = probability_metrics(y, xgboost, n_bins=10)
    samples: dict[str, list[float]] = {metric: [] for metric in METRIC_ORDER}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(positions), size=len(positions))
        sampled = np.concatenate([positions[index] for index in chosen])
        current_lightgbm = probability_metrics(
            y[sampled], lightgbm[sampled], n_bins=10
        )
        current_xgboost = probability_metrics(
            y[sampled], xgboost[sampled], n_bins=10
        )
        for metric in METRIC_ORDER:
            raw_difference = (
                current_lightgbm[metric] - current_xgboost[metric]
            )
            performance_advantage = (
                raw_difference
                if metric in HIGHER_IS_BETTER
                else -raw_difference
            )
            if np.isfinite(performance_advantage):
                samples[metric].append(float(performance_advantage))

    rows = []
    for metric in METRIC_ORDER:
        values = np.asarray(samples[metric], dtype=float)
        if values.size == 0:
            raise RuntimeError(
                f"M24 paired bootstrap produced no valid values for {metric}"
            )
        raw_difference = point_lightgbm[metric] - point_xgboost[metric]
        performance_advantage = (
            raw_difference if metric in HIGHER_IS_BETTER else -raw_difference
        )
        lower = float(np.quantile(values, 0.025))
        upper = float(np.quantile(values, 0.975))
        rows.append(
            {
                "metric": metric,
                "lightgbm": float(point_lightgbm[metric]),
                "xgboost": float(point_xgboost[metric]),
                "raw_difference_lightgbm_minus_xgboost": float(raw_difference),
                "performance_advantage_lightgbm": float(performance_advantage),
                "performance_advantage_ci_lower_95": lower,
                "performance_advantage_ci_upper_95": upper,
                "ci_includes_zero": bool(lower <= 0 <= upper),
                "lightgbm_significantly_better": bool(lower > 0),
                "successful_bootstraps": int(values.size),
                "bootstrap_unit": "series_id_paired",
            }
        )
    return pd.DataFrame(rows)


def assess_global_intervals(
    intervals: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> dict[str, Any]:
    required = {
        "metric",
        "ci_lower_95",
        "ci_upper_95",
        "successful_bootstraps",
    }
    _require_columns(intervals, required, "M24 global intervals")
    indexed = intervals.set_index("metric")
    metrics_complete = set(METRIC_ORDER).issubset(indexed.index)
    bootstrap_complete = bool(
        metrics_complete
        and indexed.loc[list(METRIC_ORDER), "successful_bootstraps"]
        .eq(n_bootstrap)
        .all()
    )
    if not metrics_complete:
        raise ValueError("M24 global intervals do not contain all five metrics")
    auc_lower = float(indexed.loc["auc", "ci_lower_95"])
    log_loss_upper = float(indexed.loc["log_loss", "ci_upper_95"])
    return {
        "bootstrap_complete": bootstrap_complete,
        "auc_ci_lower_95": auc_lower,
        "log_loss_ci_upper_95": log_loss_upper,
        "minimum_passed": bool(auc_lower >= 0.700 and log_loss_upper <= 0.610),
        "stage_passed": bool(auc_lower >= 0.710 and log_loss_upper <= 0.605),
        "minimum_thresholds": {
            "auc_ci_lower_95": 0.700,
            "log_loss_ci_upper_95": 0.610,
        },
        "stage_thresholds": {
            "auc_ci_lower_95": 0.710,
            "log_loss_ci_upper_95": 0.605,
        },
    }


def assess_group_robustness(
    grouped: Mapping[str, pd.DataFrame],
    source_gap: Mapping[str, Any],
) -> dict[str, Any]:
    expected_groups = {"map", "source", "round_stage", "equipment_band"}
    outputs_complete = set(grouped) == expected_groups and all(
        not table.empty for table in grouped.values()
    )
    map_table = grouped.get("map")
    if map_table is None or map_table.empty:
        return {
            "outputs_complete": False,
            "large_map_count": 0,
            "large_map_min_auc": None,
            "large_map_min_auc_ci_lower": None,
            "large_map_minimum_passed": False,
            "large_map_stage_passed": False,
            "large_map_ci_stage_passed": False,
            "source_gap_passed": False,
            "source_gap_ci_includes_zero": False,
        }
    _require_columns(
        map_table,
        {"rounds", "auc", "auc_ci_lower_95"},
        "M24 map metrics",
    )
    large_maps = map_table.loc[map_table["rounds"].ge(300)].copy()
    if large_maps.empty:
        raise RuntimeError("M24 found no map with at least 300 test rounds")
    minimum_auc = float(large_maps["auc"].min())
    minimum_auc_ci_lower = float(large_maps["auc_ci_lower_95"].min())
    source_difference = float(source_gap.get("absolute_difference", np.inf))
    return {
        "outputs_complete": outputs_complete,
        "large_map_count": int(len(large_maps)),
        "large_map_min_auc": minimum_auc,
        "large_map_min_auc_ci_lower": minimum_auc_ci_lower,
        "large_map_minimum_passed": bool(minimum_auc >= 0.670),
        "large_map_stage_passed": bool(minimum_auc >= 0.690),
        "large_map_ci_stage_passed": bool(minimum_auc_ci_lower >= 0.670),
        "source_gap_passed": bool(source_difference <= 0.040),
        "source_gap_ci_includes_zero": bool(
            source_gap.get("ci_includes_zero", False)
        ),
    }


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
    _require_columns(
        validation_predictions,
        set(KEY_COLUMNS) | {"series_id"},
        "M24 validation predictions",
    )
    _require_columns(
        validation_oof,
        set(KEY_COLUMNS) | {"fold"},
        "M24 validation OOF predictions",
    )
    unique_keys = not validation_oof.duplicated(list(KEY_COLUMNS)).any()
    exact_rows = len(validation_oof) == len(validation_predictions)
    key_sets_match = False
    if unique_keys and not validation_predictions.duplicated(list(KEY_COLUMNS)).any():
        left = _key_index(validation_predictions)
        right = _key_index(validation_oof)
        key_sets_match = bool(left.isin(right).all() and right.isin(left).all())
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
    forbidden_columns = sorted(
        {
            column
            for table in (validation_comparison, validation_oof)
            for column in table.columns
            if str(column).lower().startswith("test_")
        }
    )
    checks = {
        "exact_rows": exact_rows,
        "unique_complete_keys": unique_keys,
        "key_sets_match": key_sets_match,
        "series_one_fold": series_one_fold,
        "fold_count": fold_count == n_splits,
        "probability_columns": probability_columns_present,
        "finite_probabilities": finite_probabilities,
        "methods_complete": methods_complete,
        "selected_from_oof": selected_from_oof,
        "no_test_columns": not forbidden_columns,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "forbidden_columns": forbidden_columns,
        "selection_data": "validation only",
        "selection_rule": (
            "lowest grouped OOF log_loss, then brier_score, then method"
        ),
        "validation_folds": n_splits,
        "selected_method": selected_method,
    }


def decide_acceptance(checks: Mapping[str, bool]) -> dict[str, Any]:
    failures = [
        name for name in BLOCKING_CHECKS if not bool(checks.get(name, False))
    ]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "m24_lightgbm_evaluation_complete": not failures,
        "ready_for_m25": not failures,
    }


def save_plots(
    map_metrics: pd.DataFrame,
    calibration_curve: pd.DataFrame,
    selected_method: str,
    paired: pd.DataFrame,
    error_summary: pd.DataFrame,
    report_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    maps = map_metrics.sort_values("auc").copy()
    maps["display_name"] = maps.apply(
        lambda row: (
            f"{row['map_name']} (n={int(row['rounds'])}, "
            f"series={int(row['series'])})"
        ),
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
    ax.axvline(0.670, color="#C44E52", linestyle="--", linewidth=1, label="Minimum 0.670")
    ax.axvline(0.690, color="#2A9D8F", linestyle=":", linewidth=1.5, label="Target 0.690")
    ax.set(
        xlabel="Test AUC with series-level 95% CI",
        ylabel="Map",
        title="M24 pre-round LightGBM map robustness",
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
        title="M24 test reliability comparison",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(report_dir / "reliability_comparison.png", dpi=160)
    plt.close(fig)

    ordered = paired.set_index("metric").loc[list(METRIC_ORDER)].reset_index()
    lower = np.maximum(
        ordered["performance_advantage_lightgbm"]
        - ordered["performance_advantage_ci_lower_95"],
        0,
    )
    upper = np.maximum(
        ordered["performance_advantage_ci_upper_95"]
        - ordered["performance_advantage_lightgbm"],
        0,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(
        ordered["performance_advantage_lightgbm"],
        ordered["metric"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#176B87",
        ecolor="#6B7280",
        capsize=3,
    )
    ax.axvline(0, color="#C44E52", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Performance advantage of LightGBM with paired 95% CI",
        ylabel="Metric",
        title="M24 LightGBM vs XGBoost paired comparison",
    )
    fig.tight_layout()
    fig.savefig(report_dir / "paired_model_advantage_95ci.png", dpi=160)
    plt.close(fig)

    patterns = error_summary.loc[
        error_summary["dimension"].eq("outcome_pattern")
    ].sort_values("cases")
    if not patterns.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.barh(
            patterns["value"].astype(str).str.replace("_", " "),
            patterns["cases"],
            color="#C44E52",
        )
        ax.set(
            xlabel="High-confidence wrong rounds",
            ylabel="Post-hoc outcome pattern",
            title="M24 high-confidence error outcomes",
        )
        fig.tight_layout()
        fig.savefig(report_dir / "error_pattern_counts.png", dpi=160)
        plt.close(fig)


def render_m24_report(
    summary: Mapping[str, Any],
    global_intervals: pd.DataFrame,
    paired: pd.DataFrame,
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
        "# M24 开局前 LightGBM 固定模型评估报告",
        "",
        "## 阶段结论",
        "",
        f"阻断验收：**{acceptance['blocking_passed']}/{acceptance['blocking_total']}**；"
        f"状态 **{acceptance['status']}**；可进入 M25："
        f"**{acceptance['ready_for_m25']}**。",
        "M24 没有训练或调参 LightGBM，只回放 M23 冻结模型并评估。",
        f"测试概率回放最大绝对误差为 "
        f"`{summary['prediction_replay']['max_absolute_probability_difference']:.3e}`。",
        "",
        "## 70/20/10 与主键",
        "",
        f"总样本 {summary['data']['rows']:,}，系列赛 {summary['data']['series']:,}；"
        f"train/validation/test 行数为 "
        f"{summary['data']['split_rows']['train']:,}/"
        f"{summary['data']['split_rows']['val']:,}/"
        f"{summary['data']['split_rows']['test']:,}。",
        f"系列赛数为 {summary['data']['split_series']['train']}/"
        f"{summary['data']['split_series']['val']}/"
        f"{summary['data']['split_series']['test']}；跨 split 系列赛为 "
        f"{summary['data']['cross_split_series']}。",
        f"完整主键为 `{'+'.join(KEY_COLUMNS)}`。",
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
            f"AUC CI 下界 `{global_assessment['auc_ci_lower_95']:.6f}`；"
            f"Log Loss CI 上界 `{global_assessment['log_loss_ci_upper_95']:.6f}`。",
            f"最低区间目标：`{global_assessment['minimum_passed']}`；"
            f"更高阶段目标：`{global_assessment['stage_passed']}`。",
            "",
            "## LightGBM 与 XGBoost 配对差值",
            "",
            "性能优势已统一方向，正数表示 LightGBM 更好。",
            "",
            "| 指标 | LightGBM | XGBoost | 原始差值 | 性能优势 95% CI | 显著更好 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in paired.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['lightgbm']:.6f} | "
            f"{row['xgboost']:.6f} | "
            f"{row['raw_difference_lightgbm_minus_xgboost']:+.6f} | "
            f"[{row['performance_advantage_ci_lower_95']:+.6f}, "
            f"{row['performance_advantage_ci_upper_95']:+.6f}] | "
            f"{row['lightgbm_significantly_better']} |"
        )
    lines.extend(
        [
            "",
            f"五项中性能优势区间排除 0 的数量为 "
            f"`{summary['paired_comparison']['significant_better_count']}/5`。"
            "未排除 0 不等于两模型相同，只表示当前测试系列不足以证明稳定优势。",
            "",
            "## 地图与来源",
            "",
            "| 地图 | 回合 | 系列 | AUC | AUC 95% CI | Log Loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in map_metrics.to_dict(orient="records"):
        lines.append(
            f"| {row['map_name']} | {int(row['rounds'])} | "
            f"{int(row['series'])} | {row['auc']:.6f} | "
            f"[{row['auc_ci_lower_95']:.6f}, {row['auc_ci_upper_95']:.6f}] | "
            f"{row['log_loss']:.6f} |"
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
            f"LAN-online AUC 点差 `{source_gap['signed_difference']:+.6f}`，"
            f"95% CI [{source_gap['ci_lower_95']:.6f}, "
            f"{source_gap['ci_upper_95']:.6f}]。",
            f"主要地图最低 AUC `{robustness['large_map_min_auc']:.6f}`，"
            f"最低 CI 下界 `{robustness['large_map_min_auc_ci_lower']:.6f}`。",
            "",
            "## 校准",
            "",
            f"只根据 validation 的 5 折 GroupKFold OOF 选择 "
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
            "## 高置信错误",
            "",
            f"共有 {summary['errors']['available']} 个概率至少 0.80 的错误回合，"
            f"已复核前 {summary['errors']['reviewed']} 个。",
            "首杀只作为结果发生后的诊断，不属于模型输入。",
            "",
            "## 与公开结果差多少",
            "",
            "| 可比性 | 外部工作 | 指标 | 当前 | 外部 | 当前减外部 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in external.to_dict(orient="records"):
        difference = float(row["raw_difference_ours_minus_reported"])
        difference_text = (
            f"{difference * 100:+.2f} 个百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| {row.get('comparability', '')} | "
            f"{row.get('source_title', row['benchmark_id'])} | "
            f"{row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {difference_text} |"
        )
    lines.extend(
        [
            "",
            "外部结果使用不同数据、切分和预测时点，只能作为参照。",
            "",
            "## 下一阶段",
            "",
            "M25 将对冻结 LightGBM 做 gain、Permutation Importance、SHAP、"
            "泄漏审计和与 XGBoost 的解释差异分析。",
            "",
            "复现命令：",
            "",
            "```powershell",
            ".\\scripts\\run_pre_round_lightgbm_evaluation.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    project_root: str | Path,
    data_path: str | Path = "data/processed/esta_full/pre_round.parquet",
    kills_path: str | Path = "data/interim/esta_full/kills.parquet",
    model_path: str | Path = "models/esta_full_m23/pre_round_lightgbm_tuned.joblib",
    m23_summary_path: str | Path = "reports/esta_full_m23/m23_summary.json",
    m23_predictions_path: str | Path = "reports/esta_full_m23/test_predictions.csv",
    benchmarks_path: str | Path = "benchmarks/external_round_model_metrics.csv",
    model_dir: str | Path = "models/esta_full_m24",
    report_dir: str | Path = "reports/esta_full_m24",
    n_bootstrap: int = 2000,
    seed: int = 42,
    n_splits: int = 5,
    review_cases: int = 30,
    run_tests: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_path = _resolve(root, data_path)
    kills_path = _resolve(root, kills_path)
    model_path = _resolve(root, model_path)
    m23_summary_path = _resolve(root, m23_summary_path)
    m23_predictions_path = _resolve(root, m23_predictions_path)
    benchmarks_path = _resolve(root, benchmarks_path)
    model_dir = _resolve(root, model_dir)
    report_dir = _resolve(root, report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m23_summary = _read_json(m23_summary_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("M24 expected the M23 model artifact to contain a bundle")
    prerequisite = verify_m23_prerequisite(
        data_path,
        model_path,
        m23_summary,
        bundle,
    )
    if not prerequisite["passed"]:
        raise RuntimeError("M24 input does not match the accepted M23 artifacts")

    data = read_table(data_path)
    data_audit = audit_data_contract(data)
    expected_rows = {
        name: int(value)
        for name, value in m23_summary.get("data", {})
        .get("split_rows", {})
        .items()
    }
    expected_series = {
        name: int(value)
        for name, value in m23_summary.get("data", {})
        .get("split_series", {})
        .items()
    }
    split_and_key_contract = bool(
        data_audit["passed"]
        and data_audit["split_rows"] == expected_rows
        and data_audit["split_series"] == expected_series
    )
    if not split_and_key_contract:
        raise RuntimeError("M24 data identity or split membership differs from M23")

    replayed, model_replay = replay_frozen_model(data, bundle)
    saved_predictions = read_table(m23_predictions_path)
    prediction_replay = audit_prediction_replay(
        saved_predictions,
        replayed["test"],
        tolerance=1e-12,
    )
    if not prediction_replay["passed"]:
        raise RuntimeError("M24 could not exactly replay the M23 test probabilities")

    kills = read_table(kills_path)
    analysis = prepare_analysis_table(replayed["test"], data, kills)
    global_intervals = bootstrap_metric_intervals(
        analysis,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    global_assessment = assess_global_intervals(
        global_intervals,
        n_bootstrap=n_bootstrap,
    )

    paired_predictions = build_paired_prediction_table(
        replayed["test"],
        saved_predictions,
    )
    paired = paired_model_bootstrap(
        paired_predictions,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    paired_complete = bool(
        set(paired["metric"]) == set(METRIC_ORDER)
        and paired["successful_bootstraps"].eq(n_bootstrap).all()
    )

    group_columns = {
        "map": "map_name",
        "source": "source_subset",
        "round_stage": "round_stage",
        "equipment_band": "equipment_band",
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

    calibrator_path = model_dir / "pre_round_lightgbm_calibrator.joblib"
    joblib.dump(
        {
            "calibrator": calibrators[selected_method],
            "method": selected_method,
            "task": "pre_round",
            "base_model_path": model_path.as_posix(),
            "base_model_sha256": prerequisite["model_artifact"]["sha256"],
            "data_sha256": prerequisite["data_artifact"]["sha256"],
            "selection_data": "validation only",
            "validation_folds": n_splits,
        },
        calibrator_path,
    )
    calibrator_artifact = fingerprint_file(calibrator_path)

    current_metrics = probability_metrics(
        analysis["y_true"],
        analysis["ct_win_probability"],
        n_bins=10,
    )
    m23_metrics = m23_summary.get("metrics", {})
    metric_replay_max_difference = max(
        abs(float(current_metrics[name]) - float(m23_metrics[name]))
        for name in METRIC_ORDER
    )
    external_benchmarks = read_table(benchmarks_path)
    external = compare_benchmarks(current_metrics, external_benchmarks)
    external_report_path = report_dir / "external_benchmark_comparison.md"
    write_markdown_report(
        external,
        current_metrics,
        external_report_path,
        stage_label="M24 LightGBM",
    )
    external_report_passed = bool(
        not external.empty and external_report_path.is_file()
    )

    if run_tests:
        automated = run_automated_tests(root)
        match = re.search(r"Ran (\d+) tests?", automated["output"])
        automated_test_count = int(match.group(1)) if match else None
    else:
        automated = {
            "passed": True,
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "command": [],
            "output": "Skipped by caller; exercised separately.\n",
        }
        automated_test_count = None
    compile_check = run_compile_check(root)
    script_path = root / "scripts/run_pre_round_lightgbm_evaluation.ps1"

    checks = {
        "m23_prerequisite": prerequisite["passed"],
        "split_and_key_contract": split_and_key_contract,
        "prediction_replay": prediction_replay["passed"]
        and metric_replay_max_difference <= 1e-12,
        "global_bootstrap": global_assessment["bootstrap_complete"],
        "global_metric_minimum": global_assessment["minimum_passed"],
        "paired_comparison": paired_complete,
        "group_outputs": robustness["outputs_complete"],
        "source_stability": robustness["source_gap_passed"],
        "large_map_minimum": robustness["large_map_minimum_passed"],
        "calibration_protocol": calibration_protocol["passed"],
        "calibration_no_material_harm": calibration_assessment[
            "no_material_probability_metric_harm"
        ],
        "error_review": len(reviewed) >= review_cases,
        "external_report": external_report_passed,
        "automated_tests": automated["passed"],
        "source_compile": compile_check["passed"],
        "reproduction_entrypoint": script_path.is_file(),
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
    significant_metrics = paired.loc[
        paired["lightgbm_significantly_better"], "metric"
    ].tolist()
    summary = {
        "stage": "M24",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "model_policy": "M23 LightGBM frozen; no LightGBM training or tuning in M24",
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
            "split_series": data_audit["split_series"],
            "cross_split_series": data_audit["cross_split_series"],
            "cross_split_games": data_audit["cross_split_games"],
            "cross_split_rounds": data_audit["cross_split_rounds"],
            "duplicate_key_rows": data_audit["duplicate_key_rows"],
            "key_columns": list(KEY_COLUMNS),
        },
        "model_replay": model_replay,
        "prediction_replay": prediction_replay,
        "metric_replay_max_absolute_difference": metric_replay_max_difference,
        "metrics": {name: float(current_metrics[name]) for name in METRIC_ORDER},
        "bootstrap": {
            "unit": "series_id",
            "samples": n_bootstrap,
            "seed": seed,
        },
        "global_assessment": global_assessment,
        "paired_comparison": {
            "bootstrap_complete": paired_complete,
            "bootstrap_unit": "series_id_paired",
            "significant_better_count": len(significant_metrics),
            "significant_better_metrics": significant_metrics,
            "superiority_required_for_acceptance": False,
        },
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
            "review_target": review_cases,
            "first_kill_usage": "post-hoc diagnosis only; never a pre-round feature",
            "pre_round_pattern_counts": {
                str(name): int(count)
                for name, count in all_errors["pre_round_pattern"].value_counts().items()
            },
            "outcome_pattern_counts": {
                str(name): int(count)
                for name, count in all_errors["outcome_pattern"].value_counts().items()
            },
        },
        "external_comparison_rows": int(len(external)),
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "lightgbm_version": importlib.metadata.version("lightgbm"),
            "device_type": bundle["model"].get_params().get("device_type"),
            "cuda_required": False,
        },
        "automated_tests": {
            "passed": automated["passed"],
            "return_code": automated["return_code"],
            "elapsed_seconds": automated["elapsed_seconds"],
            "test_count": automated_test_count,
            "skipped": not run_tests,
        },
        "source_compile": {
            "passed": compile_check["passed"],
            "return_code": compile_check["return_code"],
        },
        "next_stage": "M25 LightGBM explanation and leakage audit",
    }

    global_intervals.to_csv(report_dir / "global_bootstrap_95ci.csv", index=False)
    paired.to_csv(
        report_dir / "paired_lightgbm_vs_xgboost_bootstrap.csv",
        index=False,
    )
    analysis.to_csv(report_dir / "test_predictions_enriched.csv", index=False)
    for name, table in grouped.items():
        table.to_csv(report_dir / f"metrics_by_{name}_with_ci.csv", index=False)
    pd.DataFrame([source_gap]).to_csv(
        report_dir / "source_auc_gap.csv",
        index=False,
    )
    validation_comparison.to_csv(
        report_dir / "validation_oof_calibration.csv",
        index=False,
    )
    validation_oof.to_csv(
        report_dir / "validation_oof_predictions.csv",
        index=False,
    )
    test_calibration.to_csv(
        report_dir / "test_calibration_comparison.csv",
        index=False,
    )
    calibrated_test.to_csv(
        report_dir / "calibrated_test_predictions.csv",
        index=False,
    )
    curves.to_csv(report_dir / "calibration_curves.csv", index=False)
    all_errors.to_csv(
        report_dir / "all_high_confidence_errors.csv",
        index=False,
    )
    reviewed.to_csv(report_dir / "reviewed_top30_errors.csv", index=False)
    error_summary.to_csv(report_dir / "error_pattern_summary.csv", index=False)
    write_case_review(reviewed, report_dir / "top30_error_review.md")
    external.to_csv(
        report_dir / "external_benchmark_comparison.csv",
        index=False,
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated["output"],
        encoding="utf-8",
    )
    (report_dir / "source_compile_output.txt").write_text(
        compile_check["output"],
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"check": name, "passed": passed, "blocking": True}
            for name, passed in checks.items()
        ]
    ).to_csv(report_dir / "m24_checks.csv", index=False)
    write_json(summary, report_dir / "m24_summary.json")
    (report_dir / "m24_pre_round_lightgbm_evaluation_report.md").write_text(
        render_m24_report(
            summary,
            global_intervals,
            paired,
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
        paired,
        error_summary,
        report_dir,
    )

    manifest = {
        "stage": "M24",
        "generated_at_utc": summary["generated_at_utc"],
        "code": _collect_git_state(root, report_dir),
        "input_artifacts": {
            "pre_round_data": prerequisite["data_artifact"],
            "kills": fingerprint_file(kills_path),
            "m23_model": prerequisite["model_artifact"],
            "m23_summary": fingerprint_file(m23_summary_path),
            "m23_test_predictions": fingerprint_file(m23_predictions_path),
            "external_benchmarks": fingerprint_file(benchmarks_path),
            "requirements_lock": fingerprint_file(root / "requirements-lock.txt"),
        },
        "output_artifacts": {
            "calibrator": calibrator_artifact,
        },
        "contract": {
            "model_frozen": True,
            "lightgbm_fit_calls": 0,
            "key_columns": list(KEY_COLUMNS),
            "bootstrap_unit": "series_id",
            "bootstrap_samples": n_bootstrap,
            "paired_model_bootstrap": True,
            "calibration_selection_data": "validation only",
            "calibration_folds": n_splits,
            "first_kill_usage": "post-hoc error diagnosis only",
        },
        "checks": checks,
        "acceptance": acceptance,
    }
    write_json(manifest, report_dir / "m24_experiment_manifest.json")

    if acceptance["status"] != "passed":
        raise RuntimeError(
            "M24 acceptance failed: "
            + ", ".join(acceptance["blocking_failures"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M24 frozen pre-round LightGBM evaluation."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--data",
        default="data/processed/esta_full/pre_round.parquet",
    )
    parser.add_argument(
        "--kills",
        default="data/interim/esta_full/kills.parquet",
    )
    parser.add_argument(
        "--model",
        default="models/esta_full_m23/pre_round_lightgbm_tuned.joblib",
    )
    parser.add_argument(
        "--m23-summary",
        default="reports/esta_full_m23/m23_summary.json",
    )
    parser.add_argument(
        "--m23-predictions",
        default="reports/esta_full_m23/test_predictions.csv",
    )
    parser.add_argument(
        "--benchmarks",
        default="benchmarks/external_round_model_metrics.csv",
    )
    parser.add_argument("--model-dir", default="models/esta_full_m24")
    parser.add_argument("--report-dir", default="reports/esta_full_m24")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--review-cases", type=int, default=30)
    args = parser.parse_args()

    summary = run(
        project_root=args.project_root,
        data_path=args.data,
        kills_path=args.kills,
        model_path=args.model,
        m23_summary_path=args.m23_summary,
        m23_predictions_path=args.m23_predictions,
        benchmarks_path=args.benchmarks,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        n_splits=args.folds,
        review_cases=args.review_cases,
    )
    print(
        f"M24 {summary['acceptance']['status']}; "
        f"ready_for_m25={summary['acceptance']['ready_for_m25']}"
    )


if __name__ == "__main__":
    main()

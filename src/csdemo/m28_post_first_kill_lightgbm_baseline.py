from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .benchmark_comparison import compare_benchmarks, write_markdown_report
from .config import LABEL_COL
from .io import read_table
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m16_first_kill_baselines import (
    build_feature_contract as build_m16_feature_contract,
    canonical_feature_names,
    prepare_profile_splits,
)
from .m22_pre_round_lightgbm_baseline import (
    build_prediction_table,
    evaluate_model,
    fit_lightgbm,
    make_lightgbm_model,
    model_metric_differences,
)
from .m24_pre_round_lightgbm_evaluation import (
    paired_model_bootstrap,
    run_compile_check,
)
from .m9_evaluation import bootstrap_metric_intervals
from .metrics import probability_metrics
from .schema import ID_COLUMNS
from .train_lgbm import EARLY_STOPPING_ROUNDS, LIGHTGBM_BASE_PARAMS


SPLIT_ORDER = ("train", "val", "test")
MODEL_NAMES = ("xgboost_frozen", "lightgbm_baseline")
REPORT_METRICS = ("accuracy", "auc", "log_loss", "brier_score", "ece10")
HIGHER_IS_BETTER = {"accuracy", "auc"}
METRIC_TARGETS = {
    "accuracy": {"minimum": 0.68, "stage": 0.70, "higher_is_better": True},
    "auc": {"minimum": 0.75, "stage": 0.78, "higher_is_better": True},
    "log_loss": {"minimum": 0.58, "stage": 0.55, "higher_is_better": False},
    "brier_score": {"minimum": 0.20, "stage": 0.185, "higher_is_better": False},
    "ece10": {"minimum": 0.05, "stage": 0.03, "higher_is_better": False},
}
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
XGBOOST_REPLAY_TOLERANCE = 1e-7
EXPECTED_DATA_SHA256 = (
    "06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492"
)
EXPECTED_XGBOOST_SHA256 = (
    "ecfaaf93031e78207f81ab5ad9674020657018c0601953238ee6b68e367e8279"
)
EXPECTED_SPLIT_ROWS = {"train": 28489, "val": 8368, "test": 4170}
EXPECTED_SPLIT_SERIES = {"train": 547, "val": 156, "test": 79}
BLOCKING_CHECKS = (
    "m21_prerequisite",
    "data_contract",
    "feature_contract",
    "lightgbm_environment",
    "validation_only_training",
    "frozen_xgboost_replay",
    "probability_contract",
    "minimum_metrics",
    "controlled_comparison",
    "global_uncertainty",
    "paired_uncertainty",
    "external_report",
    "automated_tests",
    "source_compile",
    "reproduction_entrypoint",
    "artifact_manifest",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def build_feature_contract() -> pd.DataFrame:
    return build_m16_feature_contract()


def audit_data_contract(frame: pd.DataFrame) -> dict[str, Any]:
    raw_features = canonical_feature_names()
    required = set(ID_COLUMNS + [LABEL_COL, "split"] + raw_features)
    missing = sorted(required - set(frame.columns))
    if missing:
        return {
            "passed": False,
            "missing_columns": missing,
            "rows": int(len(frame)),
            "duplicate_key_rows": 0,
            "cross_split_series": 0,
            "cross_split_games": 0,
            "cross_split_rounds": 0,
            "invalid_split_rows": int(len(frame)),
            "invalid_label_rows": int(len(frame)),
            "null_identity_cells": 0,
            "null_feature_cells": 0,
            "split_rows": {},
            "split_series": {},
        }

    duplicate_key_rows = int(frame.duplicated(ID_COLUMNS).sum())
    cross_split_series = int(frame.groupby("series_id")["split"].nunique().gt(1).sum())
    cross_split_games = int(frame.groupby("game_id")["split"].nunique().gt(1).sum())
    cross_split_rounds = int(frame.groupby("round_id")["split"].nunique().gt(1).sum())
    invalid_split_rows = int((~frame["split"].isin(SPLIT_ORDER)).sum())
    invalid_label_rows = int((~frame[LABEL_COL].isin([0, 1])).sum())
    null_identity_cells = int(frame[ID_COLUMNS].isna().sum().sum())
    null_feature_cells = int(frame[raw_features].isna().sum().sum())
    split_rows = {
        split: int(frame["split"].eq(split).sum()) for split in SPLIT_ORDER
    }
    split_series = {
        split: int(frame.loc[frame["split"].eq(split), "series_id"].nunique())
        for split in SPLIT_ORDER
    }
    all_splits_present = all(split_rows[split] > 0 for split in SPLIT_ORDER)
    passed = (
        duplicate_key_rows == 0
        and cross_split_series == 0
        and cross_split_games == 0
        and cross_split_rounds == 0
        and invalid_split_rows == 0
        and invalid_label_rows == 0
        and null_identity_cells == 0
        and null_feature_cells == 0
        and all_splits_present
    )
    return {
        "passed": passed,
        "missing_columns": [],
        "rows": int(len(frame)),
        "series": int(frame["series_id"].nunique()),
        "games": int(frame["game_id"].nunique()),
        "duplicate_key_rows": duplicate_key_rows,
        "cross_split_series": cross_split_series,
        "cross_split_games": cross_split_games,
        "cross_split_rounds": cross_split_rounds,
        "invalid_split_rows": invalid_split_rows,
        "invalid_label_rows": invalid_label_rows,
        "null_identity_cells": null_identity_cells,
        "null_feature_cells": null_feature_cells,
        "all_splits_present": all_splits_present,
        "split_rows": split_rows,
        "split_series": split_series,
    }


def prepare_first_kill_splits(
    frame: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    return prepare_profile_splits(frame, canonical_feature_names())


def replay_frozen_xgboost(
    model_bundle: Mapping[str, Any],
    test_x: pd.DataFrame,
    test_rows: pd.DataFrame,
    saved_predictions: pd.DataFrame,
    *,
    expected_metrics: Mapping[str, float] | None = None,
    tolerance: float = XGBOOST_REPLAY_TOLERANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    required_saved = set(ID_COLUMNS + [LABEL_COL, "xgboost_tuned_probability"])
    missing_saved = sorted(required_saved - set(saved_predictions.columns))
    if missing_saved:
        raise KeyError(f"Saved M17 predictions are missing columns: {missing_saved}")
    bundle_columns = list(model_bundle.get("columns", []))
    if bundle_columns != test_x.columns.tolist():
        raise ValueError("Frozen M21 XGBoost columns do not match M28 encoded columns")
    if len(test_rows) != len(test_x):
        raise ValueError("M28 test identities and encoded rows must have equal length")

    probability = np.asarray(
        model_bundle["model"].predict_proba(test_x)[:, 1], dtype=float
    )
    replayed = test_rows[ID_COLUMNS + [LABEL_COL]].reset_index(drop=True).copy()
    replayed["replayed_probability"] = probability
    replayed["_row_order"] = np.arange(len(replayed))
    saved = saved_predictions[
        ID_COLUMNS + [LABEL_COL, "xgboost_tuned_probability"]
    ].copy()
    duplicate_replayed = int(replayed.duplicated(ID_COLUMNS).sum())
    duplicate_saved = int(saved.duplicated(ID_COLUMNS).sum())
    merged = replayed.merge(
        saved,
        on=ID_COLUMNS,
        how="outer",
        suffixes=("_replayed", "_saved"),
        indicator=True,
    )
    key_mismatch_count = int(merged["_merge"].ne("both").sum())
    matched = merged.loc[merged["_merge"].eq("both")].sort_values("_row_order")
    label_mismatch_count = int(
        matched[f"{LABEL_COL}_replayed"].ne(matched[f"{LABEL_COL}_saved"]).sum()
    )
    differences = (
        matched["replayed_probability"] - matched["xgboost_tuned_probability"]
    ).abs()
    max_difference = float(differences.max()) if len(differences) else math.inf
    invalid_probability_cells = int(
        (~np.isfinite(probability)).sum()
        + ((probability < 0) | (probability > 1)).sum()
    )
    current_metrics = probability_metrics(
        replayed[LABEL_COL].to_numpy(dtype=int), probability, n_bins=10
    )
    metric_differences = {}
    metric_replay_passed = True
    if expected_metrics is not None:
        metric_differences = {
            metric: abs(float(current_metrics[metric]) - float(expected_metrics[metric]))
            for metric in REPORT_METRICS
        }
        metric_replay_passed = max(metric_differences.values()) <= tolerance
    passed = (
        len(probability) == len(test_rows)
        and len(matched) == len(test_rows)
        and duplicate_replayed == 0
        and duplicate_saved == 0
        and key_mismatch_count == 0
        and label_mismatch_count == 0
        and invalid_probability_cells == 0
        and max_difference <= tolerance
        and metric_replay_passed
    )
    return probability, {
        "passed": passed,
        "tolerance": tolerance,
        "replayed_rows": int(len(replayed)),
        "saved_rows": int(len(saved)),
        "matched_rows": int(len(matched)),
        "replayed_duplicate_key_rows": duplicate_replayed,
        "saved_duplicate_key_rows": duplicate_saved,
        "key_mismatch_count": key_mismatch_count,
        "label_mismatch_count": label_mismatch_count,
        "invalid_probability_cells": invalid_probability_cells,
        "max_absolute_probability_difference": max_difference,
        "metrics": current_metrics,
        "metric_absolute_differences": metric_differences,
        "metric_replay_passed": metric_replay_passed,
    }


def audit_predictions(predictions: pd.DataFrame, expected_rows: int) -> dict[str, Any]:
    probability_columns = [f"{name}_probability" for name in MODEL_NAMES]
    values = predictions[probability_columns].to_numpy(dtype=float)
    invalid = int((~np.isfinite(values)).sum() + ((values < 0) | (values > 1)).sum())
    duplicate_keys = int(predictions.duplicated(ID_COLUMNS).sum())
    return {
        "passed": len(predictions) == expected_rows and invalid == 0 and duplicate_keys == 0,
        "rows": int(len(predictions)),
        "expected_rows": int(expected_rows),
        "invalid_probability_cells": invalid,
        "duplicate_key_rows": duplicate_keys,
    }


def assess_metric_targets(metrics: Mapping[str, float]) -> dict[str, Any]:
    assessed = {}
    for metric, target in METRIC_TARGETS.items():
        value = float(metrics[metric])
        if target["higher_is_better"]:
            minimum_passed = value >= target["minimum"]
            stage_passed = value >= target["stage"]
            minimum_gap = max(target["minimum"] - value, 0.0)
            stage_gap = max(target["stage"] - value, 0.0)
        else:
            minimum_passed = value <= target["minimum"]
            stage_passed = value <= target["stage"]
            minimum_gap = max(value - target["minimum"], 0.0)
            stage_gap = max(value - target["stage"], 0.0)
        assessed[metric] = {
            "value": value,
            **target,
            "minimum_passed": bool(minimum_passed),
            "stage_passed": bool(stage_passed),
            "minimum_gap": float(minimum_gap),
            "stage_gap": float(stage_gap),
        }
    return {
        "all_minimum_passed": all(item["minimum_passed"] for item in assessed.values()),
        "all_stage_passed": all(item["stage_passed"] for item in assessed.values()),
        "minimum_passed_count": sum(item["minimum_passed"] for item in assessed.values()),
        "stage_passed_count": sum(item["stage_passed"] for item in assessed.values()),
        "metrics": assessed,
    }


def audit_global_uncertainty(
    intervals: pd.DataFrame, *, n_bootstrap: int
) -> dict[str, Any]:
    required = {
        "metric",
        "point_estimate",
        "ci_lower_95",
        "ci_upper_95",
        "successful_bootstraps",
        "bootstrap_unit",
    }
    missing = sorted(required - set(intervals.columns))
    metric_set = set(intervals["metric"]) if not missing else set()
    finite = bool(
        not missing
        and np.isfinite(
            intervals[["point_estimate", "ci_lower_95", "ci_upper_95"]]
            .to_numpy(dtype=float)
        ).all()
    )
    passed = bool(
        not missing
        and metric_set == set(REPORT_METRICS)
        and len(intervals) == len(REPORT_METRICS)
        and finite
        and (intervals["ci_lower_95"] <= intervals["ci_upper_95"]).all()
        and intervals["successful_bootstraps"].eq(n_bootstrap).all()
        and intervals["bootstrap_unit"].eq("series_id").all()
    )
    return {
        "passed": passed,
        "missing_columns": missing,
        "metric_count": int(len(metric_set)),
        "expected_bootstraps": int(n_bootstrap),
    }


def audit_paired_uncertainty(
    intervals: pd.DataFrame, *, n_bootstrap: int
) -> dict[str, Any]:
    required = {
        "metric",
        "performance_advantage_ci_lower_95",
        "performance_advantage_ci_upper_95",
        "ci_includes_zero",
        "lightgbm_significantly_better",
        "successful_bootstraps",
        "bootstrap_unit",
    }
    missing = sorted(required - set(intervals.columns))
    metric_set = set(intervals["metric"]) if not missing else set()
    if missing:
        return {
            "passed": False,
            "missing_columns": missing,
            "metric_count": 0,
            "significant_better_count": 0,
            "expected_bootstraps": int(n_bootstrap),
        }
    lower = intervals["performance_advantage_ci_lower_95"].to_numpy(dtype=float)
    upper = intervals["performance_advantage_ci_upper_95"].to_numpy(dtype=float)
    includes_zero = (lower <= 0) & (upper >= 0)
    significant = lower > 0
    passed = bool(
        metric_set == set(REPORT_METRICS)
        and len(intervals) == len(REPORT_METRICS)
        and np.isfinite(np.column_stack([lower, upper])).all()
        and (lower <= upper).all()
        and np.array_equal(
            intervals["ci_includes_zero"].to_numpy(dtype=bool), includes_zero
        )
        and np.array_equal(
            intervals["lightgbm_significantly_better"].to_numpy(dtype=bool),
            significant,
        )
        and intervals["successful_bootstraps"].eq(n_bootstrap).all()
        and intervals["bootstrap_unit"].eq("series_id_paired").all()
    )
    return {
        "passed": passed,
        "missing_columns": [],
        "metric_count": int(len(metric_set)),
        "significant_better_count": int(significant.sum()),
        "significant_better_metrics": intervals.loc[
            significant, "metric"
        ].tolist(),
        "all_intervals_include_zero": bool(includes_zero.all()),
        "expected_bootstraps": int(n_bootstrap),
    }


def decide_acceptance(checks: Mapping[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not bool(checks.get(name, False))]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "m28_lightgbm_baseline_complete": not failures,
        "ready_for_m29": not failures,
    }


def _audit_environment(project_root: Path, model: Any) -> dict[str, Any]:
    runtime_version = importlib.metadata.version("lightgbm")
    locked_version = None
    for line in (project_root / "requirements-lock.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.lower().startswith("lightgbm=="):
            locked_version = line.split("==", 1)[1].strip()
            break
    params = model.get_params()
    frozen_params_match = all(
        params.get(name) == expected for name, expected in LIGHTGBM_BASE_PARAMS.items()
    )
    return {
        "passed": bool(
            runtime_version == locked_version
            and params.get("device_type") == "cpu"
            and params.get("random_state") == 42
            and params.get("deterministic") is True
            and params.get("force_col_wise") is True
            and frozen_params_match
        ),
        "runtime_version": runtime_version,
        "locked_version": locked_version,
        "device_type": params.get("device_type"),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cuda_required": False,
        "frozen_params_match": frozen_params_match,
    }


def _collect_git_state(project_root: Path, report_dir: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain").splitlines()
    try:
        report_prefix = report_dir.relative_to(project_root).as_posix() + "/"
    except ValueError:
        report_prefix = None
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "remote": git("remote", "get-url", "origin"),
        "working_tree_status": [
            line
            for line in status
            if report_prefix is None
            or report_prefix not in line.replace("\\", "/")
        ],
    }


def _audit_reproduction_entrypoint(script_path: Path) -> dict[str, Any]:
    if not script_path.is_file():
        return {"passed": False, "missing_tokens": [script_path.as_posix()]}
    text = script_path.read_text(encoding="utf-8")
    required = (
        "src.csdemo.m28_post_first_kill_lightgbm_baseline",
        "data\\processed\\esta_full\\first_kill.parquet",
        "models\\esta_full_m28",
        "reports\\esta_full_m28",
    )
    missing = [token for token in required if token not in text]
    return {"passed": not missing, "missing_tokens": missing}


def _render_report(
    comparison: pd.DataFrame,
    global_intervals: pd.DataFrame,
    paired_intervals: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    test = comparison.loc[comparison["split"].eq("test")].set_index("model")
    lines = [
        "# M28 首杀后 LightGBM 控制变量基线报告",
        "",
        "## 结论",
        "",
        f"验收状态：**{summary['acceptance']['status']}**。",
        "M28 固定 M21 数据、系列赛切分、预测时点、40/82 特征合同和指标口径，",
        "只把算法从冻结 XGBoost 替换为固定参数 LightGBM。测试集未参与训练或选择。",
        "",
        "## 数据与训练合同",
        "",
        f"- 数据：{summary['data']['rows']:,} 行，SHA-256 `{summary['data']['sha256']}`。",
        f"- split 行数：{summary['data']['split_rows']}。",
        f"- split 系列赛：{summary['data']['split_series']}。",
        f"- 特征：{summary['features']['raw_count']} 个原始、{summary['features']['encoded_count']} 个编码。",
        f"- LightGBM 最佳迭代：{summary['model']['best_iteration']}。",
        "- 训练只用 train，早停只用 validation；测试只评估一次。",
        "- 冻结 XGBoost `fit` 调用：0。",
        "",
        "## 五项测试指标",
        "",
        "| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        row = test.loc[model_name]
        lines.append(
            f"| `{model_name}` | {row['accuracy']:.6f} | {row['auc']:.6f} | "
            f"{row['log_loss']:.6f} | {row['brier_score']:.6f} | {row['ece10']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## LightGBM 全局 95% 区间",
            "",
            "| 指标 | 点估计 | 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for row in global_intervals.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['point_estimate']:.6f} | "
            f"[{row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## LightGBM 对 XGBoost 配对区间",
            "",
            "正的性能优势统一表示 LightGBM 更好。只有区间完全大于 0 才能宣称显著领先。",
            "",
            "| 指标 | LightGBM 性能优势 | 配对 95% CI | 显著更好 |",
            "|---|---:|---:|---|",
        ]
    )
    for row in paired_intervals.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['performance_advantage_lightgbm']:.6f} | "
            f"[{row['performance_advantage_ci_lower_95']:.6f}, "
            f"{row['performance_advantage_ci_upper_95']:.6f}] | "
            f"{bool(row['lightgbm_significantly_better'])} |"
        )
    lines.extend(
        [
            "",
            "## 验收解释",
            "",
            f"- 五项最低门槛通过：{summary['metric_targets']['minimum_passed_count']}/5。",
            f"- 五项更高目标通过：{summary['metric_targets']['stage_passed_count']}/5。",
            f"- 配对显著领先指标：{summary['paired_uncertainty']['significant_better_count']}/5。",
            "- LightGBM 是否胜过 XGBoost不是本阶段阻断条件，统计结论以配对区间为准。",
            "",
            "## 后续",
            "",
            "M28 通过后进入 M29 validation-only 受控调参。第四份老师报告必须等待",
            "固定模型评估、解释、接口和最终验收完成，不能把本基线提前当作最终结论。",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _manifest_key(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def run(
    *,
    project_root: str | Path,
    data_path: str | Path = "data/processed/esta_full/first_kill.parquet",
    model_dir: str | Path = "models/esta_full_m28",
    report_dir: str | Path = "reports/esta_full_m28",
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    run_tests: bool = True,
    run_compile: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_path = _resolve(root, data_path)
    model_dir = _resolve(root, model_dir)
    report_dir = _resolve(root, report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m21_summary = _read_json(root / "reports/esta_full_m21/m21_summary.json")
    m21_manifest = _read_json(
        root / "reports/esta_full_m21/m21_experiment_manifest.json"
    )
    data_artifact = fingerprint_file(data_path)
    xgboost_path = root / "models/esta_full_m17/first_kill_xgboost_tuned.joblib"
    xgboost_artifact = fingerprint_file(xgboost_path)
    saved_path = root / "reports/esta_full_m17/test_predictions.csv"
    saved_artifact = fingerprint_file(saved_path)
    expected_saved = m21_manifest["artifact_fingerprints"][
        "reports/esta_full_m17/test_predictions.csv"
    ]
    m21_prerequisite = bool(
        m21_summary.get("acceptance", {}).get("status") == "passed"
        and m21_summary.get("acceptance", {}).get("ready_for_lightgbm_comparison")
        is True
        and data_artifact["sha256"]
        == m21_summary.get("data", {}).get("sha256")
        == EXPECTED_DATA_SHA256
        and xgboost_artifact["sha256"]
        == m21_summary.get("artifacts", {}).get("model", {}).get("sha256")
        == EXPECTED_XGBOOST_SHA256
        and saved_artifact["sha256"] == expected_saved["sha256"]
    )

    data = read_table(data_path)
    data_audit = audit_data_contract(data)
    data_contract = bool(
        data_audit["passed"]
        and data_artifact["sha256"] == EXPECTED_DATA_SHA256
        and data_audit["rows"] == 41027
        and data_audit["series"] == 782
        and data_audit["split_rows"] == EXPECTED_SPLIT_ROWS
        and data_audit["split_series"] == EXPECTED_SPLIT_SERIES
    )

    raw_features = canonical_feature_names()
    prepared = prepare_first_kill_splits(data)
    encoded_columns = prepared["train"][0].columns.tolist()
    feature_contract = build_feature_contract()
    included_features = feature_contract.loc[
        feature_contract["included"], "feature"
    ].tolist()
    xgboost_bundle = joblib.load(xgboost_path)
    feature_contract_passed = bool(
        included_features == raw_features
        and len(raw_features) == 40
        and len(encoded_columns) == 82
        and encoded_columns == list(xgboost_bundle.get("columns", []))
        and raw_features == list(xgboost_bundle.get("raw_features", []))
        and not set(encoded_columns) & set(ID_COLUMNS + [LABEL_COL, "split"])
    )

    lightgbm = make_lightgbm_model()
    environment = _audit_environment(root, lightgbm)
    fit_lightgbm(lightgbm, prepared)
    training_policy = {
        "passed": True,
        "fit_split": "train",
        "early_stopping_split": "val",
        "test_used_for_fit_or_selection": False,
        "selection_metric": "validation_binary_logloss",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "candidate_search": False,
        "official_seed": 42,
    }

    lightgbm_rows, lightgbm_probabilities = evaluate_model(
        lightgbm, prepared, "lightgbm_baseline"
    )
    xgboost_rows, xgboost_probabilities = evaluate_model(
        xgboost_bundle["model"], prepared, "xgboost_frozen"
    )
    test_rows = data.loc[data["split"].eq("test")]
    test_x, test_y, _ = prepared["test"]
    saved_predictions = pd.read_csv(saved_path)
    replayed_xgboost, xgboost_replay = replay_frozen_xgboost(
        xgboost_bundle,
        test_x,
        test_rows,
        saved_predictions,
        expected_metrics=m21_summary["metrics"],
    )
    xgboost_probabilities["test"] = replayed_xgboost
    replayed_test_metrics = probability_metrics(
        test_y, replayed_xgboost, n_bins=10
    )
    for metric, value in replayed_test_metrics.items():
        xgboost_rows.loc[
            xgboost_rows["split"].eq("test"), metric
        ] = value

    comparison = pd.concat([xgboost_rows, lightgbm_rows], ignore_index=True)
    differences = model_metric_differences(comparison)
    lightgbm_test = {
        metric: float(
            lightgbm_rows.loc[lightgbm_rows["split"].eq("test"), metric].iloc[0]
        )
        for metric in REPORT_METRICS
    }
    targets = assess_metric_targets(lightgbm_test)

    predictions = build_prediction_table(
        test_rows,
        {
            "xgboost_frozen": replayed_xgboost,
            "lightgbm_baseline": lightgbm_probabilities["test"],
        },
    )
    prediction_audit = audit_predictions(predictions, len(test_rows))
    lightgbm_bootstrap_input = predictions[
        ID_COLUMNS + [LABEL_COL, "lightgbm_baseline_probability"]
    ].rename(
        columns={
            LABEL_COL: "y_true",
            "lightgbm_baseline_probability": "ct_win_probability",
        }
    )
    global_intervals = bootstrap_metric_intervals(
        lightgbm_bootstrap_input,
        n_bootstrap=n_bootstrap,
        seed=BOOTSTRAP_SEED,
    )
    global_uncertainty = audit_global_uncertainty(
        global_intervals, n_bootstrap=n_bootstrap
    )
    paired_input = predictions[
        ID_COLUMNS
        + [LABEL_COL, "lightgbm_baseline_probability", "xgboost_frozen_probability"]
    ].rename(
        columns={
            LABEL_COL: "y_true",
            "lightgbm_baseline_probability": "lightgbm_probability",
            "xgboost_frozen_probability": "xgboost_probability",
        }
    )
    paired_intervals = paired_model_bootstrap(
        paired_input,
        n_bootstrap=n_bootstrap,
        seed=BOOTSTRAP_SEED,
    )
    paired_uncertainty = audit_paired_uncertainty(
        paired_intervals, n_bootstrap=n_bootstrap
    )

    model_path = model_dir / "post_first_kill_lightgbm_baseline.joblib"
    model_bundle = {
        "model": lightgbm,
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "model_name": "lightgbm_baseline",
        "profile": "canonical_event",
        "raw_features": raw_features,
        "columns": encoded_columns,
        "params": lightgbm.get_params(),
        "best_iteration": int(lightgbm.best_iteration_),
        "data_sha256": data_artifact["sha256"],
        "selection_metric": "validation_binary_logloss",
        "official_seed": 42,
    }
    joblib.dump(model_bundle, model_path)
    model_artifact = fingerprint_file(model_path)

    history_values = next(iter(lightgbm.evals_result_.values()))
    validation_logloss = history_values.get(
        "binary_logloss", next(iter(history_values.values()))
    )
    history = pd.DataFrame(
        {
            "iteration": np.arange(1, len(validation_logloss) + 1),
            "validation_binary_logloss": validation_logloss,
        }
    )

    benchmarks = pd.read_csv(root / "benchmarks/external_round_model_metrics.csv")
    external = compare_benchmarks(lightgbm_test, benchmarks)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    write_markdown_report(
        external,
        lightgbm_test,
        report_dir / "external_benchmark_comparison.md",
        stage_label="M28 post-first-kill LightGBM",
    )
    external_report_passed = bool(
        not external.empty
        and (report_dir / "external_benchmark_comparison.md").is_file()
    )

    comparison.to_csv(report_dir / "m28_model_comparison.csv", index=False)
    predictions.to_csv(report_dir / "m28_test_predictions.csv", index=False)
    global_intervals.to_csv(report_dir / "global_bootstrap_95ci.csv", index=False)
    paired_intervals.to_csv(
        report_dir / "paired_lightgbm_vs_xgboost_bootstrap.csv", index=False
    )
    feature_contract.to_csv(report_dir / "feature_contract.csv", index=False)
    pd.DataFrame({"encoded_feature": encoded_columns}).to_csv(
        report_dir / "encoded_feature_columns.csv", index=False
    )
    history.to_csv(report_dir / "lightgbm_training_history.csv", index=False)

    script_path = root / "scripts/run_post_first_kill_lightgbm_baseline.ps1"
    reproduction_entrypoint = _audit_reproduction_entrypoint(script_path)
    if run_tests:
        automated = run_automated_tests(root)
        match = re.search(r"Ran (\d+) tests?", automated["output"])
        test_count = int(match.group(1)) if match else None
    else:
        automated = {
            "passed": True,
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "command": [],
            "output": "Skipped by caller; exercised separately.\n",
        }
        test_count = None
    compile_result = (
        run_compile_check(root)
        if run_compile
        else {
            "passed": True,
            "return_code": 0,
            "command": [],
            "output": "Skipped by caller; exercised separately.\n",
        }
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated["output"], encoding="utf-8"
    )
    (report_dir / "source_compile_output.txt").write_text(
        compile_result["output"], encoding="utf-8"
    )

    controlled_comparison = bool(
        encoded_columns == list(xgboost_bundle.get("columns", []))
        and len(predictions) == len(test_rows)
        and xgboost_replay["passed"]
        and set(MODEL_NAMES) == set(comparison["model"])
    )
    core_output_names = (
        "m28_model_comparison.csv",
        "m28_test_predictions.csv",
        "global_bootstrap_95ci.csv",
        "paired_lightgbm_vs_xgboost_bootstrap.csv",
        "feature_contract.csv",
        "encoded_feature_columns.csv",
        "lightgbm_training_history.csv",
        "external_benchmark_comparison.csv",
        "external_benchmark_comparison.md",
        "automated_test_output.txt",
        "source_compile_output.txt",
    )
    artifact_manifest_passed = bool(
        model_path.is_file()
        and all((report_dir / name).is_file() for name in core_output_names)
    )
    checks = {
        "m21_prerequisite": m21_prerequisite,
        "data_contract": data_contract,
        "feature_contract": feature_contract_passed,
        "lightgbm_environment": environment["passed"],
        "validation_only_training": training_policy["passed"],
        "frozen_xgboost_replay": xgboost_replay["passed"],
        "probability_contract": prediction_audit["passed"],
        "minimum_metrics": targets["all_minimum_passed"],
        "controlled_comparison": controlled_comparison,
        "global_uncertainty": global_uncertainty["passed"],
        "paired_uncertainty": paired_uncertainty["passed"],
        "external_report": external_report_passed,
        "automated_tests": automated["passed"],
        "source_compile": compile_result["passed"],
        "reproduction_entrypoint": reproduction_entrypoint["passed"],
        "artifact_manifest": artifact_manifest_passed,
    }
    acceptance = decide_acceptance(checks)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "stage": "M28",
        "generated_at_utc": generated_at,
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "experiment_policy": "fixed M21 data/split/features/metrics; replace only model algorithm",
        "acceptance": acceptance,
        "checks": checks,
        "data": {
            "path": data_path.as_posix(),
            "bytes": data_artifact["bytes"],
            "sha256": data_artifact["sha256"],
            "rows": data_audit["rows"],
            "series": data_audit["series"],
            "games": data_audit["games"],
            "split_rows": data_audit["split_rows"],
            "split_series": data_audit["split_series"],
            "duplicate_key_rows": data_audit["duplicate_key_rows"],
            "cross_split_series": data_audit["cross_split_series"],
            "cross_split_games": data_audit["cross_split_games"],
            "cross_split_rounds": data_audit["cross_split_rounds"],
        },
        "features": {
            "profile": "canonical_event",
            "raw_count": len(raw_features),
            "encoded_count": len(encoded_columns),
            "raw_features": raw_features,
            "encoded_columns_match_xgboost": encoded_columns
            == list(xgboost_bundle.get("columns", [])),
        },
        "model": {
            "library": "lightgbm",
            "version": environment["runtime_version"],
            "params": lightgbm.get_params(),
            "best_iteration": int(lightgbm.best_iteration_),
            "deployment_tree_count": int(lightgbm.best_iteration_),
            "model_artifact": model_artifact,
        },
        "metrics": lightgbm_test,
        "test_metrics_by_model": {
            row["model"]: {
                metric: float(row[metric]) for metric in REPORT_METRICS
            }
            for row in comparison.loc[comparison["split"].eq("test")].to_dict(
                orient="records"
            )
        },
        "metric_targets": targets,
        "lightgbm_vs_xgboost_test": differences,
        "global_uncertainty": global_uncertainty,
        "paired_uncertainty": paired_uncertainty,
        "bootstrap": {"samples": n_bootstrap, "seed": BOOTSTRAP_SEED, "unit": "series_id"},
        "xgboost_replay": xgboost_replay,
        "xgboost_fit_calls": 0,
        "prediction_audit": prediction_audit,
        "training_policy": training_policy,
        "environment": environment,
        "external_comparison_rows": int(len(external)),
        "automated_tests": {
            "passed": automated["passed"],
            "return_code": automated["return_code"],
            "elapsed_seconds": automated["elapsed_seconds"],
            "test_count": test_count,
            "skipped": not run_tests,
        },
        "source_compile": {
            "passed": compile_result["passed"],
            "return_code": compile_result["return_code"],
            "skipped": not run_compile,
        },
        "reproduction_entrypoint": reproduction_entrypoint,
        "next_stage": "M29 validation-only post-first-kill LightGBM tuning",
    }
    pd.DataFrame(
        [
            {"check": name, "passed": bool(value), "blocking": True}
            for name, value in checks.items()
        ]
    ).to_csv(report_dir / "m28_checks.csv", index=False)
    write_json(summary, report_dir / "m28_summary.json")
    (report_dir / "m28_post_first_kill_lightgbm_controlled_baseline_report.md").write_text(
        _render_report(comparison, global_intervals, paired_intervals, summary),
        encoding="utf-8",
    )

    output_paths = [
        model_path,
        *(report_dir / name for name in core_output_names),
        report_dir / "m28_checks.csv",
        report_dir / "m28_summary.json",
        report_dir / "m28_post_first_kill_lightgbm_controlled_baseline_report.md",
    ]
    manifest = {
        "stage": "M28",
        "generated_at_utc": generated_at,
        "policy": summary["experiment_policy"],
        "code": _collect_git_state(root, report_dir),
        "inputs": {
            "first_kill_data": data_artifact,
            "frozen_xgboost": xgboost_artifact,
            "saved_m17_predictions": saved_artifact,
            "m21_summary": fingerprint_file(
                root / "reports/esta_full_m21/m21_summary.json"
            ),
            "m28_spec": fingerprint_file(
                root / "docs/m28_post_first_kill_lightgbm_controlled_baseline_spec.md"
            ),
            "m28_module": fingerprint_file(
                root / "src/csdemo/m28_post_first_kill_lightgbm_baseline.py"
            ),
            "m28_tests": fingerprint_file(
                root / "tests/test_m28_post_first_kill_lightgbm_baseline.py"
            ),
            "feature_schema": fingerprint_file(root / "src/csdemo/schema.py"),
            "lightgbm_trainer": fingerprint_file(root / "src/csdemo/train_lgbm.py"),
            "reproduction_script": fingerprint_file(script_path),
            "environment": fingerprint_file(root / "environment.yml"),
            "requirements_lock": fingerprint_file(root / "requirements-lock.txt"),
        },
        "outputs": {
            _manifest_key(root, path): fingerprint_file(path)
            for path in output_paths
        },
        "contract": {
            "raw_features": raw_features,
            "encoded_columns": encoded_columns,
            "split_rows": EXPECTED_SPLIT_ROWS,
            "split_series": EXPECTED_SPLIT_SERIES,
            "training_policy": training_policy,
            "metric_targets": METRIC_TARGETS,
            "bootstrap_samples": n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "checks": checks,
        "acceptance": acceptance,
    }
    write_json(manifest, report_dir / "m28_experiment_manifest.json")

    if acceptance["status"] != "passed":
        raise RuntimeError(
            "M28 acceptance failed: " + ", ".join(acceptance["blocking_failures"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and audit the M28 post-first-kill LightGBM controlled baseline."
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--data", default="data/processed/esta_full/first_kill.parquet"
    )
    parser.add_argument("--model-dir", default="models/esta_full_m28")
    parser.add_argument("--report-dir", default="reports/esta_full_m28")
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    summary = run(
        project_root=args.project_root,
        data_path=args.data,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        n_bootstrap=args.n_bootstrap,
        run_tests=not args.skip_tests,
        run_compile=not args.skip_compile,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

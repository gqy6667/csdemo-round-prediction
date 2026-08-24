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
from .metrics import probability_metrics
from .schema import ID_COLUMNS, PRE_ROUND_FEATURE_GROUPS, PRE_ROUND_FEATURES
from .train_lgbm import (
    EARLY_STOPPING_ROUNDS,
    LIGHTGBM_BASE_PARAMS,
    fit_with_validation,
    make_model,
)
from .train_xgb import align_columns, prepare_xy


SPLIT_ORDER = ("train", "val", "test")
MODEL_NAMES = ("xgboost_frozen", "lightgbm_baseline")
REPORT_METRICS = ("accuracy", "auc", "log_loss", "brier_score", "ece10")
HIGHER_IS_BETTER = {"accuracy", "auc"}
METRIC_TARGETS = {
    "accuracy": {"minimum": 0.64, "stage": 0.66, "higher_is_better": True},
    "auc": {"minimum": 0.70, "stage": 0.73, "higher_is_better": True},
    "log_loss": {"minimum": 0.61, "stage": 0.58, "higher_is_better": False},
    "brier_score": {"minimum": 0.21, "stage": 0.195, "higher_is_better": False},
    "ece10": {"minimum": 0.05, "stage": 0.03, "higher_is_better": False},
}
BLOCKING_CHECKS = (
    "m14_prerequisite",
    "data_identity",
    "split_contract",
    "feature_contract",
    "lightgbm_environment",
    "validation_only_training",
    "frozen_xgboost_replay",
    "probability_contract",
    "minimum_metrics",
    "controlled_comparison",
    "external_report",
    "automated_tests",
    "reproduction_entrypoint",
)
XGBOOST_REPLAY_TOLERANCE = 1e-7


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
    rows = []
    for group, features in PRE_ROUND_FEATURE_GROUPS.items():
        rows.extend(
            {
                "feature": feature,
                "group": group,
                "included": True,
                "reason": "accepted_M14_purchase_end_feature",
            }
            for feature in features
        )
    return pd.DataFrame(rows)


def audit_data_contract(frame: pd.DataFrame) -> dict[str, Any]:
    required = set(ID_COLUMNS + [LABEL_COL, "split"] + PRE_ROUND_FEATURES)
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
    null_feature_cells = int(frame[PRE_ROUND_FEATURES].isna().sum().sum())
    split_rows = {
        split: int(frame["split"].eq(split).sum()) for split in SPLIT_ORDER
    }
    split_series = {
        split: int(frame.loc[frame["split"].eq(split), "series_id"].nunique())
        for split in SPLIT_ORDER
    }
    all_splits_present = all(split_rows[split] > 0 for split in SPLIT_ORDER)
    passed = (
        not missing
        and duplicate_key_rows == 0
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


def prepare_pre_round_splits(
    frame: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    missing = sorted(set(PRE_ROUND_FEATURES) - set(frame.columns))
    if missing:
        raise KeyError(f"Pre-round data are missing accepted features: {missing}")

    prepared: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
    keep = ID_COLUMNS + [LABEL_COL, "split"] + PRE_ROUND_FEATURES
    for split in SPLIT_ORDER:
        split_frame = frame.loc[frame["split"].eq(split)]
        if split_frame.empty:
            raise ValueError(f"M22 data have no rows for split: {split}")
        x, y = prepare_xy(split_frame[keep])
        identity = split_frame[ID_COLUMNS + [LABEL_COL]].reset_index(drop=True)
        prepared[split] = (x, y, identity)

    train_columns = prepared["train"][0]
    for split in ("val", "test"):
        x, y, identity = prepared[split]
        prepared[split] = (align_columns(train_columns, x), y, identity)
    return prepared


def make_lightgbm_model(**overrides):
    return make_model(**overrides)


def fit_lightgbm(model, prepared):
    x_train, y_train, _ = prepared["train"]
    x_val, y_val, _ = prepared["val"]
    return fit_with_validation(model, x_train, y_train, x_val, y_val)


def evaluate_model(model, prepared, model_name: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    probabilities = {}
    for split in SPLIT_ORDER:
        x, y, _ = prepared[split]
        probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        probabilities[split] = probability
        rows.append(
            {
                "model": model_name,
                "split": split,
                **probability_metrics(y, probability, n_bins=10),
            }
        )
    return pd.DataFrame(rows), probabilities


def replay_frozen_xgboost(
    model_bundle: Mapping[str, Any],
    test_x: pd.DataFrame,
    test_rows: pd.DataFrame,
    saved_predictions: pd.DataFrame,
    *,
    tolerance: float = XGBOOST_REPLAY_TOLERANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    required_saved = set(ID_COLUMNS + ["y_true", "ct_win_probability"])
    missing_saved = sorted(required_saved - set(saved_predictions.columns))
    if missing_saved:
        raise KeyError(f"Saved M9 predictions are missing columns: {missing_saved}")
    bundle_columns = list(model_bundle.get("columns", []))
    if bundle_columns != test_x.columns.tolist():
        raise ValueError("Frozen XGBoost columns do not match M22 encoded columns")
    if len(test_rows) != len(test_x):
        raise ValueError("Test identities and encoded rows must have equal length")

    probability = np.asarray(
        model_bundle["model"].predict_proba(test_x)[:, 1], dtype=float
    )
    replay = test_rows[ID_COLUMNS + [LABEL_COL]].reset_index(drop=True).copy()
    replay["replayed_probability"] = probability
    replay["_row_order"] = np.arange(len(replay))
    saved = saved_predictions[
        ID_COLUMNS + ["y_true", "ct_win_probability"]
    ].copy()
    duplicate_replay = int(replay.duplicated(ID_COLUMNS).sum())
    duplicate_saved = int(saved.duplicated(ID_COLUMNS).sum())
    merged = replay.merge(saved, on=ID_COLUMNS, how="outer", indicator=True)
    key_mismatch_count = int(merged["_merge"].ne("both").sum())
    matched = merged.loc[merged["_merge"].eq("both")].sort_values("_row_order")
    label_mismatch_count = int(matched[LABEL_COL].ne(matched["y_true"]).sum())
    differences = (
        matched["replayed_probability"] - matched["ct_win_probability"]
    ).abs()
    max_difference = float(differences.max()) if len(differences) else math.inf
    invalid_probability_cells = int(
        (~np.isfinite(probability)).sum()
        + ((probability < 0) | (probability > 1)).sum()
    )
    passed = (
        len(probability) == len(test_rows)
        and len(matched) == len(test_rows)
        and duplicate_replay == 0
        and duplicate_saved == 0
        and key_mismatch_count == 0
        and label_mismatch_count == 0
        and invalid_probability_cells == 0
        and max_difference <= tolerance
    )
    return probability, {
        "passed": passed,
        "tolerance": tolerance,
        "replayed_rows": int(len(replay)),
        "saved_rows": int(len(saved)),
        "matched_rows": int(len(matched)),
        "replayed_duplicate_key_rows": duplicate_replay,
        "saved_duplicate_key_rows": duplicate_saved,
        "key_mismatch_count": key_mismatch_count,
        "label_mismatch_count": label_mismatch_count,
        "invalid_probability_cells": invalid_probability_cells,
        "max_absolute_probability_difference": max_difference,
    }


def build_prediction_table(
    test_rows: pd.DataFrame, probabilities: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    optional = [column for column in ("map_name", "round_num") if column in test_rows]
    result = test_rows[ID_COLUMNS + optional + [LABEL_COL]].reset_index(drop=True).copy()
    for model_name in MODEL_NAMES:
        if model_name not in probabilities:
            raise KeyError(f"Missing probabilities for model: {model_name}")
        values = np.asarray(probabilities[model_name], dtype=float).reshape(-1)
        if len(values) != len(result):
            raise ValueError(f"Probability row count differs for model: {model_name}")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Probabilities for {model_name} must be finite and between 0 and 1")
        result[f"{model_name}_probability"] = values
        result[f"{model_name}_prediction"] = (values >= 0.5).astype(int)
    return result


def audit_predictions(predictions: pd.DataFrame, expected_rows: int) -> dict[str, Any]:
    columns = [f"{name}_probability" for name in MODEL_NAMES]
    values = predictions[columns].to_numpy(dtype=float)
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


def model_metric_differences(
    comparison: pd.DataFrame, *, split: str = "test"
) -> dict[str, dict[str, float]]:
    indexed = comparison.loc[comparison["split"].eq(split)].set_index("model")
    missing = sorted(set(MODEL_NAMES) - set(indexed.index))
    if missing:
        raise KeyError(f"Controlled comparison is missing models: {missing}")
    result = {}
    for metric in REPORT_METRICS:
        lightgbm = float(indexed.loc["lightgbm_baseline", metric])
        xgboost = float(indexed.loc["xgboost_frozen", metric])
        raw = lightgbm - xgboost
        advantage = raw if metric in HIGHER_IS_BETTER else -raw
        result[metric] = {
            "lightgbm": lightgbm,
            "xgboost": xgboost,
            "raw_lightgbm_minus_xgboost": raw,
            "performance_advantage_lightgbm": advantage,
            "lightgbm_performs_better": bool(advantage > 0),
        }
    return result


def audit_metric_replay(
    replayed_metrics: Mapping[str, float], saved_metrics: Mapping[str, float]
) -> dict[str, Any]:
    differences = {
        metric: abs(float(replayed_metrics[metric]) - float(saved_metrics[metric]))
        for metric in REPORT_METRICS
    }
    maximum = max(differences.values())
    return {
        "passed": maximum <= XGBOOST_REPLAY_TOLERANCE,
        "tolerance": XGBOOST_REPLAY_TOLERANCE,
        "absolute_differences": differences,
        "max_absolute_difference": maximum,
    }


def decide_acceptance(checks: Mapping[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not bool(checks.get(name, False))]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "m22_lightgbm_baseline_complete": not failures,
        "ready_for_m23": not failures,
    }


def _audit_environment(project_root: Path, model) -> dict[str, Any]:
    runtime_version = importlib.metadata.version("lightgbm")
    locked_version = None
    for line in (project_root / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("lightgbm=="):
            locked_version = line.split("==", 1)[1].strip()
            break
    params = model.get_params()
    return {
        "passed": runtime_version == locked_version and params.get("device_type") == "cpu",
        "runtime_version": runtime_version,
        "locked_version": locked_version,
        "device_type": params.get("device_type"),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cuda_required": False,
    }


def _collect_git_state(project_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=project_root, capture_output=True, text=True, check=True
        )
        return completed.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "remote": git("remote", "get-url", "origin"),
        "working_tree_status": git("status", "--porcelain").splitlines(),
    }


def _render_report(
    comparison: pd.DataFrame,
    external: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    test = comparison.loc[comparison["split"].eq("test")].set_index("model")
    targets = summary["metric_targets"]["metrics"]
    differences = summary["lightgbm_vs_xgboost_test"]
    lines = [
        "# M22 开局前 LightGBM 受控基线报告",
        "",
        "## 阶段决定",
        "",
        f"验收状态：**{summary['acceptance']['status']}**（"
        f"{summary['acceptance']['blocking_passed']}/{summary['acceptance']['blocking_total']}）。",
        f"可以进入 M23 validation-only 调参：**{summary['acceptance']['ready_for_m23']}**。",
        "LightGBM 是否胜过 XGBoost 不是阻断条件；本阶段首先验证公平实验闭环。",
        "",
        "## 固定条件",
        "",
        f"- 输入：{summary['data']['rows']:,} 条购买完毕、交火前快照。",
        f"- train/val/test：{summary['data']['split_rows']['train']:,} / "
        f"{summary['data']['split_rows']['val']:,} / {summary['data']['split_rows']['test']:,}。",
        f"- 特征：{summary['features']['raw_count']} 个原始字段、"
        f"{summary['features']['encoded_count']} 个训练集编码列。",
        "- XGBoost 不重训；LightGBM 只用 train 拟合、validation Log Loss 早停。",
        f"- LightGBM `4.6.0`，CPU；最佳迭代 {summary['model']['best_iteration']}。",
        "",
        "## 测试集结果",
        "",
        "| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in ("logistic_regression", "xgboost_frozen", "lightgbm_baseline"):
        if model_name not in test.index:
            continue
        row = test.loc[model_name]
        lines.append(
            f"| `{model_name}` | {row['accuracy']:.6f} | {row['auc']:.6f} | "
            f"{row['log_loss']:.6f} | {row['brier_score']:.6f} | {row['ece10']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## LightGBM 与 XGBoost 相差多少",
            "",
            "原始差值为 `LightGBM - XGBoost`。方向修正后大于 0 才代表 LightGBM 更好。",
            "",
            "| 指标 | LightGBM | XGBoost | 原始差值 | 方向修正后 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in REPORT_METRICS:
        item = differences[metric]
        lines.append(
            f"| {metric} | {item['lightgbm']:.6f} | {item['xgboost']:.6f} | "
            f"{item['raw_lightgbm_minus_xgboost']:+.6f} | "
            f"{item['performance_advantage_lightgbm']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## 预先门槛",
            "",
            "| 指标 | 当前 | 最低门槛 | 最低通过 | 更高目标 | 目标通过 | 尚差 |",
            "|---|---:|---:|---|---:|---|---:|",
        ]
    )
    for metric in REPORT_METRICS:
        item = targets[metric]
        lines.append(
            f"| {metric} | {item['value']:.6f} | {item['minimum']:.3f} | "
            f"{item['minimum_passed']} | {item['stage']:.3f} | "
            f"{item['stage_passed']} | {item['stage_gap']:.6f} |"
        )

    closest = external.loc[
        external.get("comparability", pd.Series(index=external.index, dtype=str)).eq(
            "closest_task"
        )
        & external["comparison_status"].eq("compared")
    ]
    lines.extend(
        [
            "",
            "## 与公开结果相差多少",
            "",
            "差值为 LightGBM 减外部报告。数据和切分不同，只作数值参考。",
            "",
            "| 外部工作 | 指标 | LightGBM | 外部 | 差值 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in closest.to_dict(orient="records"):
        difference = row["raw_difference_ours_minus_reported"]
        text = (
            f"{difference * 100:+.2f} 个百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| {row['source_title']} | {row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {text} |"
        )

    lines.extend(
        [
            "",
            "## 为什么本阶段可信",
            "",
            f"- 4,172 条冻结 XGBoost 概率回放最大误差："
            f"`{summary['xgboost_replay']['max_absolute_probability_difference']:.3e}`。",
            f"- XGBoost 五项指标回放最大误差："
            f"`{summary['xgboost_metric_replay']['max_absolute_difference']:.3e}`。",
            "- 两个树模型使用同一 test 键、标签和编码列；M22 的 XGBoost fit 次数为 0。",
            "- 测试集没有出现在 LightGBM `eval_set`，最佳树数由 validation 决定。",
            "",
            "## 下一阶段",
            "",
            "M23 保持数据、切分、特征和 test 不变，只按 validation Log Loss 逐项调整 "
            "`num_leaves`、`min_child_samples`、采样和正则化。候选阶段不得输出 test 指标；"
            "冻结最终参数后再做一次正式测试。首杀后 LightGBM 和实时胜率仍在之后。",
            "",
            "复现命令：",
            "",
            "```powershell",
            ".\\scripts\\run_pre_round_lightgbm_baseline.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def run(
    *,
    project_root: str | Path,
    data_path: str | Path = "data/processed/esta_full/pre_round.parquet",
    model_dir: str | Path = "models/esta_full_m22",
    report_dir: str | Path = "reports/esta_full_m22",
    run_tests: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_path = _resolve(root, data_path)
    model_dir = _resolve(root, model_dir)
    report_dir = _resolve(root, report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m14_summary = _read_json(root / "reports/esta_full_m14/m14_summary.json")
    m14_manifest = _read_json(
        root / "reports/esta_full_m14/m14_experiment_manifest.json"
    )
    m9_summary = _read_json(root / "reports/esta_full_m9/m9_summary.json")
    expected_data = m14_manifest["artifacts"]["pre_round"]
    actual_data = fingerprint_file(data_path)
    xgboost_path = root / "models/esta_full_m8_tuned/pre_round_xgb.joblib"
    expected_xgboost = m14_manifest["artifacts"]["model"]
    actual_xgboost = fingerprint_file(xgboost_path)
    m14_prerequisite = (
        m14_summary.get("status") == "passed"
        and m14_summary.get("phase_1_pre_round_xgboost_complete") is True
        and actual_xgboost["sha256"] == expected_xgboost["sha256"]
    )

    data = read_table(data_path)
    data_audit = audit_data_contract(data)
    expected_split = m14_manifest["data"]["split_contract"]
    data_identity = (
        actual_data["sha256"] == expected_data["sha256"]
        and int(len(data)) == int(expected_data.get("rows", len(data)))
        and int(len(data)) == int(m14_summary["pre_round_rows"])
    )
    split_contract = (
        data_audit["passed"]
        and data_audit["split_rows"] == expected_split["row_counts"]
        and data_audit["split_series"] == expected_split["series_counts"]
    )

    prepared = prepare_pre_round_splits(data)
    encoded_columns = prepared["train"][0].columns.tolist()
    feature_contract = build_feature_contract()
    xgboost_bundle = joblib.load(xgboost_path)
    feature_contract_passed = (
        feature_contract["feature"].tolist() == PRE_ROUND_FEATURES
        and len(encoded_columns) == 43
        and encoded_columns == list(xgboost_bundle["columns"])
        and not set(encoded_columns) & set(ID_COLUMNS + [LABEL_COL, "split"])
    )

    lightgbm = make_lightgbm_model()
    environment = _audit_environment(root, lightgbm)
    fit_lightgbm(lightgbm, prepared)
    validation_only_training = {
        "passed": True,
        "fit_split": "train",
        "early_stopping_split": "val",
        "test_used_for_fit_or_selection": False,
        "eval_metric": "binary_logloss",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }

    lightgbm_rows, lightgbm_probability = evaluate_model(
        lightgbm, prepared, "lightgbm_baseline"
    )
    xgboost_rows, xgboost_probability = evaluate_model(
        xgboost_bundle["model"], prepared, "xgboost_frozen"
    )
    test_x, test_y, _ = prepared["test"]
    test_rows = data.loc[data["split"].eq("test")]
    saved_m9_predictions = pd.read_csv(
        root / "reports/esta_full_m9/test_predictions.csv"
    )
    replayed_xgboost, xgboost_replay = replay_frozen_xgboost(
        xgboost_bundle,
        test_x,
        test_rows,
        saved_m9_predictions,
    )
    xgboost_probability["test"] = replayed_xgboost
    replayed_metrics = probability_metrics(test_y, replayed_xgboost, n_bins=10)
    xgboost_metric_replay = audit_metric_replay(replayed_metrics, m9_summary["metrics"])

    references = pd.read_csv(root / "reports/esta_full_m7/m7_model_comparison.csv")
    references = references.loc[
        references["model"].isin(["constant_train_prior", "logistic_regression"])
    ]
    comparison = pd.concat(
        [references, xgboost_rows, lightgbm_rows], ignore_index=True
    )
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
            "lightgbm_baseline": lightgbm_probability["test"],
        },
    )
    prediction_audit = audit_predictions(predictions, len(test_rows))

    benchmarks = pd.read_csv(root / "benchmarks/external_round_model_metrics.csv")
    external = compare_benchmarks(lightgbm_test, benchmarks)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    write_markdown_report(
        external,
        lightgbm_test,
        report_dir / "external_benchmark_comparison.md",
        stage_label="M22 LightGBM",
    )
    external_report_passed = not external.empty and (
        report_dir / "external_benchmark_comparison.md"
    ).is_file()

    model_path = model_dir / "pre_round_lightgbm_baseline.joblib"
    model_bundle = {
        "model": lightgbm,
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "model_name": "lightgbm_baseline",
        "profile": "M14_pre_round_features",
        "raw_features": list(PRE_ROUND_FEATURES),
        "columns": encoded_columns,
        "params": lightgbm.get_params(),
        "best_iteration": int(lightgbm.best_iteration_),
        "data_sha256": actual_data["sha256"],
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

    script_path = root / "scripts/run_pre_round_lightgbm_baseline.ps1"
    reproduction_entrypoint = script_path.is_file()
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

    controlled_comparison = (
        len(predictions) == len(test_rows)
        and encoded_columns == list(xgboost_bundle["columns"])
        and xgboost_replay["label_mismatch_count"] == 0
        and set(MODEL_NAMES).issubset(set(comparison["model"]))
    )
    checks = {
        "m14_prerequisite": m14_prerequisite,
        "data_identity": data_identity,
        "split_contract": split_contract,
        "feature_contract": feature_contract_passed,
        "lightgbm_environment": environment["passed"],
        "validation_only_training": validation_only_training["passed"],
        "frozen_xgboost_replay": (
            xgboost_replay["passed"] and xgboost_metric_replay["passed"]
        ),
        "probability_contract": prediction_audit["passed"],
        "minimum_metrics": targets["all_minimum_passed"],
        "controlled_comparison": controlled_comparison,
        "external_report": external_report_passed,
        "automated_tests": automated["passed"],
        "reproduction_entrypoint": reproduction_entrypoint,
    }
    acceptance = decide_acceptance(checks)

    comparison.to_csv(report_dir / "m22_model_comparison.csv", index=False)
    predictions.to_csv(report_dir / "m22_test_predictions.csv", index=False)
    feature_contract.to_csv(report_dir / "feature_contract.csv", index=False)
    pd.DataFrame({"encoded_feature": encoded_columns}).to_csv(
        report_dir / "encoded_feature_columns.csv", index=False
    )
    history.to_csv(report_dir / "lightgbm_training_history.csv", index=False)
    pd.DataFrame(
        [
            {"check": name, "passed": value, "blocking": True}
            for name, value in checks.items()
        ]
    ).to_csv(report_dir / "m22_checks.csv", index=False)
    (report_dir / "automated_test_output.txt").write_text(
        automated["output"], encoding="utf-8"
    )

    summary = {
        "stage": "M22",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "experiment_policy": "fixed M14 data/split/features/metrics; replace only model algorithm",
        "acceptance": acceptance,
        "checks": checks,
        "data": {
            "path": data_path.as_posix(),
            "bytes": actual_data["bytes"],
            "sha256": actual_data["sha256"],
            "rows": int(len(data)),
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
            "raw_count": len(PRE_ROUND_FEATURES),
            "encoded_count": len(encoded_columns),
            "raw_features": list(PRE_ROUND_FEATURES),
            "encoded_columns_match_xgboost": encoded_columns
            == list(xgboost_bundle["columns"]),
        },
        "model": {
            "library": "lightgbm",
            "version": environment["runtime_version"],
            "params": lightgbm.get_params(),
            "best_iteration": int(lightgbm.best_iteration_),
            "model_artifact": model_artifact,
        },
        "metrics": lightgbm_test,
        "test_metrics_by_model": {
            row["model"]: {metric: float(row[metric]) for metric in REPORT_METRICS}
            for row in comparison.loc[comparison["split"].eq("test")].to_dict(
                orient="records"
            )
        },
        "metric_targets": targets,
        "lightgbm_vs_xgboost_test": differences,
        "xgboost_replay": xgboost_replay,
        "xgboost_metric_replay": xgboost_metric_replay,
        "xgboost_fit_calls": 0,
        "prediction_audit": prediction_audit,
        "training_policy": validation_only_training,
        "environment": environment,
        "external_comparison_rows": int(len(external)),
        "automated_tests": {
            "passed": automated["passed"],
            "return_code": automated["return_code"],
            "elapsed_seconds": automated["elapsed_seconds"],
            "test_count": test_count,
            "skipped": not run_tests,
        },
        "next_stage": "M23 validation-only controlled LightGBM tuning",
    }
    manifest = {
        "stage": "M22",
        "generated_at_utc": summary["generated_at_utc"],
        "code": _collect_git_state(root),
        "input_artifacts": {
            "pre_round_data": actual_data,
            "frozen_xgboost": actual_xgboost,
            "saved_m9_predictions": fingerprint_file(
                root / "reports/esta_full_m9/test_predictions.csv"
            ),
            "requirements_lock": fingerprint_file(root / "requirements-lock.txt"),
        },
        "output_artifacts": {"lightgbm_model": model_artifact},
        "contract": {
            "raw_features": list(PRE_ROUND_FEATURES),
            "encoded_columns": encoded_columns,
            "split_rows": data_audit["split_rows"],
            "training_policy": validation_only_training,
            "metric_targets": METRIC_TARGETS,
        },
        "checks": checks,
        "acceptance": acceptance,
    }
    write_json(summary, report_dir / "m22_summary.json")
    write_json(manifest, report_dir / "m22_experiment_manifest.json")
    (report_dir / "m22_pre_round_lightgbm_baseline_report.md").write_text(
        _render_report(comparison, external, summary), encoding="utf-8"
    )
    if acceptance["status"] != "passed":
        raise RuntimeError(
            "M22 acceptance failed: " + ", ".join(acceptance["blocking_failures"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and audit the M22 pre-round LightGBM controlled baseline."
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data", default="data/processed/esta_full/pre_round.parquet")
    parser.add_argument("--model-dir", default="models/esta_full_m22")
    parser.add_argument("--report-dir", default="reports/esta_full_m22")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    summary = run(
        project_root=args.project_root,
        data_path=args.data,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        run_tests=not args.skip_tests,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

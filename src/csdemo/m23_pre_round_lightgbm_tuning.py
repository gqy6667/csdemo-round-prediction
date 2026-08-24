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
from time import perf_counter
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .benchmark_comparison import compare_benchmarks, write_markdown_report
from .io import read_table
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m22_pre_round_lightgbm_baseline import (
    assess_metric_targets,
    audit_data_contract,
    prepare_pre_round_splits,
    write_json,
)
from .metrics import probability_metrics
from .schema import ID_COLUMNS, PRE_ROUND_FEATURES
from .train_lgbm import LIGHTGBM_BASE_PARAMS, fit_with_validation, make_model


MINIMUM_PHASE_IMPROVEMENT = 0.0001
STABILITY_SEEDS = (42, 43, 44, 45, 46)
STABILITY_LIMITS = {"val_log_loss": 0.002, "val_auc": 0.003}
REPORT_METRICS = ("accuracy", "auc", "log_loss", "brier_score", "ece10")
HIGHER_IS_BETTER = {"accuracy", "auc"}

BASE_TUNING_PARAMS: dict[str, Any] = {
    **LIGHTGBM_BASE_PARAMS,
    "max_depth": -1,
    "min_split_gain": 0.0,
    "random_state": 42,
}


def _phase(
    name: str, values: tuple[Any, ...]
) -> dict[str, Any]:
    return {
        "name": name,
        "allowed_parameters": (name,),
        "candidates": tuple(
            {
                "id": f"{name}_{value}",
                "label": str(value),
                "overrides": {name: value},
            }
            for value in values
        ),
    }


PHASE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    _phase("num_leaves", (7, 15, 31, 63)),
    _phase("max_depth", (-1, 3, 4, 5, 6)),
    _phase("min_child_samples", (10, 20, 40, 80, 160)),
    _phase("reg_lambda", (0.0, 1.0, 3.0, 10.0)),
    _phase("reg_alpha", (0.0, 0.1, 0.5, 1.0)),
    _phase("subsample", (0.7, 0.85, 1.0)),
    _phase("colsample_bytree", (0.7, 0.85, 1.0)),
    _phase("min_split_gain", (0.0, 0.01, 0.05, 0.1)),
    _phase("learning_rate", (0.01, 0.02, 0.03, 0.05)),
)

BLOCKING_CHECKS = (
    "m22_prerequisite",
    "data_contract",
    "feature_contract",
    "candidate_grid",
    "validation_only",
    "phase_selection",
    "frozen_model",
    "seed_stability",
    "final_predictions",
    "minimum_metrics",
    "controlled_comparison",
    "external_report",
    "automated_tests",
    "reproduction_entrypoint",
)


def audit_candidate_grid(
    phases: tuple[dict[str, Any], ...], base_params: Mapping[str, Any]
) -> dict[str, Any]:
    violations: list[str] = []
    candidate_ids: list[str] = []
    changed_by_phase: dict[str, list[str]] = {}

    for phase in phases:
        name = str(phase["name"])
        allowed = tuple(phase.get("allowed_parameters", ()))
        candidates = tuple(phase.get("candidates", ()))
        if len(allowed) != 1:
            violations.append(f"{name}: must allow exactly one parameter")
            continue
        if not candidates:
            violations.append(f"{name}: no candidates")
            continue
        parameter = allowed[0]
        resolved_values = []
        changed_union: set[str] = set()
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            candidate_ids.append(candidate_id)
            overrides = dict(candidate.get("overrides", {}))
            extra = set(overrides) - {parameter}
            if extra:
                violations.append(
                    f"{candidate_id}: changes forbidden parameters {sorted(extra)}"
                )
            if parameter not in overrides:
                violations.append(f"{candidate_id}: missing {parameter} override")
                continue
            value = overrides[parameter]
            resolved_values.append(value)
            if base_params.get(parameter) != value:
                changed_union.add(parameter)
        if len(resolved_values) != len(set(resolved_values)):
            violations.append(f"{name}: duplicate candidate parameter values")
        if base_params.get(parameter) not in resolved_values:
            violations.append(f"{name}: base incumbent is absent")
        changed_by_phase[name] = sorted(changed_union)

    duplicate_ids = sorted(
        candidate_id
        for candidate_id in set(candidate_ids)
        if candidate_ids.count(candidate_id) > 1
    )
    if duplicate_ids:
        violations.append(f"duplicate candidate ids: {duplicate_ids}")
    return {
        "passed": not violations,
        "phase_count": len(phases),
        "candidate_count": len(candidate_ids),
        "changed_parameters_by_phase": changed_by_phase,
        "violations": violations,
    }


def select_phase_winner(
    results: pd.DataFrame, minimum_improvement: float
) -> dict[str, Any]:
    required = {"candidate_id", "candidate_order", "is_incumbent", "val_log_loss"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise KeyError(f"Phase results are missing columns: {missing}")
    incumbents = results.loc[results["is_incumbent"]]
    if len(incumbents) != 1:
        raise ValueError("Each phase must contain exactly one incumbent candidate")
    values = results["val_log_loss"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Validation Log Loss values must be finite")

    incumbent = incumbents.iloc[0]
    best = results.sort_values(
        ["val_log_loss", "candidate_order"], kind="stable"
    ).iloc[0]
    improvement = float(incumbent["val_log_loss"] - best["val_log_loss"])
    changed = improvement + 1e-12 >= minimum_improvement
    selected = best if changed else incumbent
    accepted_improvement = float(
        incumbent["val_log_loss"] - selected["val_log_loss"]
    )
    return {
        "candidate_id": str(selected["candidate_id"]),
        "changed": bool(changed),
        "incumbent_id": str(incumbent["candidate_id"]),
        "best_observed_id": str(best["candidate_id"]),
        "best_observed_improvement": improvement,
        "accepted_improvement": accepted_improvement,
        "selected_val_log_loss": float(selected["val_log_loss"]),
    }


def validate_validation_only_table(results: pd.DataFrame) -> dict[str, Any]:
    forbidden = sorted(
        column for column in results.columns if column.startswith("test_")
    )
    return {
        "passed": not forbidden,
        "rows": int(len(results)),
        "forbidden_columns": forbidden,
    }


def audit_frozen_params(model: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    actual = model.get_params()
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    return {"passed": not mismatches, "expected": dict(expected), "mismatches": mismatches}


def assess_seed_stability(seed_results: pd.DataFrame) -> dict[str, Any]:
    required = {"seed", "val_log_loss", "val_auc"}
    missing = sorted(required - set(seed_results.columns))
    if missing:
        raise KeyError(f"Seed stability results are missing columns: {missing}")
    forbidden = sorted(
        column for column in seed_results.columns if column.startswith("test_")
    )
    observed_seeds = tuple(sorted(seed_results["seed"].astype(int).tolist()))
    log_loss_range = float(
        seed_results["val_log_loss"].max() - seed_results["val_log_loss"].min()
    )
    auc_range = float(seed_results["val_auc"].max() - seed_results["val_auc"].min())
    passed = (
        observed_seeds == STABILITY_SEEDS
        and not forbidden
        and log_loss_range <= STABILITY_LIMITS["val_log_loss"]
        and auc_range <= STABILITY_LIMITS["val_auc"]
    )
    return {
        "passed": passed,
        "observed_seeds": list(observed_seeds),
        "forbidden_columns": forbidden,
        "val_log_loss_range": log_loss_range,
        "val_auc_range": auc_range,
        "limits": dict(STABILITY_LIMITS),
    }


def build_final_prediction_table(
    test_rows: pd.DataFrame,
    tuned_probabilities: np.ndarray,
    m22_predictions: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ID_COLUMNS + ["ct_win"]
    missing = sorted(set(key_columns) - set(test_rows.columns))
    if missing:
        raise KeyError(f"Test rows are missing identity columns: {missing}")
    missing_m22 = sorted(set(key_columns) - set(m22_predictions.columns))
    if missing_m22:
        raise KeyError(f"M22 predictions are missing identity columns: {missing_m22}")

    current_keys = test_rows[key_columns].reset_index(drop=True)
    m22_keys = m22_predictions[key_columns].reset_index(drop=True)
    if not current_keys.equals(m22_keys):
        raise ValueError("M23 test rows do not match the exact M22 test keys")

    probability = np.asarray(tuned_probabilities, dtype=float).reshape(-1)
    if len(probability) != len(current_keys):
        raise ValueError("M23 tuned probability row count differs from test rows")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("M23 tuned probabilities must be finite and between 0 and 1")

    result = m22_predictions.reset_index(drop=True).copy()
    result["lightgbm_tuned_probability"] = probability
    result["lightgbm_tuned_prediction"] = (probability >= 0.5).astype(int)
    return result


def decide_acceptance(checks: Mapping[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not bool(checks.get(name, False))]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "m23_lightgbm_tuning_complete": not failures,
        "ready_for_m24": not failures,
    }


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None else None


def _candidate_metrics(
    model: Any,
    train_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    val_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
) -> dict[str, float]:
    x_train, y_train, _ = train_prepared
    x_val, y_val, _ = val_prepared
    train_probability = np.asarray(model.predict_proba(x_train)[:, 1], dtype=float)
    val_probability = np.asarray(model.predict_proba(x_val)[:, 1], dtype=float)
    train_metrics = probability_metrics(y_train, train_probability, n_bins=10)
    val_metrics = probability_metrics(y_val, val_probability, n_bins=10)
    return {
        **{f"train_{name}": float(value) for name, value in train_metrics.items()},
        **{f"val_{name}": float(value) for name, value in val_metrics.items()},
    }


def _fit_candidate(
    train_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    val_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    params: Mapping[str, Any],
) -> tuple[Any, dict[str, float], float]:
    x_train, y_train, _ = train_prepared
    x_val, y_val, _ = val_prepared
    model = make_model(**dict(params))
    started = perf_counter()
    fit_with_validation(model, x_train, y_train, x_val, y_val)
    elapsed_seconds = perf_counter() - started
    return model, _candidate_metrics(model, train_prepared, val_prepared), elapsed_seconds


def _is_incumbent_candidate(
    candidate: Mapping[str, Any],
    current_params: Mapping[str, Any],
    parameter: str,
) -> bool:
    return candidate["overrides"][parameter] == current_params[parameter]


def run_sequential_search(
    train_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    val_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    phases: tuple[dict[str, Any], ...] = PHASE_DEFINITIONS,
    minimum_improvement: float = MINIMUM_PHASE_IMPROVEMENT,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, dict[str, Any]]:
    grid_audit = audit_candidate_grid(phases, BASE_TUNING_PARAMS)
    if not grid_audit["passed"]:
        raise ValueError(f"Invalid M23 candidate grid: {grid_audit['violations']}")

    current_params = dict(BASE_TUNING_PARAMS)
    all_results: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    selected_model: Any = None

    for phase_order, phase in enumerate(phases, start=1):
        phase_name = str(phase["name"])
        parameter = str(phase["allowed_parameters"][0])
        candidate_rows: list[dict[str, Any]] = []
        candidate_models: dict[str, Any] = {}
        candidate_params: dict[str, dict[str, Any]] = {}

        for candidate_order, candidate in enumerate(phase["candidates"]):
            candidate_id = str(candidate["id"])
            params = {**current_params, **candidate["overrides"]}
            model, metrics, elapsed_seconds = _fit_candidate(
                train_prepared, val_prepared, params
            )
            best_iteration = _best_iteration(model)
            candidate_models[candidate_id] = model
            candidate_params[candidate_id] = params
            candidate_rows.append(
                {
                    "phase_order": phase_order,
                    "phase": phase_name,
                    "candidate_order": candidate_order,
                    "candidate_id": candidate_id,
                    "candidate_label": str(candidate["label"]),
                    "is_incumbent": _is_incumbent_candidate(
                        candidate, current_params, parameter
                    ),
                    "changed_parameter": parameter,
                    "parameter_value": json.dumps(
                        candidate["overrides"][parameter], sort_keys=True
                    ),
                    "best_iteration": best_iteration,
                    "best_tree_count": (
                        best_iteration
                        if best_iteration is not None
                        else int(params["n_estimators"])
                    ),
                    "elapsed_seconds": elapsed_seconds,
                    **metrics,
                }
            )

        phase_results = pd.DataFrame(candidate_rows)
        selected = select_phase_winner(phase_results, minimum_improvement)
        selected_id = selected["candidate_id"]
        phase_results["selected"] = phase_results["candidate_id"].eq(selected_id)
        selected_row = phase_results.loc[phase_results["selected"]].iloc[0]
        current_params = candidate_params[selected_id]
        selected_model = candidate_models[selected_id]
        selections.append(
            {
                "phase_order": phase_order,
                "phase": phase_name,
                **selected,
                "selected_candidate_label": selected_row["candidate_label"],
                "selected_parameter_value": selected_row["parameter_value"],
                "selected_val_auc": float(selected_row["val_auc"]),
                "selected_val_brier_score": float(
                    selected_row["val_brier_score"]
                ),
                "selected_best_tree_count": int(selected_row["best_tree_count"]),
                "selected_params": json.dumps(current_params, sort_keys=True),
            }
        )
        all_results.append(phase_results)

    if selected_model is None:
        raise ValueError("M23 requires at least one tuning phase")
    results = pd.concat(all_results, ignore_index=True)
    validation_audit = validate_validation_only_table(results)
    if not validation_audit["passed"]:
        raise RuntimeError("M23 tuning results unexpectedly contain test metrics")
    return results, pd.DataFrame(selections), selected_model, current_params


def audit_phase_selections(
    tuning_results: pd.DataFrame,
    phase_selections: pd.DataFrame,
    minimum_improvement: float = MINIMUM_PHASE_IMPROVEMENT,
) -> dict[str, Any]:
    failures = []
    for phase_name, rows in tuning_results.groupby("phase", sort=False):
        expected = select_phase_winner(rows, minimum_improvement)
        actual = phase_selections.loc[phase_selections["phase"].eq(phase_name)]
        if len(actual) != 1:
            failures.append(f"{phase_name}: expected one selection row")
            continue
        if actual.iloc[0]["candidate_id"] != expected["candidate_id"]:
            failures.append(f"{phase_name}: selected candidate differs from policy")
        marked = rows.loc[rows["selected"]]
        if len(marked) != 1 or marked.iloc[0]["candidate_id"] != expected["candidate_id"]:
            failures.append(f"{phase_name}: candidate marker differs from policy")
    return {
        "passed": not failures,
        "phase_count": int(tuning_results["phase"].nunique()),
        "selection_rows": int(len(phase_selections)),
        "failures": failures,
    }


def run_seed_stability(
    train_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    val_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    frozen_params: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    for seed in STABILITY_SEEDS:
        params = {**dict(frozen_params), "random_state": seed}
        model, metrics, elapsed_seconds = _fit_candidate(
            train_prepared, val_prepared, params
        )
        rows.append(
            {
                "seed": seed,
                "best_iteration": _best_iteration(model),
                "elapsed_seconds": elapsed_seconds,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_frozen_model(
    model: Any,
    prepared: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    probabilities = {}
    for split in ("train", "val", "test"):
        x, y, _ = prepared[split]
        probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        probabilities[split] = probability
        rows.append(
            {
                "model": "lightgbm_tuned",
                "split": split,
                **probability_metrics(y, probability, n_bins=10),
            }
        )
    return pd.DataFrame(rows), probabilities


def metric_differences(
    comparison: pd.DataFrame,
    left_model: str,
    right_model: str,
    *,
    split: str = "test",
) -> dict[str, dict[str, Any]]:
    indexed = comparison.loc[comparison["split"].eq(split)].set_index("model")
    if left_model not in indexed.index or right_model not in indexed.index:
        raise KeyError("Both comparison models must have one row in the requested split")
    result = {}
    for metric in REPORT_METRICS:
        left = float(indexed.loc[left_model, metric])
        right = float(indexed.loc[right_model, metric])
        raw = left - right
        advantage = raw if metric in HIGHER_IS_BETTER else -raw
        result[metric] = {
            "left": left,
            "right": right,
            "raw_left_minus_right": raw,
            "performance_advantage_left": advantage,
            "left_performs_better": bool(advantage > 0),
        }
    return result


def assess_stage_goals(
    baseline_metrics: pd.DataFrame,
    tuned_metrics: pd.DataFrame,
) -> dict[str, Any]:
    baseline = baseline_metrics.set_index("split")
    tuned = tuned_metrics.set_index("split")
    improvement = float(
        baseline.loc["val", "log_loss"] - tuned.loc["val", "log_loss"]
    )
    validation_auc = float(tuned.loc["val", "auc"])
    train_val_auc_gap = float(
        abs(tuned.loc["train", "auc"] - tuned.loc["val", "auc"])
    )
    auc_floor = float(baseline.loc["val", "auc"] - 0.002)
    goals = {
        "validation_log_loss_improvement": {
            "value": improvement,
            "target": 0.0005,
            "direction": "higher",
            "passed": improvement >= 0.0005,
        },
        "validation_auc": {
            "value": validation_auc,
            "target": auc_floor,
            "direction": "higher",
            "passed": validation_auc >= auc_floor,
        },
        "train_validation_auc_gap": {
            "value": train_val_auc_gap,
            "target": 0.03,
            "direction": "lower",
            "passed": train_val_auc_gap <= 0.03,
        },
    }
    return {
        "all_passed": all(item["passed"] for item in goals.values()),
        "passed_count": sum(item["passed"] for item in goals.values()),
        "goal_count": len(goals),
        "goals": goals,
    }


def audit_final_predictions(
    predictions: pd.DataFrame, expected_rows: int
) -> dict[str, Any]:
    probability = predictions["lightgbm_tuned_probability"].to_numpy(dtype=float)
    invalid = int(
        (~np.isfinite(probability)).sum()
        + ((probability < 0) | (probability > 1)).sum()
    )
    duplicate_keys = int(predictions.duplicated(ID_COLUMNS).sum())
    return {
        "passed": len(predictions) == expected_rows and invalid == 0 and duplicate_keys == 0,
        "rows": int(len(predictions)),
        "expected_rows": int(expected_rows),
        "invalid_probability_cells": invalid,
        "duplicate_key_rows": duplicate_keys,
    }


def audit_seed42_replay(
    model: Any,
    val_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    seed_results: pd.DataFrame,
) -> dict[str, Any]:
    x_val, y_val, _ = val_prepared
    probability = np.asarray(model.predict_proba(x_val)[:, 1], dtype=float)
    metrics = probability_metrics(y_val, probability, n_bins=10)
    seed42 = seed_results.loc[seed_results["seed"].eq(42)]
    if len(seed42) != 1:
        return {"passed": False, "reason": "seed 42 row missing or duplicated"}
    differences = {
        metric: abs(float(metrics[metric]) - float(seed42.iloc[0][f"val_{metric}"]))
        for metric in REPORT_METRICS
    }
    maximum = max(differences.values())
    return {
        "passed": maximum <= 1e-12,
        "tolerance": 1e-12,
        "absolute_differences": differences,
        "max_absolute_difference": maximum,
    }


def audit_phase_selections_and_tables(
    tuning_results: pd.DataFrame,
    phase_selections: pd.DataFrame,
    seed_results: pd.DataFrame,
) -> dict[str, Any]:
    tables = {
        "tuning_candidates": validate_validation_only_table(tuning_results),
        "phase_selections": validate_validation_only_table(phase_selections),
        "seed_stability": validate_validation_only_table(seed_results),
    }
    return {
        "passed": all(item["passed"] for item in tables.values()),
        "tables": tables,
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
        ignored_prefix = report_dir.relative_to(project_root).as_posix().rstrip("/") + "/"
    except ValueError:
        ignored_prefix = ""
    ignored = []
    relevant = []
    for line in status:
        changed_path = line[3:].replace("\\", "/")
        if ignored_prefix and changed_path.startswith(ignored_prefix):
            ignored.append(line)
        else:
            relevant.append(line)
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "remote": git("remote", "get-url", "origin"),
        "working_tree_clean_before_report_generation": not relevant,
        "working_tree_status_before_report_generation": relevant,
        "ignored_report_status": ignored,
    }


def _render_report(
    comparison: pd.DataFrame,
    phase_selections: pd.DataFrame,
    seed_results: pd.DataFrame,
    external: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    test = comparison.loc[comparison["split"].eq("test")].set_index("model")
    tuned_vs_m22 = summary["tuned_vs_m22_test"]
    tuned_vs_xgboost = summary["tuned_vs_xgboost_test"]
    lines = [
        "# M23 开局前 LightGBM 控制变量调参报告",
        "",
        "## 阶段决定",
        "",
        f"验收状态：**{summary['acceptance']['status']}**（"
        f"{summary['acceptance']['blocking_passed']}/{summary['acceptance']['blocking_total']}）。",
        f"可以进入 M24 固定模型评估：**{summary['acceptance']['ready_for_m24']}**。",
        "36 个候选和五种子实验只读取 train/validation；test 在参数冻结后评价一次。",
        "",
        "## 固定合同",
        "",
        f"- 样本：{summary['data']['rows']:,}；train/val/test："
        f"{summary['data']['split_rows']['train']:,} / "
        f"{summary['data']['split_rows']['val']:,} / "
        f"{summary['data']['split_rows']['test']:,}。",
        f"- 特征：{summary['features']['raw_count']} 个原始、"
        f"{summary['features']['encoded_count']} 个编码列。",
        f"- 选择：validation Log Loss，最小接受改善 "
        f"{summary['search']['minimum_phase_improvement']:.4f}。",
        f"- 正式模型：seed 42，最佳迭代 {summary['model']['best_iteration']}。",
        "",
        "## 九阶段选择",
        "",
        "| 阶段 | 选择值 | 是否改变 | Validation Log Loss | 接受改善 |",
        "|---|---:|---|---:|---:|",
    ]
    for row in phase_selections.to_dict(orient="records"):
        lines.append(
            f"| `{row['phase']}` | {row['selected_candidate_label']} | "
            f"{row['changed']} | {row['selected_val_log_loss']:.6f} | "
            f"{row['accepted_improvement']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Validation 目标",
            "",
            "| 目标 | 当前 | 门槛 | 通过 |",
            "|---|---:|---:|---|",
        ]
    )
    for name, item in summary["stage_goals"]["goals"].items():
        lines.append(
            f"| {name} | {item['value']:.6f} | {item['target']:.6f} | "
            f"{item['passed']} |"
        )

    lines.extend(
        [
            "",
            "## 五种子稳定性",
            "",
            "| seed | 最佳迭代 | Validation Log Loss | Validation AUC |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in seed_results.to_dict(orient="records"):
        lines.append(
            f"| {int(row['seed'])} | {int(row['best_iteration'])} | "
            f"{row['val_log_loss']:.6f} | {row['val_auc']:.6f} |"
        )
    stability = summary["seed_stability"]
    lines.extend(
        [
            "",
            f"Log Loss 范围 `{stability['val_log_loss_range']:.6f}`，"
            f"AUC 范围 `{stability['val_auc_range']:.6f}`，稳定性通过："
            f"**{stability['passed']}**。",
            "",
            "## 正式测试结果",
            "",
            "| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in (
        "logistic_regression",
        "xgboost_frozen",
        "lightgbm_baseline",
        "lightgbm_tuned",
    ):
        row = test.loc[model_name]
        lines.append(
            f"| `{model_name}` | {row['accuracy']:.6f} | {row['auc']:.6f} | "
            f"{row['log_loss']:.6f} | {row['brier_score']:.6f} | "
            f"{row['ece10']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## M23 与 M22 相差多少",
            "",
            "原始差值为 M23 tuned 减 M22 baseline；方向修正后正值表示 M23 更好。",
            "",
            "| 指标 | 原始差值 | 方向修正后 | M23 更好 |",
            "|---|---:|---:|---|",
        ]
    )
    for metric in REPORT_METRICS:
        item = tuned_vs_m22[metric]
        lines.append(
            f"| {metric} | {item['raw_left_minus_right']:+.6f} | "
            f"{item['performance_advantage_left']:+.6f} | "
            f"{item['left_performs_better']} |"
        )

    lines.extend(
        [
            "",
            "## M23 与冻结 XGBoost 相差多少",
            "",
            "| 指标 | 原始差值 | 方向修正后 |",
            "|---|---:|---:|",
        ]
    )
    for metric in REPORT_METRICS:
        item = tuned_vs_xgboost[metric]
        lines.append(
            f"| {metric} | {item['raw_left_minus_right']:+.6f} | "
            f"{item['performance_advantage_left']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## 最低门槛与更高目标",
            "",
            "| 指标 | 当前 | 最低门槛 | 最低通过 | 更高目标 | 目标通过 | 尚差 |",
            "|---|---:|---:|---|---:|---|---:|",
        ]
    )
    for metric, item in summary["metric_targets"]["metrics"].items():
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
            "| 外部工作 | 指标 | M23 | 外部 | 差值 |",
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
            f"| {row['source_title']} | {row['metric']} | "
            f"{row['current_value']:.6f} | {row['reported_value']:.6f} | {text} |"
        )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这是固定顺序的 greedy search，不是穷举全部参数组合。",
            "- test 相对 M22 的变化不参与参数选择，负结果也必须保留。",
            "- 小数点级差异尚未做置信区间检验，不能直接称为显著提升。",
            "- 公开 DNN 使用不同数据和随机行切分，只能报告数值差。",
            "",
            "## 下一阶段",
            "",
            "M24 不再调参，只对冻结 M23 模型做系列赛 bootstrap、地图和来源稳健性、"
            "校准选择与错误分析。",
            "",
            "复现命令：",
            "",
            "```powershell",
            ".\\scripts\\run_pre_round_lightgbm_tuning.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def run(
    *,
    project_root: str | Path,
    data_path: str | Path = "data/processed/esta_full/pre_round.parquet",
    model_dir: str | Path = "models/esta_full_m23",
    report_dir: str | Path = "reports/esta_full_m23",
    run_tests: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_path = _resolve(root, data_path)
    model_dir = _resolve(root, model_dir)
    report_dir = _resolve(root, report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m22_summary_path = root / "reports/esta_full_m22/m22_summary.json"
    m22_predictions_path = root / "reports/esta_full_m22/m22_test_predictions.csv"
    m22_comparison_path = root / "reports/esta_full_m22/m22_model_comparison.csv"
    m22_columns_path = root / "reports/esta_full_m22/encoded_feature_columns.csv"
    m22_summary = _read_json(m22_summary_path)
    m22_predictions = pd.read_csv(m22_predictions_path)
    m22_comparison = pd.read_csv(m22_comparison_path)
    m22_columns = pd.read_csv(m22_columns_path)["encoded_feature"].tolist()

    data_artifact = fingerprint_file(data_path)
    data = read_table(data_path)
    data_audit = audit_data_contract(data)
    m22_prerequisite = (
        m22_summary.get("acceptance", {}).get("status") == "passed"
        and m22_summary.get("acceptance", {}).get("m22_lightgbm_baseline_complete")
        is True
        and m22_summary.get("data", {}).get("sha256") == data_artifact["sha256"]
        and m22_summary.get("features", {}).get("raw_count") == len(PRE_ROUND_FEATURES)
        and m22_summary.get("features", {}).get("encoded_count") == len(m22_columns)
        and all(
            m22_summary.get("model", {}).get("params", {}).get(name) == value
            for name, value in BASE_TUNING_PARAMS.items()
        )
    )
    expected_split_rows = m22_summary.get("data", {}).get("split_rows", {})
    expected_split_series = m22_summary.get("data", {}).get("split_series", {})
    data_contract = (
        data_audit["passed"]
        and len(data) == 41074
        and data_audit["split_rows"] == expected_split_rows
        and data_audit["split_series"] == expected_split_series
        and data_artifact["sha256"] == m22_summary["data"]["sha256"]
    )

    prepared = prepare_pre_round_splits(data)
    encoded_columns = prepared["train"][0].columns.tolist()
    feature_contract = (
        len(PRE_ROUND_FEATURES) == 36
        and len(encoded_columns) == 43
        and encoded_columns == m22_columns
        and not set(encoded_columns) & set(ID_COLUMNS + ["ct_win", "split"])
    )
    grid_audit = audit_candidate_grid(PHASE_DEFINITIONS, BASE_TUNING_PARAMS)

    tuning_results, phase_selections, frozen_model, frozen_params = (
        run_sequential_search(
            prepared["train"],
            prepared["val"],
            PHASE_DEFINITIONS,
            MINIMUM_PHASE_IMPROVEMENT,
        )
    )
    phase_audit = audit_phase_selections(
        tuning_results, phase_selections, MINIMUM_PHASE_IMPROVEMENT
    )
    seed_results = run_seed_stability(
        prepared["train"], prepared["val"], frozen_params
    )
    stability = assess_seed_stability(seed_results)
    validation_only = audit_phase_selections_and_tables(
        tuning_results, phase_selections, seed_results
    )
    frozen_param_audit = audit_frozen_params(frozen_model, frozen_params)
    seed42_replay = audit_seed42_replay(frozen_model, prepared["val"], seed_results)

    tuned_metrics, tuned_probabilities = evaluate_frozen_model(frozen_model, prepared)
    test_rows = data.loc[data["split"].eq("test")]
    predictions = build_final_prediction_table(
        test_rows, tuned_probabilities["test"], m22_predictions
    )
    prediction_audit = audit_final_predictions(predictions, len(test_rows))

    comparison = pd.concat([m22_comparison, tuned_metrics], ignore_index=True)
    tuned_vs_m22 = metric_differences(
        comparison, "lightgbm_tuned", "lightgbm_baseline"
    )
    tuned_vs_xgboost = metric_differences(
        comparison, "lightgbm_tuned", "xgboost_frozen"
    )
    tuned_vs_logistic = metric_differences(
        comparison, "lightgbm_tuned", "logistic_regression"
    )
    m22_baseline_metrics = m22_comparison.loc[
        m22_comparison["model"].eq("lightgbm_baseline")
    ]
    stage_goals = assess_stage_goals(m22_baseline_metrics, tuned_metrics)
    tuned_test_row = tuned_metrics.loc[tuned_metrics["split"].eq("test")].iloc[0]
    tuned_test_metrics = {
        metric: float(tuned_test_row[metric]) for metric in REPORT_METRICS
    }
    metric_targets = assess_metric_targets(tuned_test_metrics)

    controlled_comparison = (
        prediction_audit["passed"]
        and len(m22_predictions) == len(predictions)
        and encoded_columns == m22_columns
        and set(
            [
                "xgboost_frozen",
                "lightgbm_baseline",
                "lightgbm_tuned",
            ]
        ).issubset(set(comparison["model"]))
    )

    external_benchmarks = pd.read_csv(
        root / "benchmarks/external_round_model_metrics.csv"
    )
    external = compare_benchmarks(tuned_test_metrics, external_benchmarks)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    write_markdown_report(
        external,
        tuned_test_metrics,
        report_dir / "external_benchmark_comparison.md",
        stage_label="M23 LightGBM",
    )
    external_report = (
        not external.empty
        and (report_dir / "external_benchmark_comparison.md").is_file()
    )

    model_path = model_dir / "pre_round_lightgbm_tuned.joblib"
    model_bundle = {
        "model": frozen_model,
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "model_name": "lightgbm_tuned",
        "profile": "M14_pre_round_features",
        "raw_features": list(PRE_ROUND_FEATURES),
        "columns": encoded_columns,
        "params": dict(frozen_params),
        "best_iteration": _best_iteration(frozen_model),
        "data_sha256": data_artifact["sha256"],
        "selection_metric": "validation_log_loss",
        "minimum_phase_improvement": MINIMUM_PHASE_IMPROVEMENT,
        "official_seed": 42,
    }
    joblib.dump(model_bundle, model_path)
    model_artifact = fingerprint_file(model_path)

    evaluation_history = next(iter(frozen_model.evals_result_.values()))
    validation_logloss = evaluation_history.get(
        "binary_logloss", next(iter(evaluation_history.values()))
    )
    training_history = pd.DataFrame(
        {
            "iteration": np.arange(1, len(validation_logloss) + 1),
            "validation_binary_logloss": validation_logloss,
        }
    )

    script_path = root / "scripts/run_pre_round_lightgbm_tuning.ps1"
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

    checks = {
        "m22_prerequisite": m22_prerequisite,
        "data_contract": data_contract,
        "feature_contract": feature_contract,
        "candidate_grid": (
            grid_audit["passed"]
            and grid_audit["phase_count"] == 9
            and grid_audit["candidate_count"] == 36
        ),
        "validation_only": validation_only["passed"],
        "phase_selection": phase_audit["passed"] and phase_audit["phase_count"] == 9,
        "frozen_model": frozen_param_audit["passed"] and seed42_replay["passed"],
        "seed_stability": stability["passed"],
        "final_predictions": prediction_audit["passed"],
        "minimum_metrics": metric_targets["all_minimum_passed"],
        "controlled_comparison": controlled_comparison,
        "external_report": external_report,
        "automated_tests": automated["passed"],
        "reproduction_entrypoint": reproduction_entrypoint,
    }
    acceptance = decide_acceptance(checks)

    tuning_results.to_csv(report_dir / "tuning_candidates.csv", index=False)
    phase_selections.to_csv(report_dir / "phase_selections.csv", index=False)
    seed_results.to_csv(report_dir / "seed_stability.csv", index=False)
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    comparison.to_csv(report_dir / "m23_model_comparison.csv", index=False)
    training_history.to_csv(
        report_dir / "lightgbm_training_history.csv", index=False
    )
    pd.DataFrame({"encoded_feature": encoded_columns}).to_csv(
        report_dir / "encoded_feature_columns.csv", index=False
    )
    pd.DataFrame(
        [
            {"check": name, "passed": passed, "blocking": True}
            for name, passed in checks.items()
        ]
    ).to_csv(report_dir / "m23_checks.csv", index=False)
    write_json(
        {"params": frozen_params, "best_iteration": _best_iteration(frozen_model)},
        report_dir / "frozen_params.json",
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated["output"], encoding="utf-8"
    )

    selected_changes = phase_selections.loc[phase_selections["changed"]]
    summary = {
        "stage": "M23",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "experiment_policy": "validation-only greedy sequential LightGBM tuning",
        "acceptance": acceptance,
        "checks": checks,
        "data": {
            "path": data_path.as_posix(),
            "bytes": data_artifact["bytes"],
            "sha256": data_artifact["sha256"],
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
            "encoded_columns_match_m22": encoded_columns == m22_columns,
        },
        "search": {
            "selection_metric": "validation_log_loss",
            "minimum_phase_improvement": MINIMUM_PHASE_IMPROVEMENT,
            "phase_count": grid_audit["phase_count"],
            "candidate_count": grid_audit["candidate_count"],
            "accepted_change_count": int(len(selected_changes)),
            "accepted_phases": selected_changes["phase"].tolist(),
            "test_columns_in_candidate_tables": validation_only,
            "test_used_for_selection": False,
            "formal_test_evaluations": 1,
        },
        "model": {
            "library": "lightgbm",
            "version": importlib.metadata.version("lightgbm"),
            "official_seed": 42,
            "params": frozen_params,
            "best_iteration": _best_iteration(frozen_model),
            "model_artifact": model_artifact,
        },
        "metrics": tuned_test_metrics,
        "metric_targets": metric_targets,
        "stage_goals": stage_goals,
        "tuned_vs_m22_test": tuned_vs_m22,
        "tuned_vs_xgboost_test": tuned_vs_xgboost,
        "tuned_vs_logistic_test": tuned_vs_logistic,
        "seed_stability": stability,
        "seed42_replay": seed42_replay,
        "prediction_audit": prediction_audit,
        "external_comparison_rows": int(len(external)),
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "lightgbm_version": importlib.metadata.version("lightgbm"),
            "device_type": frozen_model.get_params().get("device_type"),
            "cuda_required": False,
        },
        "automated_tests": {
            "passed": automated["passed"],
            "return_code": automated["return_code"],
            "elapsed_seconds": automated["elapsed_seconds"],
            "test_count": test_count,
            "skipped": not run_tests,
        },
        "next_stage": "M24 frozen LightGBM evaluation, robustness, and calibration",
    }
    manifest = {
        "stage": "M23",
        "generated_at_utc": summary["generated_at_utc"],
        "code": _collect_git_state(root, report_dir),
        "input_artifacts": {
            "pre_round_data": data_artifact,
            "m22_summary": fingerprint_file(m22_summary_path),
            "m22_test_predictions": fingerprint_file(m22_predictions_path),
            "m22_model_comparison": fingerprint_file(m22_comparison_path),
            "m22_encoded_columns": fingerprint_file(m22_columns_path),
            "requirements_lock": fingerprint_file(root / "requirements-lock.txt"),
        },
        "output_artifacts": {"lightgbm_tuned_model": model_artifact},
        "contract": {
            "base_params": BASE_TUNING_PARAMS,
            "frozen_params": frozen_params,
            "phase_definitions": PHASE_DEFINITIONS,
            "minimum_phase_improvement": MINIMUM_PHASE_IMPROVEMENT,
            "stability_seeds": STABILITY_SEEDS,
            "stability_limits": STABILITY_LIMITS,
            "encoded_columns": encoded_columns,
            "test_use": "once after parameters and seed protocol were frozen",
        },
        "checks": checks,
        "acceptance": acceptance,
    }
    write_json(summary, report_dir / "m23_summary.json")
    write_json(manifest, report_dir / "m23_experiment_manifest.json")
    (report_dir / "m23_pre_round_lightgbm_tuning_report.md").write_text(
        _render_report(comparison, phase_selections, seed_results, external, summary),
        encoding="utf-8",
    )
    if acceptance["status"] != "passed":
        raise RuntimeError(
            "M23 acceptance failed: " + ", ".join(acceptance["blocking_failures"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M23 validation-only pre-round LightGBM tuning."
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data", default="data/processed/esta_full/pre_round.parquet")
    parser.add_argument("--model-dir", default="models/esta_full_m23")
    parser.add_argument("--report-dir", default="reports/esta_full_m23")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    summary = run(
        project_root=args.project_root,
        data_path=args.data,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        run_tests=not args.skip_tests,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

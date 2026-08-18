from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .io import read_table
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m16_first_kill_baselines import (
    assess_metric_targets,
    audit_training_data,
    canonical_feature_names,
    compare_external_models,
    model_metric_differences,
    prepare_profile_splits,
    write_json,
)
from .schema import ID_COLUMNS
from .metrics import probability_metrics
from .train_xgb import make_model


MINIMUM_PHASE_IMPROVEMENT = 0.0001
STABILITY_SEEDS = (42, 43, 44, 45, 46)
STABILITY_LIMITS = {"val_log_loss": 0.002, "val_auc": 0.003}

BASE_TUNING_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_child_weight": 1,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "early_stopping_rounds": None,
    "random_state": 42,
}

PHASE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "tree_policy",
        "allowed_parameters": ("n_estimators", "early_stopping_rounds"),
        "candidates": (
            {
                "id": "tree_fixed_500",
                "label": "fixed_500",
                "overrides": {"n_estimators": 500, "early_stopping_rounds": None},
            },
            {
                "id": "tree_cap_1500_es50",
                "label": "cap_1500_es50",
                "overrides": {"n_estimators": 1500, "early_stopping_rounds": 50},
            },
            {
                "id": "tree_cap_3000_es100",
                "label": "cap_3000_es100",
                "overrides": {"n_estimators": 3000, "early_stopping_rounds": 100},
            },
        ),
    },
    {
        "name": "max_depth",
        "allowed_parameters": ("max_depth",),
        "candidates": tuple(
            {"id": f"max_depth_{value}", "label": str(value), "overrides": {"max_depth": value}}
            for value in (2, 3, 4, 5, 6)
        ),
    },
    {
        "name": "min_child_weight",
        "allowed_parameters": ("min_child_weight",),
        "candidates": tuple(
            {
                "id": f"min_child_weight_{value}",
                "label": str(value),
                "overrides": {"min_child_weight": value},
            }
            for value in (1, 3, 5, 10, 20)
        ),
    },
    {
        "name": "reg_lambda",
        "allowed_parameters": ("reg_lambda",),
        "candidates": tuple(
            {"id": f"reg_lambda_{value}", "label": str(value), "overrides": {"reg_lambda": value}}
            for value in (0.5, 1, 3, 5, 10)
        ),
    },
    {
        "name": "reg_alpha",
        "allowed_parameters": ("reg_alpha",),
        "candidates": tuple(
            {"id": f"reg_alpha_{value}", "label": str(value), "overrides": {"reg_alpha": value}}
            for value in (0, 0.05, 0.1, 0.5, 1)
        ),
    },
    {
        "name": "subsample",
        "allowed_parameters": ("subsample",),
        "candidates": tuple(
            {"id": f"subsample_{value}", "label": str(value), "overrides": {"subsample": value}}
            for value in (0.7, 0.8, 0.85, 0.9, 1.0)
        ),
    },
    {
        "name": "colsample_bytree",
        "allowed_parameters": ("colsample_bytree",),
        "candidates": tuple(
            {
                "id": f"colsample_bytree_{value}",
                "label": str(value),
                "overrides": {"colsample_bytree": value},
            }
            for value in (0.6, 0.7, 0.8, 0.85, 0.9, 1.0)
        ),
    },
    {
        "name": "learning_rate",
        "allowed_parameters": ("learning_rate",),
        "candidates": tuple(
            {
                "id": f"learning_rate_{value}",
                "label": str(value),
                "overrides": {"learning_rate": value},
            }
            for value in (0.01, 0.02, 0.03, 0.05, 0.1)
        ),
    },
)

BLOCKING_CHECKS = (
    "m16_prerequisite",
    "data_contract",
    "feature_contract",
    "candidate_grid",
    "validation_only",
    "phase_selection",
    "frozen_model",
    "seed_stability",
    "final_predictions",
    "minimum_metrics",
    "external_report",
    "automated_tests",
)
REPORT_METRICS = ("accuracy", "auc", "log_loss", "brier_score", "ece10")


def audit_candidate_grid(
    phases: tuple[dict[str, Any], ...], base_params: dict[str, Any]
) -> dict[str, Any]:
    violations: list[str] = []
    candidate_ids: list[str] = []
    changed_by_phase: dict[str, list[str]] = {}

    for phase in phases:
        name = str(phase["name"])
        allowed = set(phase["allowed_parameters"])
        candidates = phase.get("candidates", ())
        if not candidates:
            violations.append(f"{name}: no candidates")
            continue
        changed_union: set[str] = set()
        resolved_allowed_values = []
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            candidate_ids.append(candidate_id)
            overrides = dict(candidate["overrides"])
            extra = set(overrides) - allowed
            if extra:
                violations.append(f"{candidate_id}: changes forbidden parameters {sorted(extra)}")
            changed_union.update(
                key for key, value in overrides.items() if base_params.get(key) != value
            )
            resolved_allowed_values.append(
                tuple(overrides.get(key, base_params.get(key)) for key in sorted(allowed))
            )
        if len(resolved_allowed_values) != len(set(resolved_allowed_values)):
            violations.append(f"{name}: duplicate candidate parameter values")
        incumbent = tuple(base_params.get(key) for key in sorted(allowed))
        if incumbent not in resolved_allowed_values:
            violations.append(f"{name}: base incumbent is absent")
        changed_by_phase[name] = sorted(changed_union)
        if name == "tree_policy":
            if allowed != {"n_estimators", "early_stopping_rounds"}:
                violations.append("tree_policy: allowed parameter contract changed")
        elif len(allowed) != 1:
            violations.append(f"{name}: must allow exactly one parameter")

    duplicate_ids = sorted(
        candidate_id for candidate_id in set(candidate_ids) if candidate_ids.count(candidate_id) > 1
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
    if not np.isfinite(results["val_log_loss"].to_numpy(dtype=float)).all():
        raise ValueError("Validation Log Loss values must be finite")

    incumbent = incumbents.iloc[0]
    best = results.sort_values(
        ["val_log_loss", "candidate_order"], kind="stable"
    ).iloc[0]
    best_improvement = float(incumbent["val_log_loss"] - best["val_log_loss"])
    changed = best_improvement >= minimum_improvement
    selected = best if changed else incumbent
    accepted_improvement = float(
        incumbent["val_log_loss"] - selected["val_log_loss"]
    )
    return {
        "candidate_id": str(selected["candidate_id"]),
        "changed": bool(changed),
        "incumbent_id": str(incumbent["candidate_id"]),
        "best_observed_id": str(best["candidate_id"]),
        "best_observed_improvement": best_improvement,
        "accepted_improvement": accepted_improvement,
        "selected_val_log_loss": float(selected["val_log_loss"]),
    }


def validate_tuning_table(results: pd.DataFrame) -> dict[str, Any]:
    forbidden = sorted(column for column in results.columns if column.startswith("test_"))
    return {
        "passed": not forbidden,
        "rows": int(len(results)),
        "forbidden_columns": forbidden,
    }


def audit_frozen_params(model: Any, expected: dict[str, Any]) -> dict[str, Any]:
    actual = model.get_params()
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    return {"passed": not mismatches, "expected": expected, "mismatches": mismatches}


def assess_seed_stability(seed_results: pd.DataFrame) -> dict[str, Any]:
    required = {"seed", "val_log_loss", "val_auc"}
    missing = sorted(required - set(seed_results.columns))
    if missing:
        raise KeyError(f"Seed stability results are missing columns: {missing}")
    unexpected_test_columns = sorted(
        column for column in seed_results.columns if column.startswith("test_")
    )
    observed_seeds = tuple(sorted(seed_results["seed"].astype(int).tolist()))
    log_loss_range = float(
        seed_results["val_log_loss"].max() - seed_results["val_log_loss"].min()
    )
    auc_range = float(seed_results["val_auc"].max() - seed_results["val_auc"].min())
    passed = (
        observed_seeds == STABILITY_SEEDS
        and not unexpected_test_columns
        and log_loss_range <= STABILITY_LIMITS["val_log_loss"]
        and auc_range <= STABILITY_LIMITS["val_auc"]
    )
    return {
        "passed": passed,
        "observed_seeds": list(observed_seeds),
        "forbidden_columns": unexpected_test_columns,
        "val_log_loss_range": log_loss_range,
        "val_auc_range": auc_range,
        "limits": STABILITY_LIMITS,
    }


def build_final_prediction_table(
    test_rows: pd.DataFrame,
    tuned_probabilities: np.ndarray,
    m16_predictions: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ID_COLUMNS + ["ct_win"]
    missing = sorted(set(key_columns) - set(test_rows.columns))
    if missing:
        raise KeyError(f"Test rows are missing identity columns: {missing}")
    missing_m16 = sorted(set(key_columns) - set(m16_predictions.columns))
    if missing_m16:
        raise KeyError(f"M16 predictions are missing identity columns: {missing_m16}")

    current_keys = test_rows[key_columns].reset_index(drop=True)
    m16_keys = m16_predictions[key_columns].reset_index(drop=True)
    if not current_keys.equals(m16_keys):
        raise ValueError("M17 test rows do not match the exact M16 test keys")

    probabilities = np.asarray(tuned_probabilities, dtype=float).reshape(-1)
    if len(probabilities) != len(current_keys):
        raise ValueError("M17 tuned probability row count differs from test rows")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("M17 tuned probabilities must be finite and between 0 and 1")

    result = m16_predictions.reset_index(drop=True).copy()
    result["xgboost_tuned_probability"] = probabilities
    result["xgboost_tuned_prediction"] = (probabilities >= 0.5).astype(int)
    return result


def decide_acceptance(checks: dict[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not checks.get(name, False)]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "ready_for_m18": not failures,
    }


def _best_iteration(model: Any) -> int | None:
    try:
        return int(model.best_iteration)
    except (AttributeError, TypeError):
        return None


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
    params: dict[str, Any],
) -> tuple[Any, dict[str, float], float]:
    x_train, y_train, _ = train_prepared
    x_val, y_val, _ = val_prepared
    model = make_model(task="first_kill", **params)
    started = perf_counter()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    elapsed_seconds = perf_counter() - started
    return model, _candidate_metrics(model, train_prepared, val_prepared), elapsed_seconds


def _is_incumbent_candidate(
    candidate: dict[str, Any], current_params: dict[str, Any], allowed: tuple[str, ...]
) -> bool:
    overrides = candidate["overrides"]
    return all(overrides.get(name, current_params.get(name)) == current_params.get(name) for name in allowed)


def run_sequential_search(
    train_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    val_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    phases: tuple[dict[str, Any], ...],
    minimum_improvement: float,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, dict[str, Any]]:
    grid_audit = audit_candidate_grid(phases, BASE_TUNING_PARAMS)
    if not grid_audit["passed"]:
        raise ValueError(f"Invalid M17 candidate grid: {grid_audit['violations']}")

    current_params = dict(BASE_TUNING_PARAMS)
    all_results: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    selected_model: Any = None

    for phase_order, phase in enumerate(phases, start=1):
        phase_name = str(phase["name"])
        allowed = tuple(phase["allowed_parameters"])
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
                        candidate, current_params, allowed
                    ),
                    "changed_parameters": ",".join(allowed),
                    "parameter_values": json.dumps(
                        {name: params[name] for name in allowed}, sort_keys=True
                    ),
                    "best_iteration": best_iteration,
                    "best_tree_count": (
                        best_iteration + 1
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
                "selected_val_auc": float(selected_row["val_auc"]),
                "selected_val_brier_score": float(selected_row["val_brier_score"]),
                "selected_best_tree_count": int(selected_row["best_tree_count"]),
                "selected_params": json.dumps(current_params, sort_keys=True),
            }
        )
        all_results.append(phase_results)

    if selected_model is None:
        raise ValueError("M17 requires at least one tuning phase")
    results = pd.concat(all_results, ignore_index=True)
    validation_audit = validate_tuning_table(results)
    if not validation_audit["passed"]:
        raise RuntimeError("M17 tuning results unexpectedly contain test metrics")
    return results, pd.DataFrame(selections), selected_model, current_params


def audit_phase_selections(
    tuning_results: pd.DataFrame,
    phase_selections: pd.DataFrame,
    minimum_improvement: float,
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
        selected_rows = rows.loc[rows["selected"]]
        if len(selected_rows) != 1 or selected_rows.iloc[0]["candidate_id"] != expected["candidate_id"]:
            failures.append(f"{phase_name}: candidate table selection marker differs")
    return {
        "passed": not failures,
        "phase_count": int(tuning_results["phase"].nunique()),
        "selection_rows": int(len(phase_selections)),
        "failures": failures,
    }


def run_seed_stability(
    train_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    val_prepared: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    frozen_params: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for seed in STABILITY_SEEDS:
        params = {**frozen_params, "random_state": seed}
        model, metrics, elapsed_seconds = _fit_candidate(
            train_prepared, val_prepared, params
        )
        best_iteration = _best_iteration(model)
        rows.append(
            {
                "seed": seed,
                "best_iteration": best_iteration,
                "best_tree_count": (
                    best_iteration + 1
                    if best_iteration is not None
                    else int(params["n_estimators"])
                ),
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
                "model": "xgboost_tuned",
                "profile": "canonical_event",
                "split": split,
                **probability_metrics(y, probability, n_bins=10),
            }
        )
    return pd.DataFrame(rows), probabilities


def verify_m16_prerequisite(
    data_path: str | Path,
    m16_summary: dict[str, Any],
) -> dict[str, Any]:
    actual = fingerprint_file(data_path)
    expected_data = m16_summary.get("data", {})
    acceptance = m16_summary.get("acceptance", {})
    passed = (
        acceptance.get("status") == "passed"
        and acceptance.get("ready_for_m17") is True
        and expected_data.get("sha256") == actual["sha256"]
    )
    return {
        "passed": passed,
        "m16_status": acceptance.get("status"),
        "m16_ready_for_m17": acceptance.get("ready_for_m17"),
        "expected_sha256": expected_data.get("sha256"),
        "actual_sha256": actual["sha256"],
        "actual_bytes": actual["bytes"],
    }


def assess_stage_goals(
    tuned_metrics: pd.DataFrame,
    m16_comparison: pd.DataFrame,
) -> dict[str, Any]:
    tuned = tuned_metrics.set_index("split")
    baseline = m16_comparison.loc[
        m16_comparison["model"].eq("xgboost_untuned")
    ].set_index("split")
    validation_log_loss_improvement = float(
        baseline.loc["val", "log_loss"] - tuned.loc["val", "log_loss"]
    )
    validation_auc = float(tuned.loc["val", "auc"])
    train_val_auc_gap = float(tuned.loc["train", "auc"] - validation_auc)
    goals = {
        "validation_log_loss_improvement": {
            "value": validation_log_loss_improvement,
            "target": 0.001,
            "higher_is_better": True,
            "passed": validation_log_loss_improvement >= 0.001,
        },
        "validation_auc": {
            "value": validation_auc,
            "target": 0.800069,
            "higher_is_better": True,
            "passed": validation_auc >= 0.800069,
        },
        "train_validation_auc_gap": {
            "value": train_val_auc_gap,
            "target": 0.030,
            "higher_is_better": False,
            "passed": train_val_auc_gap <= 0.030,
        },
    }
    return {
        "all_passed": all(item["passed"] for item in goals.values()),
        "goals": goals,
    }


def audit_final_predictions(
    predictions: pd.DataFrame, expected_rows: int
) -> dict[str, Any]:
    probability = predictions["xgboost_tuned_probability"].to_numpy(dtype=float)
    invalid = int((~np.isfinite(probability)).sum() + ((probability < 0) | (probability > 1)).sum())
    duplicate_keys = int(predictions.duplicated(ID_COLUMNS).sum())
    return {
        "passed": len(predictions) == expected_rows and invalid == 0 and duplicate_keys == 0,
        "rows": int(len(predictions)),
        "expected_rows": int(expected_rows),
        "invalid_probability_rows": invalid,
        "duplicate_key_rows": duplicate_keys,
    }


def render_external_report(comparison: pd.DataFrame) -> str:
    lines = [
        "# M17 首杀后调优模型外部指标对照",
        "",
        "差值统一为“我们的指标 - 外部指标”。Accuracy/AUC 同时换算为百分点。",
        "这些工作使用不同数据、切分和预测时点，因此表格只比较数值，不构成模型排名。",
        "",
        "| 可比性 | 本地模型 | 外部工作 | 指标 | 我们 | 外部 | 差值 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        title = row.get("source_title", row["benchmark_id"])
        url = row.get("source_url", "")
        source = f"[{title}]({url})" if url else title
        difference = float(row["raw_difference_ours_minus_reported"])
        difference_text = (
            f"{difference * 100:+.2f} 个百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| {row.get('comparability', '')} | `{row['current_model']}` | {source} | "
            f"{row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {difference_text} |"
        )
    lines.extend(
        [
            "",
            "`closest_task` 只表示预测时点接近；`partial` 和 `not_comparable` 不用于",
            "判断模型优劣。完整限制记录在 benchmark registry 的 `notes` 字段。",
            "",
        ]
    )
    return "\n".join(lines)


def render_m17_report(
    phase_selections: pd.DataFrame,
    seed_stability: pd.DataFrame,
    comparison: pd.DataFrame,
    external: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    test = comparison.loc[comparison["split"].eq("test")].set_index("model")
    goals = summary["stage_goals"]["goals"]
    tuned_vs_untuned = summary["tuned_vs_untuned_test"]
    tuned_vs_logistic = summary["tuned_vs_logistic_test"]
    lines = [
        "# M17 首杀后 XGBoost 控制变量调参报告",
        "",
        "## 阶段决定",
        "",
        f"验收状态：**{summary['acceptance']['status']}**。",
        f"可以进入 M18：**{summary['acceptance']['ready_for_m18']}**。",
        "39 个候选只使用 validation Log Loss 选择；最终参数冻结后才评价 test。",
        "",
        "## 八阶段选择",
        "",
        "| 阶段 | 入选候选 | 是否改变 | 接受改善 | Validation Log Loss | Validation AUC | 树数 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in phase_selections.to_dict(orient="records"):
        lines.append(
            f"| {row['phase']} | `{row['candidate_id']}` | {row['changed']} | "
            f"{row['accepted_improvement']:+.6f} | {row['selected_val_log_loss']:.6f} | "
            f"{row['selected_val_auc']:.6f} | {int(row['selected_best_tree_count'])} |"
        )

    lines.extend(
        [
            "",
            "## 预先目标",
            "",
            "| 目标 | 当前 | 门槛 | 通过 |",
            "|---|---:|---:|---|",
        ]
    )
    for name, item in goals.items():
        lines.append(
            f"| {name} | {item['value']:.6f} | {item['target']:.6f} | {item['passed']} |"
        )

    lines.extend(
        [
            "",
            "## 最终测试结果",
            "",
            "| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in (
        "constant_train_prior",
        "logistic_regression",
        "xgboost_untuned",
        "xgboost_tuned",
    ):
        row = test.loc[model_name]
        lines.append(
            f"| `{model_name}` | {row['accuracy']:.6f} | {row['auc']:.6f} | "
            f"{row['log_loss']:.6f} | {row['brier_score']:.6f} | {row['ece10']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 与 M16 相差多少",
            "",
            "性能优势已按指标方向换算，正数表示 M17 更好。",
            "",
            "| 对照 | Accuracy | AUC | Log Loss | Brier | ECE10 |",
            "|---|---:|---:|---:|---:|---:|",
            "| M17 vs M16 XGBoost | "
            + " | ".join(
                f"{tuned_vs_untuned[name]['performance_advantage_left']:+.6f}"
                for name in REPORT_METRICS
            )
            + " |",
            "| M17 vs Logistic | "
            + " | ".join(
                f"{tuned_vs_logistic[name]['performance_advantage_left']:+.6f}"
                for name in REPORT_METRICS
            )
            + " |",
            "",
            "## 随机种子稳定性",
            "",
            f"Validation Log Loss 最大差：`{summary['seed_stability']['val_log_loss_range']:.6f}`；"
            f"AUC 最大差：`{summary['seed_stability']['val_auc_range']:.6f}`。",
            "种子实验没有读取 test 指标。",
            "",
            "| Seed | Validation Log Loss | Validation AUC | 树数 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in seed_stability.to_dict(orient="records"):
        lines.append(
            f"| {int(row['seed'])} | {row['val_log_loss']:.6f} | "
            f"{row['val_auc']:.6f} | {int(row['best_tree_count'])} |"
        )

    lines.extend(
        [
            "",
            "## 与外部模型相差多少",
            "",
            "| 本地模型 | 外部工作 | 指标 | 我们 | 外部 | 差值 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in external.to_dict(orient="records"):
        if row["comparability"] not in {"closest_task", "not_comparable"}:
            continue
        difference = float(row["raw_difference_ours_minus_reported"])
        difference_text = (
            f"{difference * 100:+.2f} 个百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| `{row['current_model']}` | {row['source_title']} | {row['metric']} | "
            f"{row['current_value']:.6f} | {row['reported_value']:.6f} | {difference_text} |"
        )

    lines.extend(
        [
            "",
            "外部数据、特征、预测时点和切分均不同，上表不能解释为模型本身更优。",
            "完整来源和 freeze-time 参考见 `external_benchmark_comparison.md`。",
            "",
            "## 结论与下一阶段",
            "",
            f"部署记录建议：`{summary['deployment_recommendation']}`。",
            "M18 在不再调参的前提下，对冻结模型执行系列赛 bootstrap、分地图、",
            "LAN/online 稳健性和概率校准诊断。",
            "",
            "复现命令：",
            "",
            "```powershell",
            ".\\scripts\\run_first_kill_tuning.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _save_model_bundle(
    model: Any,
    path: Path,
    raw_features: list[str],
    encoded_columns: list[str],
    data_sha256: str,
    frozen_params: dict[str, Any],
) -> dict[str, Any]:
    best_iteration = _best_iteration(model)
    bundle = {
        "model": model,
        "task": "first_kill",
        "model_name": "xgboost_tuned",
        "profile": "canonical_event",
        "raw_features": raw_features,
        "columns": encoded_columns,
        "data_sha256": data_sha256,
        "frozen_params": frozen_params,
        "best_iteration": best_iteration,
        "best_tree_count": best_iteration + 1 if best_iteration is not None else frozen_params["n_estimators"],
    }
    joblib.dump(bundle, path)
    return fingerprint_file(path)


def run(
    data_path: str | Path,
    m16_summary_path: str | Path,
    m16_comparison_path: str | Path,
    m16_predictions_path: str | Path,
    m16_encoded_columns_path: str | Path,
    benchmarks_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path,
    project_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_path = Path(data_path)
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    project_root = Path(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m16_summary = _read_json(m16_summary_path)
    prerequisite = verify_m16_prerequisite(data_path, m16_summary)
    if not prerequisite["passed"]:
        raise RuntimeError("M17 input does not match the accepted M16 prerequisite")

    data = read_table(data_path)
    data_audit = audit_training_data(data)
    expected_rows = int(m16_summary["data"]["rows"])
    expected_split_rows = {
        name: int(count) for name, count in m16_summary["data"]["split_rows"].items()
    }
    counts_match = (
        len(data) == expected_rows
        and data_audit.get("split_rows", {}) == expected_split_rows
    )
    if not data_audit["passed"] or not counts_match:
        raise RuntimeError("M17 data identity or split counts differ from M16")

    raw_features = canonical_feature_names()
    if raw_features != m16_summary["features"]["raw_features"]:
        raise RuntimeError("M17 raw feature contract differs from M16")
    prepared = prepare_profile_splits(data, raw_features)
    encoded_columns = prepared["train"][0].columns.tolist()
    expected_encoded_columns = pd.read_csv(m16_encoded_columns_path)[
        "encoded_feature"
    ].tolist()
    feature_contract_passed = encoded_columns == expected_encoded_columns
    if not feature_contract_passed:
        raise RuntimeError("M17 encoded feature columns differ from M16")

    grid_audit = audit_candidate_grid(PHASE_DEFINITIONS, BASE_TUNING_PARAMS)
    tuning_results, phase_selections, frozen_model, frozen_params = run_sequential_search(
        train_prepared=prepared["train"],
        val_prepared=prepared["val"],
        phases=PHASE_DEFINITIONS,
        minimum_improvement=MINIMUM_PHASE_IMPROVEMENT,
    )
    validation_audit = validate_tuning_table(tuning_results)
    selection_validation_audit = validate_tuning_table(phase_selections)
    selection_audit = audit_phase_selections(
        tuning_results, phase_selections, MINIMUM_PHASE_IMPROVEMENT
    )
    frozen_audit = audit_frozen_params(frozen_model, frozen_params)

    seed_results = run_seed_stability(
        prepared["train"], prepared["val"], frozen_params
    )
    seed_audit = assess_seed_stability(seed_results)

    # Test is first evaluated here, after the full parameter and seed protocol is frozen.
    tuned_metrics, tuned_probabilities = evaluate_frozen_model(frozen_model, prepared)
    m16_predictions = pd.read_csv(m16_predictions_path)
    test_rows = data.loc[data["split"].eq("test")]
    predictions = build_final_prediction_table(
        test_rows, tuned_probabilities["test"], m16_predictions
    )
    prediction_audit = audit_final_predictions(
        predictions, expected_split_rows["test"]
    )

    m16_comparison = pd.read_csv(m16_comparison_path)
    comparison = pd.concat([m16_comparison, tuned_metrics], ignore_index=True)
    stage_goals = assess_stage_goals(tuned_metrics, m16_comparison)
    test_metrics_row = tuned_metrics.loc[tuned_metrics["split"].eq("test")].iloc[0]
    tuned_test_metrics = {
        name: float(test_metrics_row[name]) for name in REPORT_METRICS
    }
    minimum_metrics = assess_metric_targets(tuned_test_metrics)
    tuned_vs_untuned = model_metric_differences(
        comparison, "xgboost_tuned", "xgboost_untuned"
    )
    tuned_vs_logistic = model_metric_differences(
        comparison, "xgboost_tuned", "logistic_regression"
    )

    external_benchmarks = pd.read_csv(benchmarks_path)
    external = compare_external_models(comparison, external_benchmarks)
    external_report = render_external_report(external)
    external_report_passed = not external.empty and bool(external_report.strip())

    automated_tests = run_automated_tests(project_root)
    test_count_match = re.search(r"Ran (\d+) tests?", automated_tests["output"])
    automated_test_count = int(test_count_match.group(1)) if test_count_match else None

    validation_only_passed = (
        validation_audit["passed"] and selection_validation_audit["passed"]
    )
    candidate_grid_passed = (
        grid_audit["passed"]
        and grid_audit["candidate_count"] == 39
        and grid_audit["phase_count"] == 8
    )
    phase_selection_passed = (
        selection_audit["passed"]
        and selection_audit["phase_count"] == 8
        and selection_audit["selection_rows"] == 8
    )
    checks = {
        "m16_prerequisite": prerequisite["passed"],
        "data_contract": data_audit["passed"] and counts_match,
        "feature_contract": feature_contract_passed,
        "candidate_grid": candidate_grid_passed,
        "validation_only": validation_only_passed,
        "phase_selection": phase_selection_passed,
        "frozen_model": frozen_audit["passed"],
        "seed_stability": seed_audit["passed"],
        "final_predictions": prediction_audit["passed"],
        "minimum_metrics": minimum_metrics["all_minimum_passed"],
        "external_report": external_report_passed,
        "automated_tests": automated_tests["passed"],
    }
    acceptance = decide_acceptance(checks)

    model_path = model_dir / "first_kill_xgboost_tuned.joblib"
    model_artifact = _save_model_bundle(
        frozen_model,
        model_path,
        raw_features,
        encoded_columns,
        prerequisite["actual_sha256"],
        frozen_params,
    )

    tuning_results.to_csv(report_dir / "controlled_tuning_results.csv", index=False)
    phase_selections.to_csv(report_dir / "phase_selections.csv", index=False)
    seed_results.to_csv(report_dir / "seed_stability.csv", index=False)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    (report_dir / "external_benchmark_comparison.md").write_text(
        external_report, encoding="utf-8"
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests["output"], encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"check": name, "passed": passed, "blocking": True}
            for name, passed in checks.items()
        ]
    ).to_csv(report_dir / "m17_checks.csv", index=False)

    evaluation_history = frozen_model.evals_result()["validation_0"]["logloss"]
    pd.DataFrame(
        {"iteration": range(len(evaluation_history)), "val_log_loss": evaluation_history}
    ).to_csv(report_dir / "final_training_history.csv", index=False)

    baseline_test_log_loss = tuned_vs_untuned["log_loss"]["right"]
    deployment_recommendation = (
        "M17 tuned XGBoost"
        if tuned_test_metrics["log_loss"] <= baseline_test_log_loss
        else "retain M16 untuned XGBoost"
    )
    best_iteration = _best_iteration(frozen_model)
    summary = {
        "stage": "M17",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "selection_metric": "validation log loss",
        "minimum_phase_improvement": MINIMUM_PHASE_IMPROVEMENT,
        "test_use": "evaluated once after parameters and seed protocol were frozen",
        "acceptance": acceptance,
        "checks": checks,
        "prerequisite": prerequisite,
        "data": {
            "path": data_path.as_posix(),
            "sha256": prerequisite["actual_sha256"],
            "rows": int(len(data)),
            "split_rows": data_audit["split_rows"],
            "series": int(data["series_id"].nunique()),
            "games": int(data["game_id"].nunique()),
        },
        "features": {
            "raw_count": len(raw_features),
            "encoded_count": len(encoded_columns),
            "raw_features": raw_features,
            "matches_m16": feature_contract_passed,
        },
        "tuning": {
            "phase_count": grid_audit["phase_count"],
            "candidate_count": grid_audit["candidate_count"],
            "elapsed_seconds": float(tuning_results["elapsed_seconds"].sum()),
            "selected_candidates": phase_selections[
                ["phase", "candidate_id", "changed", "accepted_improvement"]
            ].to_dict(orient="records"),
        },
        "frozen_model": {
            "params": frozen_params,
            "best_iteration": best_iteration,
            "best_tree_count": (
                best_iteration + 1
                if best_iteration is not None
                else frozen_params["n_estimators"]
            ),
            "artifact": model_artifact,
            "parameter_audit": frozen_audit,
        },
        "metrics": tuned_test_metrics,
        "stage_goals": stage_goals,
        "minimum_metrics": minimum_metrics,
        "tuned_vs_untuned_test": tuned_vs_untuned,
        "tuned_vs_logistic_test": tuned_vs_logistic,
        "seed_stability": seed_audit,
        "prediction_audit": prediction_audit,
        "candidate_grid_audit": grid_audit,
        "phase_selection_audit": selection_audit,
        "external_comparison_rows": int(len(external)),
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_test_count,
        },
        "deployment_recommendation": deployment_recommendation,
        "next_stage": "M18 fixed-model evaluation, robustness, and calibration diagnosis",
    }
    write_json(summary, report_dir / "m17_summary.json")
    (report_dir / "m17_first_kill_tuning_report.md").write_text(
        render_m17_report(
            phase_selections, seed_results, comparison, external, summary
        ),
        encoding="utf-8",
    )
    return comparison, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M17 validation-only controlled first-kill XGBoost tuning."
    )
    parser.add_argument(
        "--data", default="data/processed/esta_full/first_kill.parquet"
    )
    parser.add_argument(
        "--m16-summary", default="reports/esta_full_m16/m16_summary.json"
    )
    parser.add_argument(
        "--m16-comparison",
        default="reports/esta_full_m16/m16_model_comparison.csv",
    )
    parser.add_argument(
        "--m16-predictions", default="reports/esta_full_m16/test_predictions.csv"
    )
    parser.add_argument(
        "--m16-encoded-columns",
        default="reports/esta_full_m16/encoded_feature_columns.csv",
    )
    parser.add_argument(
        "--benchmarks", default="benchmarks/external_first_kill_tuned_metrics.csv"
    )
    parser.add_argument("--model-dir", default="models/esta_full_m17")
    parser.add_argument("--report-dir", default="reports/esta_full_m17")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    comparison, summary = run(
        data_path=args.data,
        m16_summary_path=args.m16_summary,
        m16_comparison_path=args.m16_comparison,
        m16_predictions_path=args.m16_predictions,
        m16_encoded_columns_path=args.m16_encoded_columns,
        benchmarks_path=args.benchmarks,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        project_root=args.project_root,
    )
    print(
        comparison.loc[
            comparison["model"].eq("xgboost_tuned")
        ].round(6).to_string(index=False)
    )
    print(
        f"M17 {summary['acceptance']['status']}; "
        f"ready_for_m18={summary['acceptance']['ready_for_m18']}"
    )
    if not summary["acceptance"]["ready_for_m18"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

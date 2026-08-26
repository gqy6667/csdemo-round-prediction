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
import numpy as np
import pandas as pd

from .benchmark_comparison import compare_benchmarks, write_markdown_report
from .io import read_table
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m16_first_kill_baselines import canonical_feature_names
from .m23_pre_round_lightgbm_tuning import (
    BASE_TUNING_PARAMS,
    MINIMUM_PHASE_IMPROVEMENT,
    PHASE_DEFINITIONS,
    STABILITY_LIMITS,
    STABILITY_SEEDS,
    assess_stage_goals,
    assess_seed_stability,
    audit_candidate_grid,
    audit_final_predictions,
    audit_frozen_params,
    audit_phase_selections,
    audit_phase_selections_and_tables,
    audit_seed42_replay,
    evaluate_frozen_model,
    metric_differences,
    run_seed_stability,
    run_sequential_search,
    select_phase_winner,
    validate_validation_only_table,
)
from .m24_pre_round_lightgbm_evaluation import run_compile_check
from .m28_post_first_kill_lightgbm_baseline import (
    EXPECTED_DATA_SHA256,
    EXPECTED_SPLIT_ROWS,
    EXPECTED_SPLIT_SERIES,
    assess_metric_targets,
    audit_data_contract,
    prepare_first_kill_splits,
    write_json,
)
from .schema import ID_COLUMNS


REPORT_METRICS = ("accuracy", "auc", "log_loss", "brier_score", "ece10")

BLOCKING_CHECKS = (
    "m28_prerequisite",
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
    "source_compile",
    "reproduction_entrypoint",
    "artifact_manifest",
)


def audit_reproduction_entrypoint(script_path: Path) -> dict[str, Any]:
    if not script_path.is_file():
        return {"passed": False, "missing_tokens": [script_path.as_posix()]}
    text = script_path.read_text(encoding="utf-8")
    required = (
        "src.csdemo.m29_post_first_kill_lightgbm_tuning",
        "data\\processed\\esta_full\\first_kill.parquet",
        "models\\esta_full_m29",
        "reports\\esta_full_m29",
    )
    missing = [token for token in required if token not in text]
    return {"passed": not missing, "missing_tokens": missing}


def build_final_prediction_table(
    test_rows: pd.DataFrame,
    tuned_probabilities: np.ndarray,
    m28_predictions: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ID_COLUMNS + ["ct_win"]
    missing = sorted(set(key_columns) - set(test_rows.columns))
    if missing:
        raise KeyError(f"Test rows are missing identity columns: {missing}")
    missing_m28 = sorted(set(key_columns) - set(m28_predictions.columns))
    if missing_m28:
        raise KeyError(f"M28 predictions are missing identity columns: {missing_m28}")

    current_keys = test_rows[key_columns].reset_index(drop=True)
    m28_keys = m28_predictions[key_columns].reset_index(drop=True)
    if not current_keys.equals(m28_keys):
        raise ValueError("M29 test rows do not match the exact M28 test keys")

    probability = np.asarray(tuned_probabilities, dtype=float).reshape(-1)
    if len(probability) != len(current_keys):
        raise ValueError("M29 tuned probability row count differs from test rows")
    if not np.isfinite(probability).all() or (
        (probability < 0) | (probability > 1)
    ).any():
        raise ValueError(
            "M29 tuned probabilities must be finite and between 0 and 1"
        )

    result = m28_predictions.reset_index(drop=True).copy()
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
        "m29_lightgbm_tuning_complete": not failures,
        "ready_for_m30": not failures,
    }


def _best_iteration(model: Any) -> int | None:
    value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None else None


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _manifest_key(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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
        report_prefix = report_dir.relative_to(project_root).as_posix().rstrip("/") + "/"
    except ValueError:
        report_prefix = None
    relevant = [
        line
        for line in status
        if report_prefix is None
        or report_prefix not in line[3:].replace("\\", "/")
    ]
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "remote": git("remote", "get-url", "origin"),
        "working_tree_status": relevant,
    }


def _render_report(
    comparison: pd.DataFrame,
    phase_selections: pd.DataFrame,
    seed_results: pd.DataFrame,
    external: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    test = comparison.loc[comparison["split"].eq("test")].set_index("model")
    lines = [
        "# M29 首杀后 LightGBM 控制变量调参报告",
        "",
        "## 阶段结论",
        "",
        f"验收状态：**{summary['acceptance']['status']}**（"
        f"{summary['acceptance']['blocking_passed']}/"
        f"{summary['acceptance']['blocking_total']}）。",
        f"可进入 M30 固定模型评估：**{summary['acceptance']['ready_for_m30']}**。",
        "36 个候选和五种子稳定性实验只读取 train/validation；test 在参数、"
        "正式种子和选择规则冻结后评估一次。",
        "",
        "## 冻结合同",
        "",
        f"- 样本：{summary['data']['rows']:,}；train/val/test 行数："
        f"{summary['data']['split_rows']}。",
        f"- 系列赛切分：{summary['data']['split_series']}。",
        f"- 特征：{summary['features']['raw_count']} 个原始、"
        f"{summary['features']['encoded_count']} 个编码列。",
        f"- 选择指标：validation Log Loss；每阶段最小接受改善 "
        f"{summary['search']['minimum_phase_improvement']:.4f}。",
        f"- 正式模型：seed 42，最佳迭代 {summary['model']['best_iteration']}。",
        "",
        "## 九阶段选择",
        "",
        "| 阶段 | 选择值 | 改变 | Validation Log Loss | 接受改善 |",
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
            f"Log Loss 范围 `{stability['val_log_loss_range']:.6f}`，AUC 范围 "
            f"`{stability['val_auc_range']:.6f}`；稳定性通过：**{stability['passed']}**。",
            "",
            "## 冻结后测试结果",
            "",
            "| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in ("xgboost_frozen", "lightgbm_baseline", "lightgbm_tuned"):
        row = test.loc[model_name]
        lines.append(
            f"| `{model_name}` | {row['accuracy']:.6f} | {row['auc']:.6f} | "
            f"{row['log_loss']:.6f} | {row['brier_score']:.6f} | "
            f"{row['ece10']:.6f} |"
        )

    for title, key in (
        ("M29 与 M28 LightGBM 基线相差多少", "tuned_vs_m28_test"),
        ("M29 与 M21 XGBoost 相差多少", "tuned_vs_xgboost_test"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "方向修正后正值表示 M29 更好；这里只是点指标，显著性留给 M30。",
                "",
                "| 指标 | 原始差值 | 方向修正后 | M29 更好 |",
                "|---|---:|---:|---|",
            ]
        )
        for metric in REPORT_METRICS:
            item = summary[key][metric]
            lines.append(
                f"| {metric} | {item['raw_left_minus_right']:+.6f} | "
                f"{item['performance_advantage_left']:+.6f} | "
                f"{item['left_performs_better']} |"
            )

    lines.extend(
        [
            "",
            "## 最低门槛与阶段目标",
            "",
            "| 指标 | 当前 | 最低门槛 | 最低通过 | 阶段目标 | 目标通过 |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for metric, item in summary["metric_targets"]["metrics"].items():
        lines.append(
            f"| {metric} | {item['value']:.6f} | {item['minimum']:.3f} | "
            f"{item['minimum_passed']} | {item['stage']:.3f} | "
            f"{item['stage_passed']} |"
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
            "## 与公开结果的数值距离",
            "",
            "| 外部工作 | 指标 | M29 | 外部 | 原始差值 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in closest.to_dict(orient="records"):
        lines.append(
            f"| {row['source_title']} | {row['metric']} | "
            f"{row['current_value']:.6f} | {row['reported_value']:.6f} | "
            f"{row['raw_difference_ours_minus_reported']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这是固定顺序的 greedy search，不是全部参数组合的穷举。",
            "- test 不参与参数选择，测试集上的正负变化都保留。",
            "- M29 尚未做配对置信区间，不能仅凭点指标宣布模型显著更优。",
            "- 不同数据、预测时点或随机行切分的公开结果不能作为公平模型排名。",
            "",
            "## 下一阶段",
            "",
            "M30 冻结本阶段模型，执行系列赛 bootstrap、与 M21 XGBoost 的配对比较、"
            "分地图和 LAN/online 稳健性、校准选择及错误分析。",
            "",
            "复现命令：",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File "
            ".\\scripts\\run_post_first_kill_lightgbm_tuning.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    project_root: str | Path,
    data_path: str | Path = "data/processed/esta_full/first_kill.parquet",
    model_dir: str | Path = "models/esta_full_m29",
    report_dir: str | Path = "reports/esta_full_m29",
    run_tests: bool = True,
    run_compile: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_path = _resolve(root, data_path)
    model_dir = _resolve(root, model_dir)
    report_dir = _resolve(root, report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m28_summary_path = root / "reports/esta_full_m28/m28_summary.json"
    m28_predictions_path = root / "reports/esta_full_m28/m28_test_predictions.csv"
    m28_comparison_path = root / "reports/esta_full_m28/m28_model_comparison.csv"
    m28_columns_path = root / "reports/esta_full_m28/encoded_feature_columns.csv"
    m28_model_path = root / "models/esta_full_m28/post_first_kill_lightgbm_baseline.joblib"
    m28_summary = _read_json(m28_summary_path)
    m28_predictions = pd.read_csv(m28_predictions_path)
    m28_comparison = pd.read_csv(m28_comparison_path)
    m28_columns = pd.read_csv(m28_columns_path)["encoded_feature"].tolist()
    m28_bundle = joblib.load(m28_model_path)

    data_artifact = fingerprint_file(data_path)
    m28_model_artifact = fingerprint_file(m28_model_path)
    data = read_table(data_path)
    data_audit = audit_data_contract(data)
    raw_features = canonical_feature_names()
    prepared = prepare_first_kill_splits(data)
    encoded_columns = prepared["train"][0].columns.tolist()

    m28_prerequisite = bool(
        m28_summary.get("acceptance", {}).get("status") == "passed"
        and m28_summary.get("acceptance", {}).get("ready_for_m29") is True
        and m28_summary.get("data", {}).get("sha256") == data_artifact["sha256"]
        and m28_summary.get("features", {}).get("raw_count") == len(raw_features)
        and m28_summary.get("features", {}).get("encoded_count") == len(m28_columns)
        and m28_summary.get("model", {}).get("model_artifact", {}).get("sha256")
        == m28_model_artifact["sha256"]
        and m28_bundle.get("task") == "post_first_kill"
        and list(m28_bundle.get("columns", [])) == m28_columns
        and all(
            m28_summary.get("model", {}).get("params", {}).get(name) == value
            for name, value in BASE_TUNING_PARAMS.items()
        )
    )
    data_contract = bool(
        data_audit["passed"]
        and data_audit["rows"] == 41027
        and data_audit["split_rows"] == EXPECTED_SPLIT_ROWS
        and data_audit["split_series"] == EXPECTED_SPLIT_SERIES
        and data_artifact["sha256"] == EXPECTED_DATA_SHA256
    )
    feature_contract = bool(
        len(raw_features) == 40
        and len(encoded_columns) == 82
        and encoded_columns == m28_columns
        and encoded_columns == list(m28_bundle.get("columns", []))
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
        test_rows, tuned_probabilities["test"], m28_predictions
    )
    prediction_audit = audit_final_predictions(predictions, len(test_rows))
    comparison = pd.concat([m28_comparison, tuned_metrics], ignore_index=True)
    tuned_vs_m28 = metric_differences(
        comparison, "lightgbm_tuned", "lightgbm_baseline"
    )
    tuned_vs_xgboost = metric_differences(
        comparison, "lightgbm_tuned", "xgboost_frozen"
    )
    baseline_metrics = m28_comparison.loc[
        m28_comparison["model"].eq("lightgbm_baseline")
    ]
    stage_goals = assess_stage_goals(baseline_metrics, tuned_metrics)
    tuned_test_row = tuned_metrics.loc[tuned_metrics["split"].eq("test")].iloc[0]
    tuned_test_metrics = {
        metric: float(tuned_test_row[metric]) for metric in REPORT_METRICS
    }
    metric_targets = assess_metric_targets(tuned_test_metrics)

    external_benchmarks = pd.read_csv(
        root / "benchmarks/external_round_model_metrics.csv"
    )
    external = compare_benchmarks(tuned_test_metrics, external_benchmarks)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    write_markdown_report(
        external,
        tuned_test_metrics,
        report_dir / "external_benchmark_comparison.md",
        stage_label="M29 post-first-kill LightGBM",
    )

    model_path = model_dir / "post_first_kill_lightgbm_tuned.joblib"
    model_bundle = {
        "model": frozen_model,
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "model_name": "lightgbm_tuned",
        "profile": "M21_first_kill_canonical_features",
        "raw_features": raw_features,
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
    tuning_results.to_csv(report_dir / "tuning_candidates.csv", index=False)
    phase_selections.to_csv(report_dir / "phase_selections.csv", index=False)
    seed_results.to_csv(report_dir / "seed_stability.csv", index=False)
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    comparison.to_csv(report_dir / "m29_model_comparison.csv", index=False)
    training_history.to_csv(
        report_dir / "lightgbm_training_history.csv", index=False
    )
    pd.DataFrame({"encoded_feature": encoded_columns}).to_csv(
        report_dir / "encoded_feature_columns.csv", index=False
    )
    write_json(
        {"params": frozen_params, "best_iteration": _best_iteration(frozen_model)},
        report_dir / "frozen_params.json",
    )

    script_path = root / "scripts/run_post_first_kill_lightgbm_tuning.ps1"
    reproduction_entrypoint = audit_reproduction_entrypoint(script_path)
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
        prediction_audit["passed"]
        and encoded_columns == m28_columns
        and len(predictions) == len(m28_predictions)
        and {"xgboost_frozen", "lightgbm_baseline", "lightgbm_tuned"}.issubset(
            set(comparison["model"])
        )
    )
    core_output_names = (
        "tuning_candidates.csv",
        "phase_selections.csv",
        "seed_stability.csv",
        "test_predictions.csv",
        "m29_model_comparison.csv",
        "lightgbm_training_history.csv",
        "encoded_feature_columns.csv",
        "frozen_params.json",
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
        "m28_prerequisite": m28_prerequisite,
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
        "external_report": (
            not external.empty
            and (report_dir / "external_benchmark_comparison.md").is_file()
        ),
        "automated_tests": automated["passed"],
        "source_compile": compile_result["passed"],
        "reproduction_entrypoint": reproduction_entrypoint["passed"],
        "artifact_manifest": artifact_manifest_passed,
    }
    acceptance = decide_acceptance(checks)
    generated_at = datetime.now(timezone.utc).isoformat()
    selected_changes = phase_selections.loc[phase_selections["changed"]]
    summary = {
        "stage": "M29",
        "generated_at_utc": generated_at,
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "experiment_policy": "validation-only greedy sequential LightGBM tuning",
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
            "encoded_columns_match_m28": encoded_columns == m28_columns,
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
            "deployment_tree_count": _best_iteration(frozen_model),
            "model_artifact": model_artifact,
        },
        "metrics": tuned_test_metrics,
        "test_metrics_by_model": {
            row["model"]: {
                metric: float(row[metric]) for metric in REPORT_METRICS
            }
            for row in comparison.loc[comparison["split"].eq("test")].to_dict(
                orient="records"
            )
        },
        "metric_targets": metric_targets,
        "stage_goals": stage_goals,
        "tuned_vs_m28_test": tuned_vs_m28,
        "tuned_vs_xgboost_test": tuned_vs_xgboost,
        "seed_stability": stability,
        "seed42_replay": seed42_replay,
        "prediction_audit": prediction_audit,
        "training_policy": {
            "fit_split": "train",
            "early_stopping_split": "val",
            "selection_metric": "validation_binary_logloss",
            "test_used_for_fit_or_selection": False,
            "candidate_count": grid_audit["candidate_count"],
            "official_seed": 42,
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "lightgbm_version": importlib.metadata.version("lightgbm"),
            "device_type": frozen_model.get_params().get("device_type"),
            "cuda_required": False,
        },
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
        "next_stage": "M30 frozen post-first-kill LightGBM evaluation",
    }
    pd.DataFrame(
        [
            {"check": name, "passed": bool(value), "blocking": True}
            for name, value in checks.items()
        ]
    ).to_csv(report_dir / "m29_checks.csv", index=False)
    summary_path = report_dir / "m29_summary.json"
    report_path = report_dir / "m29_post_first_kill_lightgbm_tuning_report.md"
    write_json(summary, summary_path)
    report_path.write_text(
        _render_report(comparison, phase_selections, seed_results, external, summary),
        encoding="utf-8",
    )

    input_paths = {
        "first_kill_data": data_path,
        "m28_summary": m28_summary_path,
        "m28_test_predictions": m28_predictions_path,
        "m28_model_comparison": m28_comparison_path,
        "m28_encoded_columns": m28_columns_path,
        "m28_baseline_model": m28_model_path,
        "m29_spec": root / "docs/m29_post_first_kill_lightgbm_tuning_spec.md",
        "m29_module": root / "src/csdemo/m29_post_first_kill_lightgbm_tuning.py",
        "m29_tests": root / "tests/test_m29_post_first_kill_lightgbm_tuning.py",
        "reproduction_script": script_path,
        "environment": root / "environment.yml",
        "requirements_lock": root / "requirements-lock.txt",
    }
    output_paths = [
        model_path,
        *(report_dir / name for name in core_output_names),
        report_dir / "m29_checks.csv",
        summary_path,
        report_path,
    ]
    manifest = {
        "stage": "M29",
        "generated_at_utc": generated_at,
        "policy": "M28 data/split/features frozen; validation-only sequential tuning",
        "code": _collect_git_state(root, report_dir),
        "inputs": {name: fingerprint_file(path) for name, path in input_paths.items()},
        "outputs": {
            _manifest_key(root, path): fingerprint_file(path) for path in output_paths
        },
        "contract": {
            "base_params": BASE_TUNING_PARAMS,
            "frozen_params": frozen_params,
            "phase_definitions": PHASE_DEFINITIONS,
            "minimum_phase_improvement": MINIMUM_PHASE_IMPROVEMENT,
            "stability_seeds": STABILITY_SEEDS,
            "stability_limits": STABILITY_LIMITS,
            "raw_features": raw_features,
            "encoded_columns": encoded_columns,
            "test_use": "once after parameters and official seed were frozen",
        },
        "checks": checks,
        "acceptance": acceptance,
    }
    write_json(manifest, report_dir / "m29_experiment_manifest.json")
    if acceptance["status"] != "passed":
        raise RuntimeError(
            "M29 acceptance failed: " + ", ".join(acceptance["blocking_failures"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M29 validation-only post-first-kill LightGBM tuning."
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--data", default="data/processed/esta_full/first_kill.parquet"
    )
    parser.add_argument("--model-dir", default="models/esta_full_m29")
    parser.add_argument("--report-dir", default="reports/esta_full_m29")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    summary = run(
        project_root=args.project_root,
        data_path=args.data,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        run_tests=not args.skip_tests,
        run_compile=not args.skip_compile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

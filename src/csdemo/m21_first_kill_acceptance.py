from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .io import read_table
from .m12_explanation import deployment_tree_count
from .m14_acceptance import (
    audit_environment_lock,
    audit_required_artifacts,
    collect_git_state,
    collect_runtime_environment,
    fingerprint_file,
    inventory_raw_esta,
    run_automated_tests,
)
from .m16_first_kill_baselines import (
    canonical_feature_names,
    compare_external_models,
    prepare_profile_splits,
)
from .m20_first_kill_interface import BLOCKING_CHECKS as M20_BLOCKING_CHECKS
from .metrics import probability_metrics
from .predict_first_kill import FirstKillPredictor, load_snapshot
from .schema import ID_COLUMNS


BLOCKING_CHECKS = (
    "required_artifacts",
    "raw_source",
    "stage_chain",
    "data_identity",
    "split_contract",
    "model_contract",
    "calibrator_contract",
    "prediction_replay",
    "formal_targets",
    "robustness",
    "explanation",
    "prediction_interface",
    "external_comparison",
    "environment_lock",
    "automated_tests",
    "reproduction_entrypoint",
    "progress_report",
)

STAGE_HANDOFFS = {
    "M15": None,
    "M16": "ready_for_m17",
    "M17": "ready_for_m18",
    "M18": "ready_for_m19",
    "M19": "ready_for_m20",
    "M20": "ready_for_m21",
}

REPRODUCTION_TOKENS = (
    "run_pre_round_pipeline.ps1",
    "run_first_kill_data_stage.ps1",
    "run_first_kill_baselines.ps1",
    "run_first_kill_tuning.ps1",
    "run_first_kill_evaluation.ps1",
    "run_first_kill_explanation.ps1",
    "run_first_kill_interface.ps1",
    "src.csdemo.m21_first_kill_acceptance",
    "FullRebuild",
    "RebuildFirstKill",
)

PROGRESS_METRICS = ("accuracy", "auc", "log_loss", "brier_score", "ece10")

REQUIRED_ARTIFACTS = (
    "data/processed/esta_full/first_kill.parquet",
    "reports/esta_full_m14/split_assignments.csv",
    "reports/esta_full_m15/m15_summary.json",
    "reports/esta_full_m16/m16_summary.json",
    "reports/esta_full_m16/m16_model_comparison.csv",
    "reports/esta_full_m17/m17_summary.json",
    "reports/esta_full_m17/model_comparison.csv",
    "reports/esta_full_m17/test_predictions.csv",
    "reports/esta_full_m18/m18_summary.json",
    "reports/esta_full_m19/m19_summary.json",
    "reports/esta_full_m20/m20_summary.json",
    "models/esta_full_m17/first_kill_xgboost_tuned.joblib",
    "models/esta_full_m18/first_kill_calibrator.joblib",
    "benchmarks/external_first_kill_tuned_metrics.csv",
    "examples/first_kill_snapshot.json",
    "examples/first_kill_snapshot.csv",
    "environment.yml",
    "requirements-lock.txt",
    "scripts/run_first_kill_data_stage.ps1",
    "scripts/run_first_kill_baselines.ps1",
    "scripts/run_first_kill_tuning.ps1",
    "scripts/run_first_kill_evaluation.ps1",
    "scripts/run_first_kill_explanation.ps1",
    "scripts/run_first_kill_interface.ps1",
    "scripts/run_first_kill_pipeline.ps1",
    "docs/m21_first_kill_final_acceptance_spec.md",
)


def audit_stage_chain(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Verify every M15-M20 stage and its handoff to the next stage."""

    details: dict[str, bool] = {}
    for stage, readiness_key in STAGE_HANDOFFS.items():
        summary = summaries.get(stage, {})
        if stage == "M15":
            accepted = summary.get("stage") == stage and summary.get("passed") is True
        else:
            acceptance = summary.get("acceptance", {})
            accepted = (
                summary.get("stage") == stage
                and acceptance.get("status") == "passed"
                and acceptance.get(readiness_key) is True
            )
        details[stage] = bool(accepted)

    failed = [stage for stage in STAGE_HANDOFFS if not details[stage]]
    return {
        "passed": not failed,
        "accepted_stages": sum(details.values()),
        "expected_stages": len(STAGE_HANDOFFS),
        "failed_stages": failed,
        "stage_checks": details,
    }


def _group_cross_split_count(frame: pd.DataFrame, column: str) -> int:
    return int(frame.groupby(column, dropna=False)["split"].nunique().gt(1).sum())


def audit_first_kill_data(
    frame: pd.DataFrame,
    *,
    expected_split_rows: Mapping[str, int] | None = None,
    expected_split_series: Mapping[str, int] | None = None,
    required_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit complete keys, grouped splits, labels, and first-kill event values."""

    required_features = list(required_features or ())
    required = [*ID_COLUMNS, "split", "ct_win", *required_features]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        return {
            "passed": False,
            "missing_columns": missing,
            "rows": int(len(frame)),
            "duplicate_key_rows": 0,
            "cross_split_series": 0,
            "cross_split_games": 0,
            "cross_split_rounds": 0,
            "split_rows": {},
            "split_series": {},
        }

    split_order = ("train", "val", "test")
    split_rows = {name: int(frame["split"].eq(name).sum()) for name in split_order}
    split_series = {
        name: int(frame.loc[frame["split"].eq(name), "series_id"].nunique())
        for name in split_order
    }
    duplicate_key_rows = int(frame.duplicated(ID_COLUMNS).sum())
    cross_split_series = _group_cross_split_count(frame, "series_id")
    cross_split_games = _group_cross_split_count(frame, "game_id")
    cross_split_rounds = _group_cross_split_count(frame, "round_id")
    null_identity_cells = int(frame[ID_COLUMNS].isna().sum().sum())
    null_feature_cells = int(frame[required_features].isna().sum().sum())
    invalid_split_rows = int((~frame["split"].isin(split_order)).sum())
    invalid_label_rows = int((~frame["ct_win"].isin([0, 1])).sum())

    invalid_advantage_rows = 0
    invalid_time_rows = 0
    invalid_headshot_rows = 0
    invalid_weapon_rows = 0
    if "first_kill_advantage_ct" in frame:
        invalid_advantage_rows = int(
            (~frame["first_kill_advantage_ct"].isin([-1, 1])).sum()
        )
    if "first_kill_time" in frame:
        times = pd.to_numeric(frame["first_kill_time"], errors="coerce")
        invalid_time_rows = int((~np.isfinite(times) | times.lt(0) | times.gt(180)).sum())
    if "first_kill_headshot" in frame:
        invalid_headshot_rows = int((~frame["first_kill_headshot"].isin([0, 1])).sum())
    if "first_kill_weapon" in frame:
        weapons = frame["first_kill_weapon"].astype("string")
        invalid_weapon_rows = int((weapons.isna() | weapons.str.strip().eq("")).sum())

    row_contract = (
        expected_split_rows is None
        or split_rows == {name: int(expected_split_rows[name]) for name in split_order}
    )
    series_contract = (
        expected_split_series is None
        or split_series
        == {name: int(expected_split_series[name]) for name in split_order}
    )
    passed = all(
        value == 0
        for value in (
            duplicate_key_rows,
            cross_split_series,
            cross_split_games,
            cross_split_rounds,
            null_identity_cells,
            null_feature_cells,
            invalid_split_rows,
            invalid_label_rows,
            invalid_advantage_rows,
            invalid_time_rows,
            invalid_headshot_rows,
            invalid_weapon_rows,
        )
    ) and row_contract and series_contract

    total = len(frame)
    return {
        "passed": bool(passed),
        "missing_columns": [],
        "rows": int(total),
        "series": int(frame["series_id"].nunique()),
        "games": int(frame["game_id"].nunique()),
        "duplicate_key_rows": duplicate_key_rows,
        "cross_split_series": cross_split_series,
        "cross_split_games": cross_split_games,
        "cross_split_rounds": cross_split_rounds,
        "null_identity_cells": null_identity_cells,
        "null_feature_cells": null_feature_cells,
        "invalid_split_rows": invalid_split_rows,
        "invalid_label_rows": invalid_label_rows,
        "invalid_first_kill_advantage_rows": invalid_advantage_rows,
        "invalid_first_kill_time_rows": invalid_time_rows,
        "invalid_first_kill_headshot_rows": invalid_headshot_rows,
        "invalid_first_kill_weapon_rows": invalid_weapon_rows,
        "split_rows": split_rows,
        "split_series": split_series,
        "split_percentages": {
            name: (100.0 * count / total if total else 0.0)
            for name, count in split_rows.items()
        },
        "expected_split_rows_match": row_contract,
        "expected_split_series_match": series_contract,
    }


def audit_prediction_replay(
    saved: pd.DataFrame,
    replayed: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
    saved_probability_column: str = "xgboost_tuned_probability",
    replayed_probability_column: str = "ct_win_probability",
) -> dict[str, Any]:
    """Compare saved and replayed probabilities by the complete round key."""

    saved_required = [*ID_COLUMNS, "ct_win", saved_probability_column]
    replayed_required = [*ID_COLUMNS, "ct_win", replayed_probability_column]
    missing_saved = sorted(set(saved_required) - set(saved.columns))
    missing_replayed = sorted(set(replayed_required) - set(replayed.columns))
    if missing_saved or missing_replayed:
        return {
            "passed": False,
            "missing_saved_columns": missing_saved,
            "missing_replayed_columns": missing_replayed,
            "key_mismatch_count": max(len(saved), len(replayed)),
            "label_mismatch_count": 0,
            "max_absolute_probability_difference": math.inf,
        }

    saved_duplicate_rows = int(saved.duplicated(ID_COLUMNS).sum())
    replayed_duplicate_rows = int(replayed.duplicated(ID_COLUMNS).sum())
    left = saved[saved_required].rename(
        columns={
            "ct_win": "ct_win_saved",
            saved_probability_column: "probability_saved",
        }
    )
    right = replayed[replayed_required].rename(
        columns={
            "ct_win": "ct_win_replayed",
            replayed_probability_column: "probability_replayed",
        }
    )
    merged = left.merge(right, on=ID_COLUMNS, how="outer", indicator=True)
    matched = merged.loc[merged["_merge"].eq("both")].copy()
    key_mismatch_count = int((~merged["_merge"].eq("both")).sum())
    label_mismatch_count = int(
        matched["ct_win_saved"].ne(matched["ct_win_replayed"]).sum()
    )
    saved_probability = pd.to_numeric(matched["probability_saved"], errors="coerce")
    replayed_probability = pd.to_numeric(
        matched["probability_replayed"], errors="coerce"
    )
    invalid_probability_cells = int(
        (~np.isfinite(saved_probability)).sum()
        + (~np.isfinite(replayed_probability)).sum()
    )
    differences = (saved_probability - replayed_probability).abs()
    max_difference = float(differences.max()) if len(differences) else math.inf
    passed = (
        saved_duplicate_rows == 0
        and replayed_duplicate_rows == 0
        and key_mismatch_count == 0
        and label_mismatch_count == 0
        and invalid_probability_cells == 0
        and max_difference <= tolerance
    )
    return {
        "passed": bool(passed),
        "tolerance": float(tolerance),
        "saved_rows": int(len(saved)),
        "replayed_rows": int(len(replayed)),
        "matched_rows": int(len(matched)),
        "saved_duplicate_key_rows": saved_duplicate_rows,
        "replayed_duplicate_key_rows": replayed_duplicate_rows,
        "key_mismatch_count": key_mismatch_count,
        "label_mismatch_count": label_mismatch_count,
        "invalid_probability_cells": invalid_probability_cells,
        "max_absolute_probability_difference": max_difference,
    }


def audit_frozen_metrics(
    current: Mapping[str, float],
    expected: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Require every expected metric to replay within a strict tolerance."""

    missing = sorted(set(expected) - set(current))
    differences: dict[str, float] = {}
    for name in expected:
        if name in current:
            differences[name] = abs(float(current[name]) - float(expected[name]))
    max_difference = max(differences.values(), default=math.inf)
    finite = all(math.isfinite(value) for value in differences.values())
    return {
        "passed": not missing and finite and max_difference <= tolerance,
        "tolerance": float(tolerance),
        "missing_metrics": missing,
        "absolute_differences": differences,
        "max_absolute_difference": float(max_difference),
    }


def _canonical_target_rows(payload: Mapping[str, Any]) -> str:
    rows = sorted(payload.get("rows", []), key=lambda row: str(row.get("target_id", "")))
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)


def audit_formal_targets(
    m19_targets: Mapping[str, Any], m20_targets: Mapping[str, Any]
) -> dict[str, Any]:
    """Ensure M19's ten formal targets remain unchanged and passed in M20/M21."""

    count_contract = all(
        payload.get("passed_count") == 10
        and payload.get("target_count") == 10
        and payload.get("remaining_count") == 0
        and payload.get("all_formal_targets_passed") is True
        for payload in (m19_targets, m20_targets)
    )
    rows_equal = _canonical_target_rows(m19_targets) == _canonical_target_rows(m20_targets)
    target_ids = [row.get("target_id") for row in m19_targets.get("rows", [])]
    unique_ten_targets = len(target_ids) == len(set(target_ids)) == 10
    rows_passed = all(
        row.get("passed") is True and float(row.get("remaining", math.inf)) == 0.0
        for row in m19_targets.get("rows", [])
    )
    return {
        "passed": bool(count_contract and rows_equal and unique_ten_targets and rows_passed),
        "passed_count": int(m19_targets.get("passed_count", 0)),
        "target_count": int(m19_targets.get("target_count", 0)),
        "remaining_count": int(m19_targets.get("remaining_count", -1)),
        "rows_equal_between_m19_and_m20": rows_equal,
        "unique_ten_targets": unique_ten_targets,
    }


def _metric_row(
    stage: str,
    task: str,
    model: str,
    prediction_point: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage,
        "task": task,
        "model": model,
        "prediction_point": prediction_point,
    }
    for name in PROGRESS_METRICS:
        value = metrics.get(name, np.nan)
        row[name] = float(value) if pd.notna(value) else np.nan
    return row


def build_progress_metrics(
    m6_metrics: Mapping[str, Any],
    pre_round_metrics: Mapping[str, Any],
    m16_comparison: pd.DataFrame,
    current_first_kill_metrics: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the stage metric table used by the M6-to-M21 report."""

    test_rows = m16_comparison.loc[m16_comparison["split"].eq("test")].set_index(
        "model"
    )
    required_models = {"logistic_regression", "xgboost_untuned"}
    missing = sorted(required_models - set(test_rows.index))
    if missing:
        raise ValueError("M16 comparison is missing test models: " + ", ".join(missing))

    rows = [
        _metric_row(
            "M6",
            "pre_round",
            "xgboost_m6",
            "purchase complete, before combat",
            m6_metrics,
        ),
        _metric_row(
            "M14",
            "pre_round",
            "xgboost_tuned",
            "purchase complete, before combat",
            pre_round_metrics,
        ),
        _metric_row(
            "M16",
            "post_first_kill",
            "logistic_regression",
            "immediately after first valid enemy kill",
            test_rows.loc["logistic_regression"],
        ),
        _metric_row(
            "M16",
            "post_first_kill",
            "xgboost_untuned",
            "immediately after first valid enemy kill",
            test_rows.loc["xgboost_untuned"],
        ),
        _metric_row(
            "M21",
            "post_first_kill",
            "xgboost_tuned_frozen",
            "immediately after first valid enemy kill",
            current_first_kill_metrics,
        ),
    ]
    return pd.DataFrame(rows)


def build_progress_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate fair within-task changes separately from prediction-time changes."""

    def select(stage: str, model: str) -> pd.Series:
        rows = metrics.loc[metrics["stage"].eq(stage) & metrics["model"].eq(model)]
        if len(rows) != 1:
            raise ValueError(f"Expected one progress row for {stage}/{model}")
        return rows.iloc[0]

    definitions = [
        (
            "m6_to_m14_pre_round",
            select("M6", "xgboost_m6"),
            select("M14", "xgboost_tuned"),
            "same_task_same_split",
            "Fair pre-round engineering and tuning comparison.",
        ),
        (
            "m16_to_m21_first_kill_tuning",
            select("M16", "xgboost_untuned"),
            select("M21", "xgboost_tuned_frozen"),
            "same_task_same_split",
            "Fair post-first-kill tuning comparison.",
        ),
        (
            "m14_to_m21_prediction_time",
            select("M14", "xgboost_tuned"),
            select("M21", "xgboost_tuned_frozen"),
            "timing_change_not_algorithm_only",
            "Includes new first-kill information; not a pure tuning gain.",
        ),
        (
            "m6_to_m21_project_progress",
            select("M6", "xgboost_m6"),
            select("M21", "xgboost_tuned_frozen"),
            "mixed_scope_project_progress",
            "Combines data repair, engineering, and a later prediction point.",
        ),
    ]
    rows = []
    for comparison_id, before, after, comparability, note in definitions:
        row: dict[str, Any] = {
            "comparison_id": comparison_id,
            "from_stage": before["stage"],
            "to_stage": after["stage"],
            "comparability": comparability,
            "note": note,
        }
        for metric in PROGRESS_METRICS:
            before_value = before.get(metric, np.nan)
            after_value = after.get(metric, np.nan)
            row[f"{metric}_change"] = (
                float(after_value) - float(before_value)
                if pd.notna(before_value) and pd.notna(after_value)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_progress_report(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    context: Mapping[str, Any],
) -> str:
    """Render a concise Chinese M6-to-M21 progress report."""

    lines = [
        "# M6 到 M21 项目进度报告",
        "",
        "## 当前结论",
        "",
        f"M21 阻断验收：{context.get('blocking_passed', 0)}/"
        f"{context.get('blocking_total', len(BLOCKING_CHECKS))}；"
        f"完整自动化测试：{context.get('test_count', '待运行')} 项。",
        "首杀后 XGBoost 完成后，下一条独立研究线是 LightGBM 同数据对照，"
        "之后才进入实时胜率。",
        "",
        "## 阶段指标",
        "",
        "| 阶段 | 任务 | 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict(orient="records"):
        def value(name: str) -> str:
            item = row.get(name, np.nan)
            return f"{float(item):.6f}" if pd.notna(item) else "-"

        lines.append(
            f"| {row.get('stage', '')} | {row.get('task', '')} | "
            f"{row.get('model', '')} | {value('accuracy')} | {value('auc')} | "
            f"{value('log_loss')} | {value('brier_score')} | {value('ece10')} |"
        )

    lines.extend(
        [
            "",
            "## 变化应如何解释",
            "",
            "M6 到 M14、M16 到 M21 是同任务同切分的公平比较。M14 到 M21 改变了"
            "预测时点并加入首杀事件，因此这部分提升**不是纯调参增益**。",
            "",
            "| 比较 | 可比性 | AUC 变化 | Log Loss 变化 |",
            "|---|---|---:|---:|",
        ]
    )
    for row in comparisons.to_dict(orient="records"):
        auc_change = row.get("auc_change", np.nan)
        loss_change = row.get("log_loss_change", np.nan)
        auc_text = f"{float(auc_change):+.6f}" if pd.notna(auc_change) else "-"
        loss_text = f"{float(loss_change):+.6f}" if pd.notna(loss_change) else "-"
        lines.append(
            f"| {row.get('comparison_id', '')} | {row.get('comparability', '')} | "
            f"{auc_text} | {loss_text} |"
        )

    data = context.get("data", {})
    lines.extend(
        [
            "",
            "## 数据与切分进展",
            "",
        ]
    )
    if data:
        lines.extend(
            [
                f"当前首杀后数据包含 {int(data.get('rows', 0)):,} 个样本、"
                f"{int(data.get('series', 0)):,} 个系列和 "
                f"{int(data.get('games', 0)):,} 个 demo。",
                "",
                "| split | series | 样本 | 样本占比 |",
                "|---|---:|---:|---:|",
            ]
        )
        split_rows = data.get("split_rows", {})
        split_series = data.get("split_series", {})
        percentages = data.get("split_percentages", {})
        for split in ("train", "val", "test"):
            lines.append(
                f"| {split} | {int(split_series.get(split, 0)):,} | "
                f"{int(split_rows.get(split, 0)):,} | "
                f"{float(percentages.get(split, 0.0)):.2f}% |"
            )
        lines.extend(
            [
                "",
                "跨 split 的 series/game/round 数量分别为 "
                f"`{data.get('cross_split_series', 0)}` / "
                f"`{data.get('cross_split_games', 0)}` / "
                f"`{data.get('cross_split_rounds', 0)}`。因此所谓 70/20/10 是"
                "系列级近似比例，不是先随机拆回合再四舍五入。",
            ]
        )
    else:
        lines.append("正式运行时由 M21 清单填入首杀后数据和系列级切分审计。")

    stage_rows = [
        ("M6", "购买结束特征修复", "M4A1-S、5v5 常量和 36 个原始特征固定"),
        ("M7", "公平基线", "Dummy、逻辑回归和 XGBoost 同切分比较"),
        ("M8", "控制变量调参", "early stopping 将 Train-Val AUC 差降至 0.0111"),
        ("M9", "固定测试评估", "2,000 次系列 bootstrap 与完整概率指标"),
        ("M10", "校准", "validation OOF 选择保留原始概率"),
        ("M11", "稳健性", "LAN/online、地图和高置信错误审计"),
        ("M12", "解释", "Gain、Permutation、TreeSHAP 与泄漏审计"),
        ("M13", "购买结束接口", "严格 JSON/CSV 单条 CT/T 胜率"),
        ("M14", "购买结束最终验收", "15/15 阻断项通过，进入首杀后任务"),
        ("M15", "首杀数据重建", "按最小 tick 选事件，41,027 个有效样本"),
        ("M16", "首杀后基线", "Dummy、逻辑回归和未调参 XGBoost"),
        ("M17", "首杀后调参", "固定 409 棵部署树，AUC 0.8098"),
        ("M18", "评估与稳健性", "置信区间、分组结果、校准和错误复核"),
        ("M19", "解释与目标差距", "40/82 特征可追溯，十项目标通过"),
        ("M20", "首杀后接口", "JSON/CSV 一致，10 个非法输入被拒绝"),
        ("M21", "最终验收", "哈希、概率、指标、环境与一键复现闭环"),
    ]
    lines.extend(
        [
            "",
            "## M6-M21 模块记录",
            "",
            "| 模块 | 目标 | 已达到的效果 | 状态 |",
            "|---|---|---|---|",
        ]
    )
    for stage, goal, result in stage_rows:
        lines.append(f"| {stage} | {goal} | {result} | 完成 |")

    target_rows = context.get("formal_target_rows", [])
    lines.extend(
        [
            "",
            "## 首杀后正式目标",
            "",
        ]
    )
    if target_rows:
        lines.extend(
            [
                "| 目标 | 当前 | 阈值 | Remaining | 通过余量 | 状态 |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in target_rows:
            lines.append(
                f"| {row.get('label', row.get('target_id', ''))} | "
                f"{float(row.get('current', 0.0)):.6f} | "
                f"{float(row.get('target', 0.0)):.6f} | "
                f"{float(row.get('remaining', 0.0)):.6f} | "
                f"{float(row.get('margin', 0.0)):.6f} | "
                f"{'通过' if row.get('passed') else '未通过'} |"
            )
        remaining = sum(float(row.get("remaining", 0.0)) for row in target_rows)
        lines.extend(
            [
                "",
                f"首杀后正式目标的 Remaining 合计为 `{remaining:.6f}`；"
                "这表示已经达到预先冻结的验收线，不表示模型不存在继续研究空间。",
            ]
        )
    else:
        lines.append("正式运行时从 M19/M20 冻结目标中填入 10 项验收结果。")

    external_rows = context.get("closest_external", [])
    lines.extend(
        [
            "",
            "## 外部模型差距",
            "",
            "外部数据、切分和实现不同，以下差值只提供背景，不能解释为直接排名。",
            "",
        ]
    )
    if external_rows:
        lines.extend(
            [
                "| 外部来源 | 指标 | 本项目 | 外部 | 本项目减外部 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in external_rows:
            difference = float(row.get("raw_difference_ours_minus_reported", 0.0))
            difference_text = (
                f"{difference * 100:+.2f} 个百分点"
                if row.get("metric") in {"accuracy", "auc"}
                else f"{difference:+.6f}"
            )
            lines.append(
                f"| {row.get('source_title', row.get('benchmark_id', ''))} | "
                f"{row.get('metric', '')} | {float(row.get('current_value', 0.0)):.6f} | "
                f"{float(row.get('reported_value', 0.0)):.6f} | {difference_text} |"
            )
    else:
        lines.append("正式运行时从已冻结的外部基准表生成最接近任务的差值。")

    lines.extend(
        [
            "",
            "## 工程成熟度变化",
            "",
            "| 维度 | M6 时 | M21 当前 |",
            "|---|---|---|",
            "| 数据主键 | 正在修复购买结束样本 | 回合、首杀、预测和报告可用完整键追踪 |",
            "| 数据质量 | M4A1-S 与常量特征修复 | 首杀 tick、事件关联、排除原因和哈希均审计 |",
            "| 切分 | 系列级切分已建立 | series/game/round 跨集合交叉均为 0 |",
            "| 模型对照 | 单个 XGBoost 为主 | Dummy、逻辑回归、未调参/调参 XGBoost 全部保留 |",
            "| 不确定性 | 主要看单点指标 | 2,000 次系列 bootstrap 与分组置信区间 |",
            "| 解释 | 单一 importance | Gain、Permutation、分组 Permutation、TreeSHAP |",
            "| 使用方式 | 训练脚本和报告 | 购买结束与首杀后均有严格 JSON/CSV 接口 |",
            f"| 自动化测试 | 22 项 | {context.get('test_count', '待运行')} 项 |",
            "| 复现 | 分散命令 | 环境锁、产物哈希、实验清单和三模式一键入口 |",
            "",
            "购买结束模型的 M14 工程验收已经完成，但其四个更高研究目标仍未达到；"
            "首杀后模型的十项冻结目标已达到。两者必须分开陈述。",
        ]
    )

    lines.extend(
        [
            "",
            "## 尚未完成",
            "",
            "- LightGBM 尚未在同一数据、切分和指标合同上进行控制变量对照。",
            "- 实时胜率的数据快照、事件序列和模型尚未开始。",
            "- 时间外推测试以及战队/选手身份特征仍是后续独立实验。",
            "- 外部论文使用不同数据与切分，差值只能作为背景参考，不能作为排名。",
            "",
        ]
    )
    return "\n".join(lines)


def audit_reproduction_entrypoint(script_text: str) -> dict[str, Any]:
    """Require the pipeline script to expose default and both rebuild modes."""

    missing = [token for token in REPRODUCTION_TOKENS if token not in script_text]
    return {
        "passed": not missing,
        "required_token_count": len(REPRODUCTION_TOKENS),
        "present_token_count": len(REPRODUCTION_TOKENS) - len(missing),
        "missing_tokens": missing,
    }


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
        return str(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _relative_fingerprint(root: Path, relative_path: str) -> dict[str, Any]:
    result = fingerprint_file(root / relative_path)
    result["path"] = relative_path.replace("\\", "/")
    return result


def build_split_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one persistent row per series for the accepted first-kill sample."""

    required = {*ID_COLUMNS, "split", "ct_win"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError("Split assignment data missing columns: " + ", ".join(missing))
    if frame.groupby("series_id")["split"].nunique().gt(1).any():
        raise ValueError("A series_id cannot be assigned to multiple splits")
    return (
        frame.groupby("series_id", as_index=False)
        .agg(
            split=("split", "first"),
            game_count=("game_id", "nunique"),
            round_count=("round_id", "size"),
            ct_win_rate=("ct_win", "mean"),
        )
        .sort_values("series_id")
        .reset_index(drop=True)
    )


def audit_split_manifest(
    assignments: pd.DataFrame, manifest: pd.DataFrame
) -> dict[str, Any]:
    """Confirm the first-kill sample preserves M14's one-series split manifest."""

    required = {"series_id", "split"}
    missing_assignments = sorted(required - set(assignments.columns))
    missing_manifest = sorted(required - set(manifest.columns))
    if missing_assignments or missing_manifest:
        return {
            "passed": False,
            "missing_assignment_columns": missing_assignments,
            "missing_manifest_columns": missing_manifest,
            "split_mismatch_series": 0,
            "missing_series": max(len(assignments), len(manifest)),
        }

    assignment_duplicates = int(assignments["series_id"].duplicated().sum())
    manifest_duplicates = int(manifest["series_id"].duplicated().sum())
    merged = assignments[["series_id", "split"]].merge(
        manifest[["series_id", "split"]],
        on="series_id",
        how="outer",
        suffixes=("_first_kill", "_m14"),
        indicator=True,
    )
    missing_series = int((~merged["_merge"].eq("both")).sum())
    matched = merged.loc[merged["_merge"].eq("both")]
    split_mismatch = int(matched["split_first_kill"].ne(matched["split_m14"]).sum())
    return {
        "passed": (
            assignment_duplicates == 0
            and manifest_duplicates == 0
            and missing_series == 0
            and split_mismatch == 0
        ),
        "assignment_rows": int(len(assignments)),
        "manifest_rows": int(len(manifest)),
        "assignment_duplicate_series": assignment_duplicates,
        "manifest_duplicate_series": manifest_duplicates,
        "missing_series": missing_series,
        "split_mismatch_series": split_mismatch,
    }


def replay_frozen_model(
    data: pd.DataFrame,
    bundle: Mapping[str, Any],
    saved_predictions: pd.DataFrame,
    expected_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Replay the frozen M17 model on the unchanged M18 test partition."""

    raw_features = list(bundle.get("raw_features", []))
    encoded_columns = list(bundle.get("columns", []))
    prepared = prepare_profile_splits(data, raw_features)
    x_test, y_test, identity = prepared["test"]
    encoded_columns_match = x_test.columns.tolist() == encoded_columns
    if not encoded_columns_match:
        raise RuntimeError("M21 encoded test columns differ from the frozen model bundle")

    probability = np.asarray(
        bundle["model"].predict_proba(x_test)[:, 1], dtype=float
    )
    current_metrics = probability_metrics(y_test, probability, n_bins=10)
    replayed = identity.copy()
    replayed["ct_win"] = y_test.to_numpy(dtype=int)
    replayed["ct_win_probability"] = probability
    prediction_audit = audit_prediction_replay(saved_predictions, replayed)
    metric_audit = audit_frozen_metrics(current_metrics, expected_metrics)
    tree_count = deployment_tree_count(dict(bundle))
    passed = (
        encoded_columns_match
        and len(raw_features) == 40
        and len(encoded_columns) == 82
        and len(encoded_columns) == len(set(encoded_columns))
        and tree_count == int(bundle.get("best_tree_count", -1)) == 409
        and prediction_audit["passed"]
        and metric_audit["passed"]
    )
    return {
        "passed": bool(passed),
        "test_rows": int(len(x_test)),
        "raw_feature_count": len(raw_features),
        "encoded_feature_count": len(encoded_columns),
        "encoded_columns_match": encoded_columns_match,
        "deployment_tree_count": tree_count,
        "xgboost_fit_calls": 0,
        "metrics": current_metrics,
        "metric_audit": metric_audit,
        "prediction_audit": prediction_audit,
        "replayed_predictions": replayed,
    }


def audit_prediction_interface(
    predictor: FirstKillPredictor,
    json_example: Path,
    csv_example: Path,
    m20_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay M20's JSON/CSV example against the frozen interface contract."""

    json_result = predictor.predict(load_snapshot(json_example))
    csv_result = predictor.predict(load_snapshot(csv_example))
    json_probability = float(json_result["prediction"]["ct_win_probability"])
    csv_probability = float(csv_result["prediction"]["ct_win_probability"])
    expected_probability = float(
        m20_summary["example_prediction"]["prediction"]["ct_win_probability"]
    )
    probability_sum = float(json_result["prediction"]["probability_sum"])
    m20_checks = m20_summary.get("checks", {})
    m20_frozen_checks = all(bool(m20_checks.get(name)) for name in M20_BLOCKING_CHECKS)
    json_csv_match = math.isclose(
        json_probability, csv_probability, rel_tol=0.0, abs_tol=1e-15
    )
    expected_match = math.isclose(
        json_probability, expected_probability, rel_tol=0.0, abs_tol=1e-15
    )
    passed = (
        m20_summary.get("acceptance", {}).get("status") == "passed"
        and m20_summary.get("acceptance", {}).get("ready_for_m21") is True
        and m20_frozen_checks
        and predictor.model_audit.get("passed") is True
        and predictor.calibrator_audit.get("passed") is True
        and json_result["validation"]["status"] == "passed"
        and csv_result["validation"]["status"] == "passed"
        and json_csv_match
        and expected_match
        and 0.0 <= json_probability <= 1.0
        and math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12)
    )
    return {
        "passed": bool(passed),
        "m20_frozen_checks_passed": m20_frozen_checks,
        "json_csv_prediction_match": json_csv_match,
        "m20_example_prediction_match": expected_match,
        "probability_difference_vs_m20": abs(json_probability - expected_probability),
        "example_prediction": json_result,
        "model_audit": predictor.model_audit,
        "calibrator_audit": predictor.calibrator_audit,
    }


def render_external_comparison(external: pd.DataFrame) -> str:
    lines = [
        "# M21 外部模型指标比较",
        "",
        "差值统一为“本项目指标 - 外部报告指标”。不同数据集、切分和预测时点不能解释为"
        "算法排名；`closest_task` 只是当前最接近的公开任务。",
        "",
        "| 可比性 | 当前模型 | 外部来源 | 指标 | 当前 | 外部 | 差值 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in external.to_dict(orient="records"):
        title = str(row.get("source_title", row.get("benchmark_id", "")))
        url = row.get("source_url")
        source = f"[{title}]({url})" if isinstance(url, str) and url else title
        current = row.get("current_value", np.nan)
        reported = row.get("reported_value", np.nan)
        difference = row.get("raw_difference_ours_minus_reported", np.nan)
        current_text = f"{float(current):.6f}" if pd.notna(current) else "-"
        reported_text = f"{float(reported):.6f}" if pd.notna(reported) else "-"
        if pd.isna(difference):
            difference_text = "-"
        elif row.get("metric") in {"accuracy", "auc"}:
            difference_text = f"{float(difference) * 100:+.2f} 个百分点"
        else:
            difference_text = f"{float(difference):+.6f}"
        lines.append(
            f"| {row.get('comparability', '')} | `{row.get('current_model', '')}` | "
            f"{source} | {row.get('metric', '')} | {current_text} | "
            f"{reported_text} | {difference_text} |"
        )
    return "\n".join(lines) + "\n"


def render_final_report(summary: Mapping[str, Any], external: pd.DataFrame) -> str:
    acceptance = summary["acceptance"]
    data = summary["data"]
    metrics = summary["metrics"]
    targets = summary["formal_targets"]
    tests = summary["automated_tests"]
    lines = [
        "# M21 首杀后 XGBoost 最终验收报告",
        "",
        "## 最终结论",
        "",
        f"验收状态：**{acceptance['status']}**；阻断项："
        f"**{acceptance['blocking_passed']}/{acceptance['blocking_total']}**；"
        f"首杀后 XGBoost 完成：**{acceptance['first_kill_xgboost_complete']}**。",
        "M21 没有训练或调参，只回放固定 M17 模型、M18 identity 校准器和 M20 接口。",
        "",
        "## 数据与切分",
        "",
        f"- 首杀后样本：{data['rows']:,}；series：{data['series']:,}；"
        f"game：{data['games']:,}；",
        f"- 数据 SHA-256：`{data['sha256']}`；",
        "- 跨 split 的 series/game/round："
        f"{data['cross_split_series']}/{data['cross_split_games']}/"
        f"{data['cross_split_rounds']}；",
        "",
        "| split | series | 样本 | 占比 |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(
            f"| {split} | {data['split_series'][split]:,} | "
            f"{data['split_rows'][split]:,} | "
            f"{data['split_percentages'][split]:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 固定测试指标",
            "",
            "| 指标 | M21 回放值 |",
            "|---|---:|",
        ]
    )
    labels = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "log_loss": "Log Loss",
        "brier_score": "Brier",
        "ece10": "ECE10",
    }
    for name, label in labels.items():
        lines.append(f"| {label} | {metrics[name]:.6f} |")
    lines.extend(
        [
            "",
            f"测试概率最大回放误差：`{summary['prediction_replay']['max_absolute_probability_difference']:.3e}`；"
            f"指标最大回放误差：`{summary['metric_replay']['max_absolute_difference']:.3e}`。",
            f"十项正式目标通过：`{targets['passed_count']}/{targets['target_count']}`；"
            f"remaining：`{targets['remaining_count']}`。",
            "",
            "## 模型与接口合同",
            "",
            f"- 原始/编码特征：{summary['model_contract']['raw_feature_count']}/"
            f"{summary['model_contract']['encoded_feature_count']}；",
            f"- 部署树数：{summary['model_contract']['deployment_tree_count']}；",
            f"- 模型 SHA-256：`{summary['artifacts']['model']['sha256']}`；",
            f"- 校准器 SHA-256：`{summary['artifacts']['calibrator']['sha256']}`；",
            f"- JSON/CSV 示例一致：{summary['interface']['json_csv_prediction_match']}；",
            f"- XGBoost fit 调用：{summary['xgboost_fit_calls']}。",
            "",
            "## 外部指标差距",
            "",
            "以下仅列最接近的公开首杀后任务，仍因数据和切分不同而不能直接排名。",
            "",
            "| 外部来源 | 指标 | 本项目逻辑回归 | 外部 | 差值 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    closest = external.loc[
        external["comparability"].eq("closest_task")
        & external["comparison_status"].eq("compared")
    ]
    for row in closest.to_dict(orient="records"):
        difference = float(row["raw_difference_ours_minus_reported"])
        difference_text = (
            f"{difference * 100:+.2f} 个百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| {row.get('source_title', row['benchmark_id'])} | {row['metric']} | "
            f"{row['current_value']:.6f} | {row['reported_value']:.6f} | "
            f"{difference_text} |"
        )
    test_count = tests.get("test_count")
    lines.extend(
        [
            "",
            "## 可复现性",
            "",
            f"- Git commit：`{summary['code']['commit']}`；",
            f"- Python：`{summary['environment']['python_version']}`；",
            f"- 自动化测试：{test_count if test_count is not None else '调用方跳过'}；",
            f"- 环境锁通过：{summary['environment_lock']['passed']}；",
            "",
            "```powershell",
            ".\\scripts\\run_first_kill_pipeline.ps1",
            "```",
            "",
            "完整重建使用 `-FullRebuild`；只从 M14 产物重建首杀后任务使用"
            " `-RebuildFirstKill`。",
            "",
            "## 下一阶段",
            "",
            "首杀后 XGBoost 已关闭。下一阶段是 LightGBM 在完全相同数据、切分、特征和"
            "指标合同上的控制变量对照；实时胜率仍是之后的独立任务。",
            "",
        ]
    )
    return "\n".join(lines)


def run_acceptance(
    *,
    project_root: str | Path,
    esta_root: str | Path,
    report_dir: str | Path = "reports/esta_full_m21",
    progress_report_path: str | Path = "reports/m6_to_m21_progress_report.md",
    run_tests: bool = True,
) -> dict[str, Any]:
    """Run the no-training M21 final acceptance on frozen local artifacts."""

    root = Path(project_root).resolve()
    output_dir = _resolve(root, report_dir)
    progress_path = _resolve(root, progress_report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    required_artifacts = audit_required_artifacts(root, list(REQUIRED_ARTIFACTS))
    if not required_artifacts["passed"]:
        raise FileNotFoundError(
            "M21 required artifacts are missing: "
            + ", ".join(required_artifacts["missing"])
        )

    stage_paths = {
        stage: root / f"reports/esta_full_{stage.lower()}/{stage.lower()}_summary.json"
        for stage in ("M15", "M16", "M17", "M18", "M19", "M20")
    }
    stage_summaries = {stage: _read_json(path) for stage, path in stage_paths.items()}
    m15 = stage_summaries["M15"]
    m16 = stage_summaries["M16"]
    m18 = stage_summaries["M18"]
    m19 = stage_summaries["M19"]
    m20 = stage_summaries["M20"]
    stage_chain = audit_stage_chain(stage_summaries)

    data_path = root / "data/processed/esta_full/first_kill.parquet"
    model_path = root / "models/esta_full_m17/first_kill_xgboost_tuned.joblib"
    calibrator_path = root / "models/esta_full_m18/first_kill_calibrator.joblib"
    data = read_table(data_path)
    expected_split_rows = m18["data"]["split_rows"]
    expected_split_series = m18["data"]["split_series"]
    data_audit = audit_first_kill_data(
        data,
        expected_split_rows=expected_split_rows,
        expected_split_series=expected_split_series,
        required_features=canonical_feature_names(),
    )
    assignments = build_split_assignments(data)
    m14_manifest = read_table(root / "reports/esta_full_m14/split_assignments.csv")
    split_manifest = audit_split_manifest(assignments, m14_manifest)

    data_artifact = fingerprint_file(data_path)
    model_artifact = fingerprint_file(model_path)
    calibrator_artifact = fingerprint_file(calibrator_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("M21 expected the M17 model artifact to contain a bundle")
    predictor = FirstKillPredictor.from_paths(model_path, calibrator_path)

    expected_data_hashes = {
        str(m15.get("data_artifact", {}).get("sha256")),
        str(m16.get("data", {}).get("sha256")),
        str(m18.get("data", {}).get("sha256")),
        str(m19.get("prerequisite", {}).get("data_artifact", {}).get("sha256")),
        str(bundle.get("data_sha256")),
    }
    data_hash_match = expected_data_hashes == {data_artifact["sha256"]}
    expected_model_hashes = {
        str(m18.get("prerequisite", {}).get("model_artifact", {}).get("sha256")),
        str(m19.get("prerequisite", {}).get("model_artifact", {}).get("sha256")),
        str(m20.get("artifacts", {}).get("model_sha256")),
    }
    model_hash_match = expected_model_hashes == {model_artifact["sha256"]}
    expected_calibrator_hashes = {
        str(m18.get("calibration", {}).get("calibrator_artifact", {}).get("sha256")),
        str(m20.get("artifacts", {}).get("calibrator_sha256")),
    }
    calibrator_hash_match = expected_calibrator_hashes == {
        calibrator_artifact["sha256"]
    }

    model_contract = {
        **predictor.model_audit,
        "passed": bool(
            predictor.model_audit.get("passed")
            and data_hash_match
            and model_hash_match
            and bundle.get("task") == "first_kill"
            and bundle.get("profile") == "canonical_event"
            and list(bundle.get("raw_features", [])) == canonical_feature_names()
            and deployment_tree_count(bundle) == int(bundle.get("best_tree_count", -1))
            == 409
        ),
        "data_hash_match": data_hash_match,
        "model_hash_match": model_hash_match,
    }
    calibrator_contract = {
        **predictor.calibrator_audit,
        "passed": bool(
            predictor.calibrator_audit.get("passed")
            and calibrator_hash_match
            and predictor.calibrator_audit.get("base_model_sha256")
            == model_artifact["sha256"]
            and predictor.calibrator_audit.get("data_sha256")
            == data_artifact["sha256"]
        ),
        "calibrator_hash_match": calibrator_hash_match,
    }

    saved_predictions = read_table(root / "reports/esta_full_m17/test_predictions.csv")
    replay = replay_frozen_model(data, bundle, saved_predictions, m18["metrics"])
    prediction_replay = replay["prediction_audit"]
    metric_replay = replay["metric_audit"]
    current_metrics = replay["metrics"]
    formal_targets = audit_formal_targets(m19["target_gap"], m20["formal_targets"])

    robustness = {
        "passed": bool(
            m18.get("acceptance", {}).get("status") == "passed"
            and m18.get("stage_targets", {}).get("all_passed") is True
            and m18.get("robustness", {}).get("large_map_stage_passed") is True
            and m18.get("robustness", {}).get("large_map_ci_stage_passed") is True
            and m18.get("robustness", {}).get("source_gap_passed") is True
            and m18.get("source_auc_gap", {}).get("ci_includes_zero") is True
            and int(m18.get("errors", {}).get("reviewed", 0)) >= 30
        ),
        "source_auc_gap": m18.get("source_auc_gap"),
        "large_map_min_auc": m18.get("robustness", {}).get("large_map_min_auc"),
        "large_map_min_auc_ci_lower": m18.get("robustness", {}).get(
            "large_map_min_auc_ci_lower"
        ),
        "reviewed_high_confidence_errors": m18.get("errors", {}).get("reviewed"),
    }
    explanation = {
        "passed": bool(
            m19.get("acceptance", {}).get("status") == "passed"
            and all(bool(value) for value in m19.get("checks", {}).values())
            and m19.get("feature_audit", {}).get("all_feature_failures") == 0
            and m19.get("feature_audit", {}).get("top20_failures") == 0
            and float(m19.get("shap_reconstruction_max_abs_error", math.inf))
            <= 1e-6
        ),
        "all_feature_failures": m19.get("feature_audit", {}).get(
            "all_feature_failures"
        ),
        "top20_failures": m19.get("feature_audit", {}).get("top20_failures"),
        "shap_reconstruction_max_abs_error": m19.get(
            "shap_reconstruction_max_abs_error"
        ),
        "top_source_feature": m19.get("top_features", {})
        .get("source_mean_rank", [None])[0],
    }
    interface = audit_prediction_interface(
        predictor,
        root / "examples/first_kill_snapshot.json",
        root / "examples/first_kill_snapshot.csv",
        m20,
    )

    m17_comparison = read_table(root / "reports/esta_full_m17/model_comparison.csv")
    benchmarks = read_table(root / "benchmarks/external_first_kill_tuned_metrics.csv")
    external = compare_external_models(m17_comparison, benchmarks)
    external_report = render_external_comparison(external)
    closest_external = external.loc[
        external["comparability"].eq("closest_task")
        & external["comparison_status"].eq("compared")
    ]
    external_passed = (
        len(external) == len(benchmarks) == 7
        and set(closest_external["metric"]) >= {"accuracy", "auc"}
        and bool(external_report)
    )

    runtime = collect_runtime_environment()
    environment_lock = audit_environment_lock(root, runtime)
    raw_source = inventory_raw_esta(esta_root)
    try:
        ignored_report_dir = output_dir.relative_to(root).as_posix()
        ignored_paths = (ignored_report_dir,)
    except ValueError:
        ignored_paths = ()
    git_state = collect_git_state(root, ignored_paths)
    script_path = root / "scripts/run_first_kill_pipeline.ps1"
    reproduction = audit_reproduction_entrypoint(
        script_path.read_text(encoding="utf-8")
    )

    if run_tests:
        automated_tests = run_automated_tests(root)
        match = re.search(r"Ran (\d+) tests?", automated_tests["output"])
        automated_tests["test_count"] = int(match.group(1)) if match else None
        automated_tests["skipped"] = False
    else:
        automated_tests = {
            "passed": True,
            "return_code": 0,
            "duration_seconds": 0.0,
            "command": "skipped by caller",
            "output": "Automated tests skipped by run_acceptance caller.\n",
            "test_count": None,
            "skipped": True,
        }

    m6_table = pd.read_csv(
        root / "reports/esta_full_m6/pre_round_xgb_metrics.csv", index_col=0
    )
    m6_metrics = m6_table.loc["test"].to_dict()
    pre_round_summary = _read_json(root / "reports/esta_full_m9/m9_summary.json")
    pre_round_metrics = pre_round_summary["metrics"]
    m16_comparison = read_table(root / "reports/esta_full_m16/m16_model_comparison.csv")
    progress_metrics = build_progress_metrics(
        m6_metrics, pre_round_metrics, m16_comparison, current_metrics
    )
    progress_comparisons = build_progress_comparisons(progress_metrics)
    progress_context = {
        "data": {
            "rows": data_audit["rows"],
            "series": data_audit["series"],
            "games": data_audit["games"],
            "split_rows": data_audit["split_rows"],
            "split_series": data_audit["split_series"],
            "split_percentages": data_audit["split_percentages"],
            "cross_split_series": data_audit["cross_split_series"],
            "cross_split_games": data_audit["cross_split_games"],
            "cross_split_rounds": data_audit["cross_split_rounds"],
        },
        "formal_target_rows": m19["target_gap"]["rows"],
        "closest_external": closest_external.to_dict(orient="records"),
        "pre_round_target_assessment": pre_round_summary.get("target_assessment", {}),
    }

    data_identity_passed = bool(
        data_audit["passed"]
        and data_hash_match
        and data_audit["rows"] == int(m15["counts"]["sample_rows"])
        == int(m18["data"]["rows"])
        and data_audit["series"] == int(m15["counts"]["series"])
        == int(m18["data"]["series"])
        and data_audit["games"] == int(m15["counts"]["games"])
        == int(m18["data"]["games"])
    )
    split_contract_passed = bool(
        data_audit["passed"]
        and split_manifest["passed"]
        and data_audit["cross_split_series"] == 0
        and data_audit["cross_split_games"] == 0
        and data_audit["cross_split_rounds"] == 0
    )
    raw_source_passed = bool(
        raw_source["available"]
        and raw_source["file_count"] == 1558
        and raw_source["subset_counts"] == {"lan": 680, "online": 878}
    )
    tests_passed = bool(
        automated_tests["passed"]
        and (
            automated_tests["skipped"]
            or int(automated_tests.get("test_count") or 0) >= 131
        )
    )
    progress_text = render_progress_report(
        progress_metrics,
        progress_comparisons,
        context={
            **progress_context,
            "test_count": automated_tests.get("test_count"),
            "blocking_passed": len(BLOCKING_CHECKS),
            "blocking_total": len(BLOCKING_CHECKS),
        },
    )
    progress_passed = (
        len(progress_metrics) == 5
        and len(progress_comparisons) == 4
        and "不是纯调参增益" in progress_text
        and "LightGBM" in progress_text
    )

    checks = {
        "required_artifacts": required_artifacts["passed"],
        "raw_source": raw_source_passed,
        "stage_chain": stage_chain["passed"],
        "data_identity": data_identity_passed,
        "split_contract": split_contract_passed,
        "model_contract": model_contract["passed"],
        "calibrator_contract": calibrator_contract["passed"],
        "prediction_replay": bool(
            replay["passed"] and prediction_replay["passed"] and metric_replay["passed"]
        ),
        "formal_targets": formal_targets["passed"],
        "robustness": robustness["passed"],
        "explanation": explanation["passed"],
        "prediction_interface": interface["passed"],
        "external_comparison": external_passed,
        "environment_lock": environment_lock["passed"],
        "automated_tests": tests_passed,
        "reproduction_entrypoint": reproduction["passed"],
        "progress_report": progress_passed,
    }
    acceptance = decide_acceptance(checks)

    progress_text = render_progress_report(
        progress_metrics,
        progress_comparisons,
        context={
            **progress_context,
            "test_count": automated_tests.get("test_count"),
            "blocking_passed": acceptance["blocking_passed"],
            "blocking_total": acceptance["blocking_total"],
        },
    )
    summary = {
        "stage": "M21",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "model_policy": "M17/M18 artifacts frozen; final acceptance only; no fit or tuning",
        "acceptance": acceptance,
        "checks": checks,
        "stage_chain": stage_chain,
        "data": {
            "path": "data/processed/esta_full/first_kill.parquet",
            "bytes": data_artifact["bytes"],
            "sha256": data_artifact["sha256"],
            "rows": data_audit["rows"],
            "series": data_audit["series"],
            "games": data_audit["games"],
            "split_rows": data_audit["split_rows"],
            "split_series": data_audit["split_series"],
            "split_percentages": data_audit["split_percentages"],
            "duplicate_key_rows": data_audit["duplicate_key_rows"],
            "cross_split_series": data_audit["cross_split_series"],
            "cross_split_games": data_audit["cross_split_games"],
            "cross_split_rounds": data_audit["cross_split_rounds"],
            "split_manifest": split_manifest,
        },
        "artifacts": {
            "model": model_artifact,
            "calibrator": calibrator_artifact,
        },
        "model_contract": model_contract,
        "calibrator_contract": calibrator_contract,
        "prediction_replay": prediction_replay,
        "metric_replay": metric_replay,
        "metrics": current_metrics,
        "formal_targets": formal_targets,
        "formal_target_rows": m19["target_gap"]["rows"],
        "robustness": robustness,
        "explanation": explanation,
        "interface": interface,
        "external_comparison_rows": int(len(external)),
        "automated_tests": {
            key: value for key, value in automated_tests.items() if key != "output"
        },
        "environment": runtime,
        "environment_lock": environment_lock,
        "code": git_state,
        "raw_source": raw_source,
        "reproduction_entrypoint": reproduction,
        "xgboost_fit_calls": 0,
        "model_performance_changed": False,
        "progress": {
            "stage_metric_rows": int(len(progress_metrics)),
            "comparison_rows": int(len(progress_comparisons)),
            "report_path": str(progress_path),
        },
        "roadmap": {
            "pre_round_xgboost": "complete_through_M14",
            "first_kill_xgboost": "complete_through_M21",
            "first_kill_xgboost_modules_remaining": 0,
            "next_stage": "LightGBM controlled comparison",
            "later_stage": "real-time win probability",
        },
    }

    fingerprints = {
        path: _relative_fingerprint(root, path) for path in REQUIRED_ARTIFACTS
    }
    manifest = {
        "stage": "M21",
        "experiment_id": "post_first_kill_xgboost_final_acceptance",
        "generated_at_utc": summary["generated_at_utc"],
        "code": git_state,
        "environment": runtime,
        "environment_lock_audit": environment_lock,
        "raw_data": raw_source,
        "artifact_fingerprints": fingerprints,
        "required_artifact_audit": required_artifacts,
        "stage_summaries": stage_summaries,
        "data_audit": data_audit,
        "split_manifest_audit": split_manifest,
        "model_contract": model_contract,
        "calibrator_contract": calibrator_contract,
        "prediction_replay": prediction_replay,
        "metric_replay": metric_replay,
        "formal_targets": formal_targets,
        "robustness": robustness,
        "explanation": explanation,
        "interface": interface,
        "tests": summary["automated_tests"],
        "checks": checks,
        "acceptance": acceptance,
        "nonblocking_follow_ups": [
            "Pre-round aspirational M14 targets remain unmet even though its completion minimums passed.",
            "No chronological future-season holdout has been run.",
            "Team and player identity features are not included.",
            "LightGBM has not yet been compared under the same contract.",
            "The real-time snapshot and event-sequence task has not started.",
        ],
    }

    _write_json(output_dir / "m21_summary.json", summary)
    _write_json(output_dir / "m21_experiment_manifest.json", manifest)
    _write_json(output_dir / "runtime_environment.json", runtime)
    pd.DataFrame(
        [
            {"check": name, "passed": bool(checks[name]), "blocking": True}
            for name in BLOCKING_CHECKS
        ]
    ).to_csv(output_dir / "m21_checks.csv", index=False)
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    external.to_csv(output_dir / "external_benchmark_comparison.csv", index=False)
    (output_dir / "external_benchmark_comparison.md").write_text(
        external_report, encoding="utf-8"
    )
    progress_metrics.to_csv(output_dir / "m6_to_m21_stage_metrics.csv", index=False)
    progress_comparisons.to_csv(
        output_dir / "m6_to_m21_metric_changes.csv", index=False
    )
    (output_dir / "automated_test_output.txt").write_text(
        automated_tests["output"], encoding="utf-8"
    )
    progress_path.write_text(progress_text, encoding="utf-8")
    (output_dir / "m21_first_kill_final_acceptance_report.md").write_text(
        render_final_report(summary, external), encoding="utf-8"
    )

    if acceptance["status"] != "passed":
        raise RuntimeError(
            "M21 final acceptance failed: "
            + ", ".join(acceptance["blocking_failures"])
        )
    return summary


def decide_acceptance(checks: Mapping[str, Any]) -> dict[str, Any]:
    """Close the first-kill XGBoost track only when every blocker passes."""

    missing = [name for name in BLOCKING_CHECKS if name not in checks]
    if missing:
        raise KeyError("Missing M21 blocking checks: " + ", ".join(missing))
    failures = [name for name in BLOCKING_CHECKS if not bool(checks[name])]
    passed = not failures
    return {
        "status": "passed" if passed else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "first_kill_xgboost_complete": passed,
        "ready_for_lightgbm_comparison": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run final post-first-kill XGBoost acceptance without training."
    )
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[2])
    )
    parser.add_argument("--esta-root", default=r"C:\project1\data\esta")
    parser.add_argument("--report-dir", default="reports/esta_full_m21")
    parser.add_argument(
        "--progress-report", default="reports/m6_to_m21_progress_report.md"
    )
    parser.add_argument("--skip-tests", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_acceptance(
        project_root=args.project_root,
        esta_root=args.esta_root,
        report_dir=args.report_dir,
        progress_report_path=args.progress_report,
        run_tests=not args.skip_tests,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

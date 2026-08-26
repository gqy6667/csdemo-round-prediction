from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .io import read_table
from .m12_explanation import (
    build_case_explanations,
    build_importance_comparison,
    select_explanation_cases,
    shap_importance,
)
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m16_first_kill_baselines import (
    REPORT_METRICS,
    canonical_feature_names,
    prepare_profile_splits,
)
from .m19_first_kill_explanation import (
    FIRST_KILL_EVENT_FEATURES,
    audit_post_first_kill_features,
    build_macro_feature_groups,
    build_source_feature_groups,
    build_source_importance_summary,
    grouped_permutation_auc_importance,
    map_encoded_feature_to_source,
)
from .m25_pre_round_lightgbm_explanation import (
    audit_frozen_prediction_replay,
    build_model_importance_comparison,
    encoded_permutation_auc_importance,
    lightgbm_gain_importance as _strict_lightgbm_gain_importance,
    lightgbm_deployment_tree_count,
    lightgbm_tree_shap_contributions,
)
from .m28_post_first_kill_lightgbm_baseline import audit_data_contract, write_json
from .m30_post_first_kill_lightgbm_evaluation import run_compile_check
from .schema import ID_COLUMNS


BLOCKING_CHECKS = (
    "m30_prerequisite",
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


def lightgbm_gain_importance(bundle: Mapping[str, Any]) -> pd.DataFrame:
    expected_columns = list(bundle.get("columns", []))
    if not expected_columns:
        raise KeyError("Model bundle must contain non-empty columns")
    booster = bundle["model"].booster_
    booster_columns = list(booster.feature_name())
    if booster_columns == expected_columns:
        return _strict_lightgbm_gain_importance(bundle)

    sanitized_columns = [column.replace(" ", "_") for column in expected_columns]
    if (
        booster_columns != sanitized_columns
        or len(sanitized_columns) != len(set(sanitized_columns))
    ):
        raise ValueError(
            "LightGBM booster feature names differ from the bundle beyond "
            "documented space sanitization"
        )

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


def verify_m30_prerequisite(
    data_path: str | Path,
    model_path: str | Path,
    m30_summary: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    data_artifact = fingerprint_file(data_path)
    model_artifact = fingerprint_file(model_path)
    expected_data_sha = m30_summary.get("data", {}).get("sha256")
    expected_model_sha = (
        m30_summary.get("prerequisite", {})
        .get("model_artifact", {})
        .get("sha256")
    )
    raw_features = list(bundle.get("raw_features", []))
    encoded_columns = list(bundle.get("columns", []))
    replay = m30_summary.get("model_replay", {})
    deployment_trees = lightgbm_deployment_tree_count(bundle)
    available_trees = int(bundle["model"].booster_.num_trees())
    checks = {
        "m30_accepted": (
            m30_summary.get("acceptance", {}).get("status") == "passed"
            and m30_summary.get("acceptance", {}).get("ready_for_m31") is True
        ),
        "m30_task": m30_summary.get("task") == "post_first_kill",
        "data_sha256": bool(expected_data_sha)
        and data_artifact["sha256"] == expected_data_sha
        and bundle.get("data_sha256") == expected_data_sha,
        "model_sha256": bool(expected_model_sha)
        and model_artifact["sha256"] == expected_model_sha,
        "bundle_task": bundle.get("task") == "post_first_kill",
        "bundle_model_name": bundle.get("model_name") == "lightgbm_tuned",
        "raw_feature_contract": raw_features == canonical_feature_names()
        and len(raw_features) == int(replay.get("raw_feature_count", -1)),
        "encoded_feature_contract": bool(encoded_columns)
        and len(encoded_columns) == len(set(encoded_columns))
        and len(encoded_columns) == int(replay.get("encoded_feature_count", -1)),
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


def audit_reproduction_entrypoint(script_path: Path) -> dict[str, Any]:
    if not script_path.is_file():
        return {"passed": False, "missing_tokens": [script_path.as_posix()]}
    source = script_path.read_text(encoding="utf-8")
    required = (
        "src.csdemo.m31_post_first_kill_lightgbm_explanation",
        "post_first_kill_lightgbm_tuned.joblib",
        "m30_summary.json",
        "[int]$PermutationRepeats = 20",
    )
    missing = [token for token in required if token not in source]
    return {"passed": not missing, "missing_tokens": missing}


def decide_acceptance(checks: Mapping[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not bool(checks.get(name, False))]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "m31_lightgbm_explanation_complete": not failures,
        "ready_for_m32": not failures,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sigmoid(log_odds: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(log_odds, dtype=float), -709, 709)
    return 1.0 / (1.0 + np.exp(-clipped))


def prepare_explanation_inputs(
    data: pd.DataFrame,
    bundle: Mapping[str, Any],
    m30_summary: Mapping[str, Any],
    m30_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    data_audit = audit_data_contract(data)
    expected_rows = {
        name: int(value)
        for name, value in m30_summary.get("data", {}).get("split_rows", {}).items()
    }
    expected_series = {
        name: int(value)
        for name, value in m30_summary.get("data", {}).get("split_series", {}).items()
    }
    if (
        not data_audit["passed"]
        or data_audit["split_rows"] != expected_rows
        or data_audit["split_series"] != expected_series
    ):
        raise RuntimeError("M31 data identity or split contract differs from M30")

    raw_features = list(bundle["raw_features"])
    prepared = prepare_profile_splits(data, raw_features)
    x_test, y_test, identity = prepared["test"]
    encoded_columns = list(bundle["columns"])
    if x_test.columns.tolist() != encoded_columns:
        raise RuntimeError("M31 encoded test columns differ from the frozen model")

    probability = np.asarray(bundle["model"].predict_proba(x_test)[:, 1], dtype=float)
    if (
        len(probability) != len(x_test)
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise RuntimeError("M31 frozen LightGBM produced invalid probabilities")

    predictions = identity[ID_COLUMNS].copy()
    predictions["y_true"] = y_test.to_numpy(dtype=int)
    predictions["ct_win_probability"] = probability
    predictions["t_win_probability"] = 1.0 - probability
    predictions["predicted_label"] = (probability >= 0.5).astype(int)
    predictions["correct"] = predictions["predicted_label"].eq(
        predictions["y_true"]
    )

    metadata_columns = [
        *ID_COLUMNS,
        "map_name",
        "round_num",
        "source_subset",
        "first_kill_time",
        "first_kill_advantage_ct",
        "first_kill_weapon",
        "first_kill_headshot",
    ]
    missing_metadata = sorted(set(metadata_columns) - set(m30_predictions.columns))
    if missing_metadata:
        raise KeyError(f"M30 predictions are missing metadata: {missing_metadata}")
    metadata = m30_predictions[metadata_columns].copy()
    if metadata.duplicated(ID_COLUMNS).any():
        raise RuntimeError("M31 M30 metadata contains duplicate complete keys")
    predictions = predictions.merge(
        metadata,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if predictions[metadata_columns[3:]].isna().any().any():
        raise RuntimeError("M31 could not attach M30 metadata by complete key")

    replay_columns = [*ID_COLUMNS, "y_true", "ct_win_probability"]
    missing_replay = sorted(set(replay_columns) - set(m30_predictions.columns))
    if missing_replay:
        raise KeyError(f"M30 predictions are missing replay columns: {missing_replay}")
    replay = audit_frozen_prediction_replay(
        m30_predictions[replay_columns],
        predictions[replay_columns],
        {name: float(m30_summary["metrics"][name]) for name in REPORT_METRICS},
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
    m30_summary_path: str | Path,
    m30_predictions_path: str | Path,
    *,
    permutation_repeats: int = 20,
    seed: int = 42,
    case_features: int = 10,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    m30_summary = _read_json(m30_summary_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("M31 expected the M29 model artifact to contain a bundle")
    prerequisite = verify_m30_prerequisite(
        data_path,
        model_path,
        m30_summary,
        bundle,
    )
    if not prerequisite["passed"]:
        raise RuntimeError("M31 input does not match accepted M30 artifacts")

    data = read_table(data_path)
    m30_predictions = read_table(m30_predictions_path)
    x_test, y_test, predictions, model_replay = prepare_explanation_inputs(
        data,
        bundle,
        m30_summary,
        m30_predictions,
    )
    if not model_replay["passed"]:
        raise RuntimeError("M31 could not exactly replay the frozen M30 predictions")

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
    encoded_contract = audit_post_first_kill_features(
        encoded_columns,
        raw_features,
    )
    leakage_audit = audit_post_first_kill_features(
        shap["feature"].tolist(),
        raw_features,
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
            "top20_failures": int(top20_audit["audit_result"].eq("fail").sum()),
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


def load_m19_xgboost_importance(
    m19_report_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    report_dir = Path(m19_report_dir)
    tables = {
        "gain": read_table(report_dir / "gain_importance.csv"),
        "permutation": read_table(report_dir / "permutation_importance_auc.csv"),
        "shap": read_table(report_dir / "shap_importance.csv"),
        "saved_comparison": read_table(report_dir / "importance_comparison.csv"),
        "macro": read_table(report_dir / "macro_group_permutation_auc.csv"),
    }
    rebuilt = build_importance_comparison(
        tables["gain"],
        tables["permutation"],
        tables["shap"],
    )
    saved = tables["saved_comparison"].sort_values("feature").reset_index(drop=True)
    rebuilt_sorted = rebuilt.sort_values("feature").reset_index(drop=True)
    exact_columns = ["feature", "gain_rank", "permutation_rank", "shap_rank"]
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
        raise RuntimeError("M19 saved explanation ranks do not match source tables")
    return rebuilt, tables


def validate_external_comparison(
    external: pd.DataFrame,
    m30_metrics: Mapping[str, float],
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
        raise KeyError(f"M30 external comparison is missing columns: {missing}")
    duplicate_ids = int(external["benchmark_id"].duplicated().sum())
    unknown_metrics = sorted(set(external["metric"]) - set(m30_metrics))
    differences = []
    if not unknown_metrics:
        differences = [
            abs(float(row["current_value"]) - float(m30_metrics[row["metric"]]))
            for _, row in external.iterrows()
        ]
    max_difference = max(differences, default=float("inf"))
    numeric = external[
        ["reported_value", "current_value", "performance_advantage_ours"]
    ].to_numpy(dtype=float)
    invalid_numeric = int((~np.isfinite(numeric)).sum())
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
        "current_metric_max_absolute_difference_vs_m30": max_difference,
        "tolerance": tolerance,
    }


def render_m31_report(
    summary: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> str:
    metrics = summary["metrics"]
    source = tables["source_feature_importance"]
    macro = tables["macro_group_permutation_auc"]
    agreement = tables["model_importance_comparison_summary"]
    audit = tables["top20_feature_audit"]
    cases = tables["selected_cases"]
    acceptance = summary["acceptance"]
    lines = [
        "# M31 首杀后 LightGBM 模型解释与泄漏审计",
        "",
        "## 结论",
        "",
        f"M31 阻断检查 {acceptance['blocking_passed']}/{acceptance['blocking_total']} "
        f"通过，状态为 `{acceptance['status']}`。本阶段只解释 M29/M30 已冻结模型，",
        "没有训练、调参、删特征、修改阈值或改变校准方法。",
        "",
        f"测试集仍为 {summary['test_rounds']:,} 回合。Accuracy "
        f"{metrics['accuracy']:.6f}、AUC {metrics['auc']:.6f}、Log Loss "
        f"{metrics['log_loss']:.6f}、Brier {metrics['brier_score']:.6f}、ECE10 "
        f"{metrics['ece10']:.6f}；与 M30 的概率最大差为 "
        f"{summary['model_replay']['max_absolute_probability_difference']:.3e}。",
        "",
        "## 模型与泄漏审计",
        "",
        f"冻结模型部署 {summary['deployment_tree_count']} 棵树。LightGBM 原生 TreeSHAP "
        f"重建概率最大绝对误差为 {summary['shap_reconstruction_max_abs_error']:.3e}；",
        f"模型运行前后 SHA-256 均为 `{summary['model_integrity']['sha256_before']}`。",
        "",
        f"{summary['encoded_features']} 个编码列全部追溯到 "
        f"{summary['raw_features']} 个原始特征；完整审计失败 "
        f"{summary['feature_audit']['all_feature_failures']}，TreeSHAP 前 20 失败 "
        f"{summary['feature_audit']['top20_failures']}。主键、标签、首杀后的伤害/击杀、"
        "下包与回合结束信息、战队和选手身份均未进入模型。",
        "",
        "## 原始特征重要性",
        "",
        "| 特征 | 时点组 | Gain 排名 | 分组置换排名 | SHAP 排名 | 测试 AUC 平均下降 |",
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
            "Gain、固定测试集置换重要性和 TreeSHAP 回答的问题不同，相关经济列与"
            "差值列也会分摊信号。因此排名用于解释冻结模型，不用于依据测试集删特征。",
            "",
            "## 购买结束与首杀事件",
            "",
            "| 时点组 | 编码列数 | AUC 平均下降 | 标准差 |",
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
            "`purchase_end` 是首杀发生前已知的购买和比分状态；`first_kill_event` 仅包含"
            "最早有效敌方击杀的阵营优势、时间、爆头和武器。两组同时打乱的结果不是"
            "因果效应。",
            "",
            "## 与 M19 XGBoost 的解释对照",
            "",
            "| 方法 | 82 列 Spearman | Top 10 交集 | Top 10 Jaccard |",
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
            "两模型使用完全相同的样本、split 和 82 个编码列。排名一致性只描述两种"
            "树模型如何分配同一批信号，不是验收门槛，也不表示特征具有因果作用。",
            "",
            "## TreeSHAP 前 20 泄漏检查",
            "",
            "| 排名 | 编码列 | 原始特征 | 时点组 | 结果 |",
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
            "案例表中的正 SHAP 推向 CT，负 SHAP 推向 T，单位是 log-odds。主键只用于"
            "定位案例，不进入模型。每例的 10 个主要贡献见 `case_explanations.csv`。",
            "",
            "## 验收与下一步",
            "",
            f"自动化测试 {summary['automated_tests']['test_count']} 项通过，源码编译通过；"
            f"外部比较 {summary['external_comparison']['rows']} 行与 M30 原样保留。M31 "
            "证明解释链完整、预测未漂移且没有时间泄漏。下一阶段 M32 建立单条 JSON/CSV "
            "首杀后 LightGBM 推理接口，并复用 M30 由 validation 选择的 identity 校准器。",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    data_path: str | Path,
    model_path: str | Path,
    m30_summary_path: str | Path,
    m30_predictions_path: str | Path,
    m19_report_dir: str | Path,
    m30_external_path: str | Path,
    m30_external_markdown_path: str | Path,
    report_dir: str | Path,
    project_root: str | Path,
    *,
    permutation_repeats: int = 20,
    seed: int = 42,
    case_features: int = 10,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = Path(project_root).resolve()
    data_path = _resolve(root, data_path).resolve()
    model_path = _resolve(root, model_path).resolve()
    m30_summary_path = _resolve(root, m30_summary_path).resolve()
    m30_predictions_path = _resolve(root, m30_predictions_path).resolve()
    m19_report_dir = _resolve(root, m19_report_dir).resolve()
    m30_external_path = _resolve(root, m30_external_path).resolve()
    m30_external_markdown_path = _resolve(
        root, m30_external_markdown_path
    ).resolve()
    report_dir = _resolve(root, report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    model_before = fingerprint_file(model_path)
    core_summary, tables = run_explanation_core(
        data_path,
        model_path,
        m30_summary_path,
        m30_predictions_path,
        permutation_repeats=permutation_repeats,
        seed=seed,
        case_features=case_features,
    )
    m30_summary = _read_json(m30_summary_path)
    xgboost_importance, xgboost_tables = load_m19_xgboost_importance(
        m19_report_dir
    )
    comparison_detail, agreement = build_model_importance_comparison(
        tables["importance_comparison"],
        xgboost_importance,
        top_n=10,
    )
    external = read_table(m30_external_path)
    external_audit = validate_external_comparison(external, m30_summary["metrics"])
    tables.update(
        {
            "xgboost_lightgbm_importance_comparison": comparison_detail,
            "model_importance_comparison_summary": agreement,
            "xgboost_macro_group_permutation_auc": xgboost_tables["macro"],
            "external_benchmark_comparison": external,
        }
    )

    runtime_tables = {"x_test", "shap_values", "external_benchmark_comparison"}
    for name, table in tables.items():
        if name not in runtime_tables:
            table.to_csv(report_dir / f"{name}.csv", index=False)
    shutil.copyfile(
        m30_external_path,
        report_dir / "external_benchmark_comparison.csv",
    )
    shutil.copyfile(
        m30_external_markdown_path,
        report_dir / "external_benchmark_comparison.md",
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
        permutation_repeats == 20
        and len(tables["gain_importance"]) == encoded_features
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
        and len(tables["macro_group_permutation_auc"]) == 2
        and tables["grouped_permutation_importance_auc"]["n_repeats"]
        .eq(permutation_repeats)
        .all()
        and tables["macro_group_permutation_auc"]["n_repeats"]
        .eq(permutation_repeats)
        .all()
        and set(tables["macro_group_permutation_auc"]["feature_group"])
        == {"purchase_end", "first_kill_event"}
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
        and len(xgboost_tables["macro"]) == 2
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
    entrypoint = root / "scripts" / "run_post_first_kill_lightgbm_explanation.ps1"
    entrypoint_audit = audit_reproduction_entrypoint(entrypoint)
    manifest_inputs = [
        data_path,
        model_path,
        m30_summary_path,
        m30_predictions_path,
        m19_report_dir / "gain_importance.csv",
        m19_report_dir / "permutation_importance_auc.csv",
        m19_report_dir / "shap_importance.csv",
        m19_report_dir / "importance_comparison.csv",
        m19_report_dir / "macro_group_permutation_auc.csv",
        m30_external_path,
        m30_external_markdown_path,
        root / "docs" / "m31_post_first_kill_lightgbm_explanation_spec.md",
        root / "src" / "csdemo" / "m31_post_first_kill_lightgbm_explanation.py",
        entrypoint,
    ]
    required_outputs = [
        report_dir / "gain_importance.csv",
        report_dir / "permutation_importance_auc.csv",
        report_dir / "grouped_permutation_importance_auc.csv",
        report_dir / "macro_group_permutation_auc.csv",
        report_dir / "shap_importance.csv",
        report_dir / "model_importance_comparison_summary.csv",
        report_dir / "all_feature_leakage_audit.csv",
        report_dir / "case_explanations.csv",
        report_dir / "external_benchmark_comparison.csv",
        report_dir / "external_benchmark_comparison.md",
    ]
    external_copy_exact = bool(
        fingerprint_file(m30_external_path)["sha256"]
        == fingerprint_file(report_dir / "external_benchmark_comparison.csv")["sha256"]
        and fingerprint_file(m30_external_markdown_path)["sha256"]
        == fingerprint_file(report_dir / "external_benchmark_comparison.md")["sha256"]
    )
    checks = {
        "m30_prerequisite": core_summary["prerequisite"]["passed"],
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
        "external_report": external_audit["passed"] and external_copy_exact,
        "automated_tests": automated_tests["passed"],
        "source_compile": compile_check["passed"],
        "reproduction_entrypoint": entrypoint_audit["passed"],
        "artifact_manifest": all(path.is_file() for path in manifest_inputs)
        and all(path.is_file() for path in required_outputs),
    }
    acceptance = decide_acceptance(checks)
    summary = {
        "stage": "M31",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "model_policy": (
            "M29/M30 LightGBM frozen; no fit, tuning, feature selection, "
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
        "external_comparison": {
            **external_audit,
            "copied_byte_for_byte": external_copy_exact,
        },
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_test_count,
        },
        "source_compile": compile_check,
        "reproduction_entrypoint": entrypoint_audit,
        "environment": {
            "python": sys.version.split()[0],
            "lightgbm": _package_version("lightgbm"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
        },
        "roadmap": {
            "pre_round_xgboost": "complete_through_M14",
            "first_kill_xgboost": "complete_through_M21",
            "pre_round_lightgbm": "complete_through_M27",
            "post_first_kill_lightgbm_current": "M31_explanation_complete",
            "next_stage": "M32 post-first-kill LightGBM JSON/CSV prediction interface",
        },
        "next_stage": "M32 post-first-kill LightGBM prediction interface",
    }

    pd.DataFrame(
        [
            {"check": name, "passed": passed, "blocking": True}
            for name, passed in checks.items()
        ]
    ).to_csv(report_dir / "m31_checks.csv", index=False)
    write_json(summary, report_dir / "m31_summary.json")
    (report_dir / "m31_post_first_kill_lightgbm_explanation_report.md").write_text(
        render_m31_report(summary, tables),
        encoding="utf-8",
    )

    output_names = [
        path.name
        for path in report_dir.iterdir()
        if path.is_file() and path.name != "m31_experiment_manifest.json"
    ]
    manifest = {
        "stage": "M31",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            "python -m src.csdemo.m31_post_first_kill_lightgbm_explanation "
            "--permutation-repeats 20 --seed 42"
        ),
        "policy": "frozen model explanation; no training or test-driven selection",
        "parameters": {
            "permutation_repeats": permutation_repeats,
            "seed": seed,
            "case_features": case_features,
        },
        "inputs": [fingerprint_file(path) for path in manifest_inputs],
        "model_sha256_before": model_before["sha256"],
        "model_sha256_after": model_after["sha256"],
        "outputs": [
            fingerprint_file(report_dir / name) for name in sorted(output_names)
        ],
        "acceptance": acceptance,
        "environment": summary["environment"],
    }
    write_json(manifest, report_dir / "m31_experiment_manifest.json")
    return summary, tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run M31 frozen post-first-kill LightGBM Gain, permutation, "
            "TreeSHAP, leakage audit, and XGBoost explanation comparison."
        )
    )
    parser.add_argument(
        "--data", default="data/processed/esta_full/first_kill.parquet"
    )
    parser.add_argument(
        "--model",
        default="models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib",
    )
    parser.add_argument(
        "--m30-summary", default="reports/esta_full_m30/m30_summary.json"
    )
    parser.add_argument(
        "--m30-predictions",
        default="reports/esta_full_m30/test_predictions_enriched.csv",
    )
    parser.add_argument("--m19-report-dir", default="reports/esta_full_m19")
    parser.add_argument(
        "--m30-external",
        default="reports/esta_full_m30/external_benchmark_comparison.csv",
    )
    parser.add_argument(
        "--m30-external-markdown",
        default="reports/esta_full_m30/external_benchmark_comparison.md",
    )
    parser.add_argument("--report-dir", default="reports/esta_full_m31")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-features", type=int, default=10)
    args = parser.parse_args()

    summary, tables = run(
        data_path=args.data,
        model_path=args.model,
        m30_summary_path=args.m30_summary,
        m30_predictions_path=args.m30_predictions,
        m19_report_dir=args.m19_report_dir,
        m30_external_path=args.m30_external,
        m30_external_markdown_path=args.m30_external_markdown,
        report_dir=args.report_dir,
        project_root=args.project_root,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
        case_features=args.case_features,
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
        tables["model_importance_comparison_summary"].round(6).to_string(
            index=False
        )
    )
    print(
        f"M31 {summary['acceptance']['status']}; "
        f"ready_for_m32={summary['acceptance']['ready_for_m32']}"
    )
    if not summary["acceptance"]["ready_for_m32"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

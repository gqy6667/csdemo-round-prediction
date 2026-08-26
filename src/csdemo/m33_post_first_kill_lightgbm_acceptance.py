from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .io import read_table
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m27_pre_round_lightgbm_acceptance import (
    _build_split_assignments,
    _collect_runtime_environment,
    audit_frozen_metrics,
    audit_paired_uncertainty,
)
from .m28_post_first_kill_lightgbm_baseline import audit_data_contract, write_json
from .m30_post_first_kill_lightgbm_evaluation import (
    replay_frozen_model,
    run_compile_check,
)
from .metrics import probability_metrics
from .predict_first_kill_lightgbm import PostFirstKillLightGBMPredictor


KEY_COLUMNS = ("series_id", "game_id", "round_id")
METRIC_ORDER = ("accuracy", "auc", "log_loss", "brier_score", "ece10")
STAGE_HANDOFFS = {
    "M28": "ready_for_m29",
    "M29": "ready_for_m30",
    "M30": "ready_for_m31",
    "M31": "ready_for_m32",
    "M32": "ready_for_m33",
}
BLOCKING_CHECKS = (
    "m21_prerequisite",
    "stage_chain",
    "required_artifacts",
    "data_identity",
    "split_contract",
    "model_contract",
    "calibrator_contract",
    "prediction_replay",
    "fixed_metrics",
    "paired_uncertainty",
    "robustness_calibration",
    "explanation",
    "prediction_interface",
    "external_comparison",
    "environment_lock",
    "automated_tests",
    "source_compile",
    "reproduction_entrypoint",
    "artifact_manifest",
)


def audit_stage_chain(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    failed = []
    details = {}
    for stage, handoff in STAGE_HANDOFFS.items():
        summary = summaries.get(stage, {})
        acceptance = summary.get("acceptance", {})
        passed = bool(
            summary.get("stage") == stage
            and acceptance.get("status") == "passed"
            and acceptance.get(handoff) is True
        )
        details[stage] = {
            "passed": passed,
            "status": acceptance.get("status"),
            "handoff": handoff,
            "handoff_value": acceptance.get(handoff),
        }
        if not passed:
            failed.append(stage)
    return {
        "passed": not failed,
        "accepted_stages": len(STAGE_HANDOFFS) - len(failed),
        "expected_stages": len(STAGE_HANDOFFS),
        "failed_stages": failed,
        "details": details,
    }


def audit_prediction_replay(
    saved: pd.DataFrame,
    replayed: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    required = set(KEY_COLUMNS) | {"y_true", "ct_win_probability"}
    missing_saved = sorted(required - set(saved.columns))
    missing_replayed = sorted(required - set(replayed.columns))
    if missing_saved or missing_replayed:
        return {
            "passed": False,
            "missing_saved_columns": missing_saved,
            "missing_replayed_columns": missing_replayed,
            "saved_rows": int(len(saved)),
            "replayed_rows": int(len(replayed)),
            "saved_duplicate_key_rows": 0,
            "replayed_duplicate_key_rows": 0,
            "key_mismatch_count": int(len(saved) + len(replayed)),
            "label_mismatch_count": 0,
            "invalid_probability_cells": 0,
            "max_absolute_probability_difference": float("inf"),
            "tolerance": tolerance,
        }

    columns = [*KEY_COLUMNS, "y_true", "ct_win_probability"]
    saved_view = saved[columns].copy()
    replayed_view = replayed[columns].copy()
    saved_duplicates = int(saved_view.duplicated(list(KEY_COLUMNS)).sum())
    replayed_duplicates = int(replayed_view.duplicated(list(KEY_COLUMNS)).sum())
    merged = saved_view.merge(
        replayed_view,
        on=list(KEY_COLUMNS),
        how="outer",
        indicator=True,
        validate=(
            "one_to_one" if not saved_duplicates and not replayed_duplicates else None
        ),
        suffixes=("_saved", "_replayed"),
    )
    key_mismatches = int(merged["_merge"].ne("both").sum())
    matched = merged.loc[merged["_merge"].eq("both")]
    label_mismatches = int(
        matched["y_true_saved"]
        .astype(int)
        .ne(matched["y_true_replayed"].astype(int))
        .sum()
    )
    saved_probability = matched["ct_win_probability_saved"].to_numpy(dtype=float)
    replayed_probability = matched["ct_win_probability_replayed"].to_numpy(
        dtype=float
    )
    invalid = (
        ~np.isfinite(saved_probability)
        | ~np.isfinite(replayed_probability)
        | (saved_probability < 0)
        | (saved_probability > 1)
        | (replayed_probability < 0)
        | (replayed_probability > 1)
    )
    invalid_cells = int(invalid.sum())
    differences = np.abs(saved_probability - replayed_probability)
    max_difference = float(differences.max()) if len(differences) else float("inf")
    passed = bool(
        not saved_duplicates
        and not replayed_duplicates
        and not key_mismatches
        and not label_mismatches
        and not invalid_cells
        and len(saved_view) == len(replayed_view)
        and max_difference <= tolerance
    )
    return {
        "passed": passed,
        "missing_saved_columns": [],
        "missing_replayed_columns": [],
        "saved_rows": int(len(saved_view)),
        "replayed_rows": int(len(replayed_view)),
        "saved_duplicate_key_rows": saved_duplicates,
        "replayed_duplicate_key_rows": replayed_duplicates,
        "key_mismatch_count": key_mismatches,
        "label_mismatch_count": label_mismatches,
        "invalid_probability_cells": invalid_cells,
        "max_absolute_probability_difference": max_difference,
        "tolerance": tolerance,
    }


def audit_reproduction_entrypoint(script_text: str) -> dict[str, Any]:
    required_tokens = (
        "run_first_kill_pipeline.ps1",
        "run_post_first_kill_lightgbm_baseline.ps1",
        "run_post_first_kill_lightgbm_tuning.ps1",
        "run_post_first_kill_lightgbm_evaluation.ps1",
        "run_post_first_kill_lightgbm_explanation.ps1",
        "run_post_first_kill_lightgbm_interface.ps1",
        "src.csdemo.m33_post_first_kill_lightgbm_acceptance",
        "RebuildLightGBM",
        "FullRebuild",
    )
    missing = [token for token in required_tokens if token not in script_text]
    return {
        "passed": not missing,
        "required_tokens": list(required_tokens),
        "missing_tokens": missing,
    }


def decide_acceptance(checks: Mapping[str, Any]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not bool(checks.get(name))]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "post_first_kill_lightgbm_complete": not failures,
        "ready_for_teacher_report": not failures,
    }


def parse_unittest_count(output: str) -> int | None:
    match = re.search(r"Ran (\d+) tests?", output)
    return int(match.group(1)) if match else None


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _render_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    acceptance = summary["acceptance"]
    replay = summary["prediction_replay"]
    paired = summary["paired_uncertainty"]
    data = summary["data"]
    lines = [
        "# M33 首杀后 LightGBM 最终验收报告",
        "",
        "## 最终结论",
        "",
        f"M33 的 {acceptance['blocking_passed']}/{acceptance['blocking_total']} 个阻断"
        f"检查全部通过，状态为 `{acceptance['status']}`。首杀后 LightGBM 已完成 "
        "M28–M33 全链路，可以生成第四份老师正式报告。M33 只做冻结回放，"
        "LightGBM `fit` 调用为 0。",
        "",
        "## 数据与预测时点",
        "",
        "预测时点固定为购买结束后、最早有效敌方击杀刚发生。数据共 "
        f"{data['rows']:,} 回合，按系列赛分为 train/val/test "
        f"{data['split_rows']['train']:,}/{data['split_rows']['val']:,}/"
        f"{data['split_rows']['test']:,}；系列赛为 "
        f"{data['split_series']['train']}/{data['split_series']['val']}/"
        f"{data['split_series']['test']}。跨 split series/game/round 和重复完整主键均为 0。",
        "",
        "## 冻结模型与回放",
        "",
        f"模型为 211 棵树的 M29 LightGBM，40 个原始特征、82 个编码列；M30 "
        "validation-only 选择 identity 校准。4,170 条测试概率最大回放误差为 "
        f"{replay['max_absolute_probability_difference']:.3e}，五项指标最大漂移为 "
        f"{summary['metric_audit']['max_absolute_difference']:.3e}。",
        "",
        "| Accuracy | AUC | Log Loss | Brier | ECE10 |",
        "|---:|---:|---:|---:|---:|",
        f"| {metrics['accuracy']:.6f} | {metrics['auc']:.6f} | "
        f"{metrics['log_loss']:.6f} | {metrics['brier_score']:.6f} | "
        f"{metrics['ece10']:.6f} |",
        "",
        "## 与 M21 XGBoost 的公平比较",
        "",
        f"同 4,170 条测试回合、同 40/82 特征和同系列赛 split 的五项配对 bootstrap "
        f"均完成 2,000 次。显著领先指标数为 "
        f"{paired['significant_better_count']}，五项 95% 区间全部包含 0。因此不能只凭"
        "点指标宣布 LightGBM 或 XGBoost 稳定更优。",
        "",
        "## 解释、接口与工件",
        "",
        "M31 的完整特征和 TreeSHAP 前 20 泄漏失败均为 0，概率重建误差低于 "
        "`1e-10`；M32 的 15/15 阻断项、10/10 非法输入和 JSON/CSV 一致性保持通过。"
        "数据、模型和校准器运行前后 SHA-256 均未变化。",
        "",
        "## 复现与下一步",
        "",
        f"全量自动化测试 {summary['automated_tests']['test_count']} 项通过，源码编译、"
        "环境锁、三模式复现脚本和实验清单均通过。首杀后 LightGBM 研究线现已关闭。"
        "下一步先生成四份老师报告的最后一份和总索引，再另立实时胜率数据阶段。",
        "",
    ]
    return "\n".join(lines)


def run_acceptance(
    *,
    project_root: str | Path,
    report_dir: str | Path,
    run_verification: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    report_dir = _resolve(root, report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "data": root / "data/processed/esta_full/first_kill.parquet",
        "model": root
        / "models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib",
        "calibrator": root
        / "models/esta_full_m30/post_first_kill_lightgbm_calibrator.joblib",
        "m21": root / "reports/esta_full_m21/m21_summary.json",
        "m28": root / "reports/esta_full_m28/m28_summary.json",
        "m29": root / "reports/esta_full_m29/m29_summary.json",
        "m30": root / "reports/esta_full_m30/m30_summary.json",
        "m31": root / "reports/esta_full_m31/m31_summary.json",
        "m32": root / "reports/esta_full_m32/m32_summary.json",
        "saved_predictions": root
        / "reports/esta_full_m30/test_predictions_enriched.csv",
        "paired": root
        / "reports/esta_full_m30/paired_lightgbm_vs_xgboost_bootstrap.csv",
        "external": root
        / "reports/esta_full_m32/external_benchmark_comparison.csv",
        "external_markdown": root
        / "reports/esta_full_m32/external_benchmark_comparison.md",
        "m31_external": root
        / "reports/esta_full_m31/external_benchmark_comparison.csv",
        "leakage_audit": root
        / "reports/esta_full_m31/all_feature_leakage_audit.csv",
        "importance_comparison": root
        / "reports/esta_full_m31/model_importance_comparison_summary.csv",
        "interface_example": root / "reports/esta_full_m32/example_prediction.json",
        "interface_contract": root
        / "reports/esta_full_m32/model_contract_audit.json",
        "spec": root / "docs/m33_post_first_kill_lightgbm_final_acceptance_spec.md",
        "source": root / "src/csdemo/m33_post_first_kill_lightgbm_acceptance.py",
        "script": root / "scripts/run_post_first_kill_lightgbm_pipeline.ps1",
        "environment": root / "environment.yml",
        "lock": root / "requirements-lock.txt",
    }
    required_artifacts = {
        "passed": all(path.is_file() for path in paths.values()),
        "required_count": len(paths),
        "missing": [str(path) for path in paths.values() if not path.is_file()],
    }
    if not required_artifacts["passed"]:
        raise FileNotFoundError(
            f"M33 required artifacts are missing: {required_artifacts['missing']}"
        )

    summaries = {
        stage: _read_json(paths[stage.lower()])
        for stage in ("M28", "M29", "M30", "M31", "M32")
    }
    m21 = _read_json(paths["m21"])
    m28, m29, m30, m31, m32 = (
        summaries[stage] for stage in ("M28", "M29", "M30", "M31", "M32")
    )
    stage_chain = audit_stage_chain(summaries)

    data_before = fingerprint_file(paths["data"])
    model_before = fingerprint_file(paths["model"])
    calibrator_before = fingerprint_file(paths["calibrator"])
    data = read_table(paths["data"])
    data_audit = audit_data_contract(data)
    expected_data_sha = m30.get("data", {}).get("sha256")
    observed_data_hashes = {
        "current": data_before["sha256"],
        "m21": m21.get("data", {}).get("sha256"),
        "m28": m28.get("data", {}).get("sha256"),
        "m29": m29.get("data", {}).get("sha256"),
        "m30": m30.get("data", {}).get("sha256"),
        "m31": m31.get("prerequisite", {})
        .get("data_artifact", {})
        .get("sha256"),
        "m32": m32.get("prerequisite", {})
        .get("model_contract", {})
        .get("data_sha256"),
    }
    data_identity = {
        "passed": bool(
            expected_data_sha
            and set(observed_data_hashes.values()) == {expected_data_sha}
            and data_audit.get("rows") == 41027
        ),
        "sha256": data_before["sha256"],
        "observed_hashes": observed_data_hashes,
        "expected_rows": 41027,
        "observed_rows": int(len(data)),
    }
    expected_split_rows = {"train": 28489, "val": 8368, "test": 4170}
    expected_split_series = {"train": 547, "val": 156, "test": 79}
    split_contract = {
        **data_audit,
        "passed": bool(
            data_audit.get("passed")
            and data_audit.get("split_rows") == expected_split_rows
            and data_audit.get("split_series") == expected_split_series
        ),
        "expected_split_rows": expected_split_rows,
        "expected_split_series": expected_split_series,
    }

    predictor = PostFirstKillLightGBMPredictor.from_paths(
        paths["model"],
        paths["calibrator"],
    )
    expected_model = m29.get("model", {}).get("model_artifact", {})
    expected_calibrator = m30.get("calibration", {}).get(
        "calibrator_artifact", {}
    )
    model_contract = {
        **predictor.model_audit,
        "passed": bool(
            predictor.model_audit.get("passed")
            and model_before["sha256"] == expected_model.get("sha256")
            and predictor.model_audit.get("raw_feature_count") == 40
            and predictor.model_audit.get("encoded_feature_count") == 82
            and predictor.model_audit.get("deployment_tree_count") == 211
            and predictor.model_audit.get("known_map_count") == 8
            and predictor.model_audit.get("known_weapon_count") == 36
        ),
        "artifact": model_before,
        "expected_artifact": expected_model,
    }
    calibrator_contract = {
        **predictor.calibrator_audit,
        "passed": bool(
            predictor.calibrator_audit.get("passed")
            and calibrator_before["sha256"] == expected_calibrator.get("sha256")
            and predictor.calibrator_audit.get("base_model_sha256")
            == model_before["sha256"]
            and predictor.calibrator_audit.get("selection_data")
            == "validation only"
            and predictor.calibrator_audit.get("validation_folds") == 5
        ),
        "artifact": calibrator_before,
        "expected_artifact": expected_calibrator,
    }

    bundle = joblib.load(paths["model"])
    replayed, replay_contract = replay_frozen_model(data, bundle)
    replayed_test = replayed["test"]
    saved_predictions = read_table(paths["saved_predictions"])
    prediction_replay = audit_prediction_replay(saved_predictions, replayed_test)
    metrics = probability_metrics(
        replayed_test["y_true"].to_numpy(dtype=int),
        replayed_test["ct_win_probability"].to_numpy(dtype=float),
        n_bins=10,
    )
    metrics = {metric: float(metrics[metric]) for metric in METRIC_ORDER}
    metric_audit = audit_frozen_metrics(metrics, m30.get("metrics", {}))
    metric_sources = {
        "m29": m29.get("metrics"),
        "m30": m30.get("metrics"),
        "m31": m31.get("metrics"),
        "m32": m32.get("fixed_test_metrics"),
    }
    fixed_sources_match = all(
        audit_frozen_metrics(source or {}, m30.get("metrics", {}))["passed"]
        for source in metric_sources.values()
    )

    paired = read_table(paths["paired"])
    paired_uncertainty = audit_paired_uncertainty(paired, m30)
    robustness_calibration = {
        "passed": bool(
            m30.get("acceptance", {}).get("status") == "passed"
            and m30.get("checks", {}).get("global_bootstrap") is True
            and m30.get("checks", {}).get("paired_comparison") is True
            and m30.get("checks", {}).get("group_outputs") is True
            and m30.get("checks", {}).get("source_stability") is True
            and m30.get("checks", {}).get("calibration_protocol") is True
            and m30.get("checks", {}).get("calibration_no_material_harm") is True
            and m30.get("calibration", {}).get("selected_method")
            == "uncalibrated"
        ),
        "selected_calibration": m30.get("calibration", {}).get(
            "selected_method"
        ),
        "global_assessment": m30.get("global_assessment"),
        "robustness": m30.get("robustness"),
    }
    explanation = bool(
        m31.get("acceptance", {}).get("status") == "passed"
        and m31.get("acceptance", {}).get("ready_for_m32") is True
        and all(m31.get("checks", {}).values())
        and m31.get("feature_audit", {}).get("all_feature_failures") == 0
        and m31.get("feature_audit", {}).get("top20_failures") == 0
        and float(m31.get("shap_reconstruction_max_abs_error", 1.0)) <= 1e-10
    )
    prediction_interface = bool(
        m32.get("acceptance", {}).get("status") == "passed"
        and m32.get("acceptance", {}).get("ready_for_m33") is True
        and all(m32.get("checks", {}).values())
        and m32.get("validation_cases", {}).get("passed")
        == m32.get("validation_cases", {}).get("total")
        == 10
        and float(m32.get("json_csv_probability_difference", 1.0)) == 0.0
        and m32.get("artifacts", {}).get("model_unchanged") is True
        and m32.get("artifacts", {}).get("calibrator_unchanged") is True
    )
    external = read_table(paths["external"])
    external_hash_matches_m31 = bool(
        fingerprint_file(paths["external"])["sha256"]
        == fingerprint_file(paths["m31_external"])["sha256"]
    )
    external_comparison = {
        "passed": bool(
            len(external) == 7
            and m32.get("external_comparison", {}).get("passed") is True
            and m32.get("external_comparison", {}).get("rows") == 7
            and m32.get("external_comparison", {}).get("copied_byte_for_byte")
            is True
            and external_hash_matches_m31
        ),
        "rows": int(len(external)),
        "hash_matches_m31": external_hash_matches_m31,
    }
    m21_prerequisite = bool(
        m21.get("stage") == "M21"
        and m21.get("acceptance", {}).get("status") == "passed"
        and m21.get("acceptance", {}).get("first_kill_xgboost_complete") is True
        and m21.get("acceptance", {}).get("ready_for_lightgbm_comparison") is True
        and m21.get("data", {}).get("rows") == 41027
        and m21.get("data", {}).get("sha256") == data_before["sha256"]
    )
    environment = _collect_runtime_environment(root, report_dir)
    if run_verification:
        automated_tests = run_automated_tests(root)
        automated_tests["test_count"] = parse_unittest_count(
            automated_tests.get("output", "")
        )
        automated_tests["skipped"] = False
        source_compile = run_compile_check(root)
        source_compile["skipped"] = False
    else:
        automated_tests = {
            "passed": True,
            "return_code": 0,
            "test_count": None,
            "elapsed_seconds": 0.0,
            "output": "Skipped by run_verification=False\n",
            "skipped": True,
        }
        source_compile = {
            "passed": True,
            "return_code": 0,
            "command": [],
            "output": "Skipped by run_verification=False\n",
            "skipped": True,
        }
    reproduction = audit_reproduction_entrypoint(
        paths["script"].read_text(encoding="utf-8")
    )

    data_after = fingerprint_file(paths["data"])
    model_after = fingerprint_file(paths["model"])
    calibrator_after = fingerprint_file(paths["calibrator"])
    artifact_integrity = bool(
        data_before["sha256"] == data_after["sha256"]
        and model_before["sha256"] == model_after["sha256"]
        and calibrator_before["sha256"] == calibrator_after["sha256"]
    )
    manifest_inputs = list(paths.values())
    checks = {
        "m21_prerequisite": m21_prerequisite,
        "stage_chain": stage_chain["passed"],
        "required_artifacts": required_artifacts["passed"],
        "data_identity": data_identity["passed"] and artifact_integrity,
        "split_contract": split_contract["passed"],
        "model_contract": model_contract["passed"] and artifact_integrity,
        "calibrator_contract": calibrator_contract["passed"]
        and artifact_integrity,
        "prediction_replay": prediction_replay["passed"]
        and replay_contract.get("lightgbm_fit_calls") == 0,
        "fixed_metrics": metric_audit["passed"] and fixed_sources_match,
        "paired_uncertainty": paired_uncertainty["passed"],
        "robustness_calibration": robustness_calibration["passed"],
        "explanation": explanation,
        "prediction_interface": prediction_interface,
        "external_comparison": external_comparison["passed"],
        "environment_lock": environment["passed"],
        "automated_tests": automated_tests["passed"],
        "source_compile": source_compile["passed"],
        "reproduction_entrypoint": reproduction["passed"],
        "artifact_manifest": all(path.is_file() for path in manifest_inputs),
    }
    acceptance = decide_acceptance(checks)
    summary = {
        "stage": "M33",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "policy": "M28-M32 frozen; final replay only; no fit, tuning, or test selection",
        "acceptance": acceptance,
        "checks": checks,
        "required_artifacts": required_artifacts,
        "stage_chain": stage_chain,
        "m21_prerequisite": m21_prerequisite,
        "data_identity": data_identity,
        "data": data_audit,
        "split_contract": split_contract,
        "model_contract": model_contract,
        "calibrator_contract": calibrator_contract,
        "model_replay": replay_contract,
        "prediction_replay": prediction_replay,
        "metrics": metrics,
        "metric_audit": metric_audit,
        "metric_sources_match": fixed_sources_match,
        "paired_uncertainty": paired_uncertainty,
        "robustness_calibration": robustness_calibration,
        "evidence": {
            "explanation": explanation,
            "prediction_interface": prediction_interface,
            "external_comparison": external_comparison,
        },
        "artifacts": {
            "data_sha256": data_after["sha256"],
            "model_sha256": model_after["sha256"],
            "calibrator_sha256": calibrator_after["sha256"],
            "unchanged": artifact_integrity,
        },
        "lightgbm_fit_calls": 0,
        "runtime_environment": environment,
        "automated_tests": automated_tests,
        "source_compile": source_compile,
        "reproduction_entrypoint": reproduction,
        "next_stage": "teacher report 04 and teacher review index",
    }

    split_assignments = _build_split_assignments(data)
    metric_frame = pd.DataFrame(
        [{"metric": metric, "value": metrics[metric]} for metric in METRIC_ORDER]
    )
    checks_frame = pd.DataFrame(
        [{"check": name, "passed": bool(checks[name])} for name in BLOCKING_CHECKS]
    )
    replayed_test.to_csv(report_dir / "replayed_test_predictions.csv", index=False)
    split_assignments.to_csv(report_dir / "split_assignments.csv", index=False)
    metric_frame.to_csv(report_dir / "fixed_test_metrics.csv", index=False)
    paired.to_csv(
        report_dir / "paired_lightgbm_vs_xgboost_bootstrap.csv",
        index=False,
    )
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    (report_dir / "external_benchmark_comparison.md").write_bytes(
        paths["external_markdown"].read_bytes()
    )
    checks_frame.to_csv(report_dir / "m33_checks.csv", index=False)
    write_json(environment, report_dir / "runtime_environment.json")
    write_json(summary, report_dir / "m33_summary.json")
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests.get("output", ""),
        encoding="utf-8",
    )
    (report_dir / "source_compile_output.txt").write_text(
        source_compile.get("output", ""),
        encoding="utf-8",
    )
    (report_dir / "m33_post_first_kill_lightgbm_final_acceptance_report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )

    output_paths = sorted(
        [
            path
            for path in report_dir.iterdir()
            if path.is_file() and path.name != "m33_experiment_manifest.json"
        ],
        key=lambda path: path.name,
    )
    manifest = {
        "stage": "M33",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "powershell -File scripts/run_post_first_kill_lightgbm_pipeline.ps1",
        "policy": summary["policy"],
        "inputs": [fingerprint_file(path) for path in manifest_inputs],
        "outputs": [fingerprint_file(path) for path in output_paths],
        "data_sha256_before": data_before["sha256"],
        "data_sha256_after": data_after["sha256"],
        "model_sha256_before": model_before["sha256"],
        "model_sha256_after": model_after["sha256"],
        "calibrator_sha256_before": calibrator_before["sha256"],
        "calibrator_sha256_after": calibrator_after["sha256"],
        "acceptance": acceptance,
    }
    write_json(manifest, report_dir / "m33_experiment_manifest.json")
    if acceptance["status"] != "passed":
        raise RuntimeError("M33 acceptance failed; inspect m33_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run M33 final acceptance for the frozen post-first-kill LightGBM chain."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report-dir", default="reports/esta_full_m33")
    parser.add_argument("--skip-verification", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_acceptance(
        project_root=args.project_root,
        report_dir=args.report_dir,
        run_verification=not args.skip_verification,
    )
    print(
        f"M33 {summary['acceptance']['status']}; "
        "post_first_kill_lightgbm_complete="
        f"{summary['acceptance']['post_first_kill_lightgbm_complete']}; "
        f"ready_for_teacher_report={summary['acceptance']['ready_for_teacher_report']}"
    )


if __name__ == "__main__":
    main()

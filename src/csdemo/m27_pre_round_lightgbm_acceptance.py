from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .io import read_table
from .m14_acceptance import collect_git_state
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m22_pre_round_lightgbm_baseline import audit_data_contract, write_json
from .m24_pre_round_lightgbm_evaluation import (
    replay_frozen_model,
    run_compile_check,
)
from .metrics import probability_metrics
from .predict_pre_round_lightgbm import PreRoundLightGBMPredictor


KEY_COLUMNS = ("series_id", "game_id", "round_id")
METRIC_ORDER = ("accuracy", "auc", "log_loss", "brier_score", "ece10")
STAGE_HANDOFFS = {
    "M22": "ready_for_m23",
    "M23": "ready_for_m24",
    "M24": "ready_for_m25",
    "M25": "ready_for_m26",
    "M26": "ready_for_m27",
}
BLOCKING_CHECKS = (
    "m14_prerequisite",
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


def _missing_columns(frame: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(frame.columns))


def audit_prediction_replay(
    saved: pd.DataFrame,
    replayed: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    saved_required = set(KEY_COLUMNS) | {"ct_win", "lightgbm_tuned_probability"}
    replay_required = set(KEY_COLUMNS) | {"y_true", "ct_win_probability"}
    missing_saved = _missing_columns(saved, saved_required)
    missing_replayed = _missing_columns(replayed, replay_required)
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

    saved_view = saved[list(KEY_COLUMNS) + ["ct_win", "lightgbm_tuned_probability"]].copy()
    replay_view = replayed[list(KEY_COLUMNS) + ["y_true", "ct_win_probability"]].copy()
    saved_duplicates = int(saved_view.duplicated(list(KEY_COLUMNS)).sum())
    replayed_duplicates = int(replay_view.duplicated(list(KEY_COLUMNS)).sum())
    merged = saved_view.merge(
        replay_view,
        on=list(KEY_COLUMNS),
        how="outer",
        indicator=True,
        validate="one_to_one" if not saved_duplicates and not replayed_duplicates else None,
    )
    key_mismatches = int(merged["_merge"].ne("both").sum())
    matched = merged.loc[merged["_merge"].eq("both")]
    label_mismatches = int(
        matched["ct_win"].astype(int).ne(matched["y_true"].astype(int)).sum()
    )
    saved_probability = matched["lightgbm_tuned_probability"].to_numpy(dtype=float)
    replay_probability = matched["ct_win_probability"].to_numpy(dtype=float)
    invalid = (
        ~np.isfinite(saved_probability)
        | ~np.isfinite(replay_probability)
        | (saved_probability < 0)
        | (saved_probability > 1)
        | (replay_probability < 0)
        | (replay_probability > 1)
    )
    invalid_cells = int(invalid.sum())
    differences = np.abs(saved_probability - replay_probability)
    max_difference = float(differences.max()) if len(differences) else float("inf")
    passed = bool(
        not saved_duplicates
        and not replayed_duplicates
        and not key_mismatches
        and not label_mismatches
        and not invalid_cells
        and len(saved_view) == len(replay_view)
        and max_difference <= tolerance
    )
    return {
        "passed": passed,
        "missing_saved_columns": [],
        "missing_replayed_columns": [],
        "saved_rows": int(len(saved_view)),
        "replayed_rows": int(len(replay_view)),
        "saved_duplicate_key_rows": saved_duplicates,
        "replayed_duplicate_key_rows": replayed_duplicates,
        "key_mismatch_count": key_mismatches,
        "label_mismatch_count": label_mismatches,
        "invalid_probability_cells": invalid_cells,
        "max_absolute_probability_difference": max_difference,
        "tolerance": tolerance,
    }


def audit_frozen_metrics(
    current: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    required = set(METRIC_ORDER)
    current_keys = set(current)
    expected_keys = set(expected)
    missing_current = sorted(required - current_keys)
    missing_expected = sorted(required - expected_keys)
    differences = {}
    for metric in METRIC_ORDER:
        if metric in current and metric in expected:
            differences[metric] = abs(float(current[metric]) - float(expected[metric]))
    max_difference = max(differences.values(), default=float("inf"))
    return {
        "passed": bool(
            not missing_current
            and not missing_expected
            and set(differences) == required
            and np.isfinite(list(differences.values())).all()
            and max_difference <= tolerance
        ),
        "missing_current_metrics": missing_current,
        "missing_expected_metrics": missing_expected,
        "absolute_differences": differences,
        "max_absolute_difference": float(max_difference),
        "tolerance": tolerance,
    }


def audit_paired_uncertainty(
    table: pd.DataFrame,
    m24_summary: Mapping[str, Any],
) -> dict[str, Any]:
    required_columns = {
        "metric",
        "performance_advantage_ci_lower_95",
        "performance_advantage_ci_upper_95",
        "ci_includes_zero",
        "lightgbm_significantly_better",
        "successful_bootstraps",
        "bootstrap_unit",
    }
    missing = sorted(required_columns - set(table.columns))
    if missing:
        return {
            "passed": False,
            "missing_columns": missing,
            "metric_count": 0,
            "significant_better_count": 0,
        }

    expected_samples = int(m24_summary.get("bootstrap", {}).get("samples", -1))
    expected_significant = int(
        m24_summary.get("paired_comparison", {}).get(
            "significant_better_count", -1
        )
    )
    metrics = table["metric"].astype(str)
    lower = pd.to_numeric(table["performance_advantage_ci_lower_95"], errors="coerce")
    upper = pd.to_numeric(table["performance_advantage_ci_upper_95"], errors="coerce")
    includes_zero = table["ci_includes_zero"].astype(bool)
    significant = table["lightgbm_significantly_better"].astype(bool)
    observed_contains_zero = lower.le(0) & upper.ge(0)
    significant_count = int(significant.sum())
    checks = {
        "five_unique_metrics": set(metrics) == set(METRIC_ORDER)
        and len(table) == len(METRIC_ORDER)
        and not metrics.duplicated().any(),
        "finite_ordered_intervals": bool(
            lower.notna().all() and upper.notna().all() and lower.le(upper).all()
        ),
        "zero_flags_match_intervals": includes_zero.equals(observed_contains_zero),
        "all_intervals_include_zero": bool(includes_zero.all()),
        "no_significant_superiority": significant_count == 0,
        "matches_m24_claim": significant_count == expected_significant == 0,
        "bootstrap_count": bool(
            expected_samples == 2000
            and table["successful_bootstraps"].astype(int).eq(expected_samples).all()
        ),
        "bootstrap_unit": table["bootstrap_unit"].eq("series_id_paired").all(),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_columns": [],
        "metric_count": int(len(table)),
        "significant_better_count": significant_count,
        "expected_bootstraps": expected_samples,
    }


def audit_reproduction_entrypoint(script_text: str) -> dict[str, Any]:
    required_tokens = (
        "run_pre_round_pipeline.ps1",
        "run_pre_round_lightgbm_baseline.ps1",
        "run_pre_round_lightgbm_tuning.ps1",
        "run_pre_round_lightgbm_evaluation.ps1",
        "run_pre_round_lightgbm_explanation.ps1",
        "run_pre_round_lightgbm_interface.ps1",
        "src.csdemo.m27_pre_round_lightgbm_acceptance",
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
        "pre_round_lightgbm_complete": not failures,
        "ready_for_m28": not failures,
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


def _collect_runtime_environment(root: Path, report_dir: Path) -> dict[str, Any]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "pyarrow",
        "joblib",
        "matplotlib",
    )
    versions = {name: importlib.metadata.version(name) for name in packages}
    locked = {}
    for raw_line in (root / "requirements-lock.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            locked[name.lower()] = version
    mismatches = [
        {"package": name, "locked": locked.get(name), "runtime": version}
        for name, version in versions.items()
        if locked.get(name) != version
    ]
    environment_text = (root / "environment.yml").read_text(encoding="utf-8")
    python_version = platform.python_version()
    python_locked = f"python={python_version}" in environment_text
    try:
        report_relative = str(report_dir.relative_to(root))
    except ValueError:
        ignored_output_paths: tuple[str, ...] = ()
    else:
        ignored_output_paths = (report_relative,)
    git = collect_git_state(root, ignored_output_paths=ignored_output_paths)
    return {
        "passed": not mismatches and python_locked,
        "python_version": python_version,
        "python_executable": sys.executable,
        "environment_prefix": sys.prefix,
        "platform": platform.platform(),
        "packages": versions,
        "locked_packages": {name: locked.get(name) for name in packages},
        "package_mismatches": mismatches,
        "python_lock_matches": python_locked,
        "lightgbm_device": "cpu",
        "cuda_required": False,
        "git": git,
    }


def _build_split_assignments(data: pd.DataFrame) -> pd.DataFrame:
    counts = data.groupby("series_id")["split"].nunique()
    if counts.gt(1).any():
        raise ValueError("A series_id appears in more than one split")
    return (
        data[["series_id", "split"]]
        .drop_duplicates()
        .sort_values(["split", "series_id"])
        .reset_index(drop=True)
    )


def _render_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    replay = summary["prediction_replay"]
    paired = summary["paired_uncertainty"]
    data = summary["data"]
    acceptance = summary["acceptance"]
    tests = summary["automated_tests"]
    return f"""# M27 购买结束 LightGBM 最终验收报告

## 最终结论

验收状态：**{acceptance['status']}**；阻断项：
**{acceptance['blocking_passed']}/{acceptance['blocking_total']}**；购买结束 LightGBM
完成：**{acceptance['pre_round_lightgbm_complete']}**。

M27 只回放 M23 冻结模型、M24 identity 校准器和 M26 接口，没有训练、调参、特征选择
或 test 驱动的模型变更。LightGBM `fit()` 调用为 **{summary['lightgbm_fit_calls']}**。

## 数据与切分

- 回合：{data['rows']:,}；series：{data['series']:,}；game：{data['games']:,}；
- train/validation/test：{data['split_rows']['train']:,} /
  {data['split_rows']['val']:,} / {data['split_rows']['test']:,}；
- 跨 split series/game/round：{data['cross_split_series']} /
  {data['cross_split_games']} / {data['cross_split_rounds']}；
- 重复完整主键：{data['duplicate_key_rows']}。

## 冻结测试指标

| Accuracy | AUC | Log Loss | Brier | ECE10 |
|---:|---:|---:|---:|---:|
| {metrics['accuracy']:.6f} | {metrics['auc']:.6f} | {metrics['log_loss']:.6f} | {metrics['brier_score']:.6f} | {metrics['ece10']:.6f} |

4,172 条测试概率最大回放误差为
`{replay['max_absolute_probability_difference']:.3e}`；五项指标最大误差为
`{summary['metric_audit']['max_absolute_difference']:.3e}`。

## 与 XGBoost 的统计结论

M24 保存的五项系列赛级配对 bootstrap 均完成 2,000 次。显著领先指标数为
**{paired['significant_better_count']}**，五项 95% 区间都包含 0。因此只能表述为
LightGBM 点指标略好，不能宣称其稳定或显著优于 XGBoost。

## 工件与接口

- 原始/编码特征：36/43；部署树：115；
- 模型 SHA-256：`{summary['artifacts']['model_sha256']}`；
- 校准器 SHA-256：`{summary['artifacts']['calibrator_sha256']}`；
- M25 泄漏与解释检查通过：{summary['evidence']['explanation']}；
- M26 JSON/CSV 接口检查通过：{summary['evidence']['prediction_interface']}。

## 可复现性与下一步

自动化测试：{tests.get('test_count')}；源码编译：
{summary['source_compile']['passed']}；三模式一键入口：
{summary['reproduction_entrypoint']['passed']}。

M27 通过后进入 M28：保持 M21 首杀后数据、70/20/10 系列赛切分、预测时点、特征和
指标不变，只将 XGBoost 替换为固定 LightGBM 基线，并做系列赛级配对不确定性比较。
"""


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
        "data": root / "data/processed/esta_full/pre_round.parquet",
        "model": root / "models/esta_full_m23/pre_round_lightgbm_tuned.joblib",
        "calibrator": root
        / "models/esta_full_m24/pre_round_lightgbm_calibrator.joblib",
        "m14": root / "reports/esta_full_m14/m14_summary.json",
        "m22": root / "reports/esta_full_m22/m22_summary.json",
        "m23": root / "reports/esta_full_m23/m23_summary.json",
        "m24": root / "reports/esta_full_m24/m24_summary.json",
        "m25": root / "reports/esta_full_m25/m25_summary.json",
        "m26": root / "reports/esta_full_m26/m26_summary.json",
        "saved_predictions": root / "reports/esta_full_m23/test_predictions.csv",
        "paired": root
        / "reports/esta_full_m24/paired_lightgbm_vs_xgboost_bootstrap.csv",
        "external": root / "reports/esta_full_m26/external_benchmark_comparison.csv",
        "spec": root / "docs/m27_pre_round_lightgbm_final_acceptance_spec.md",
        "source": root / "src/csdemo/m27_pre_round_lightgbm_acceptance.py",
        "script": root / "scripts/run_pre_round_lightgbm_pipeline.ps1",
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
            f"M27 required artifacts are missing: {required_artifacts['missing']}"
        )

    summaries = {
        stage: _read_json(paths[stage.lower()])
        for stage in ("M22", "M23", "M24", "M25", "M26")
    }
    m14 = _read_json(paths["m14"])
    m22, m23, m24, m25, m26 = (
        summaries[stage] for stage in ("M22", "M23", "M24", "M25", "M26")
    )
    stage_chain = audit_stage_chain(summaries)

    data_before = fingerprint_file(paths["data"])
    model_before = fingerprint_file(paths["model"])
    calibrator_before = fingerprint_file(paths["calibrator"])
    data = read_table(paths["data"])
    data_audit = audit_data_contract(data)
    expected_data_sha = m22.get("data", {}).get("sha256")
    observed_data_hashes = {
        "current": data_before["sha256"],
        "m22": m22.get("data", {}).get("sha256"),
        "m23": m23.get("data", {}).get("sha256"),
        "m24": m24.get("data", {}).get("sha256"),
        "m25": m25.get("prerequisite", {}).get("data_artifact", {}).get("sha256"),
        "m26": m26.get("prerequisite", {})
        .get("model_contract", {})
        .get("data_sha256"),
    }
    data_identity = {
        "passed": bool(
            expected_data_sha
            and set(observed_data_hashes.values()) == {expected_data_sha}
            and data_audit.get("rows") == 41074
        ),
        "sha256": data_before["sha256"],
        "observed_hashes": observed_data_hashes,
        "expected_rows": 41074,
        "observed_rows": int(len(data)),
    }
    expected_split_rows = {"train": 28522, "val": 8380, "test": 4172}
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

    predictor = PreRoundLightGBMPredictor.from_paths(
        paths["model"], paths["calibrator"]
    )
    expected_model = m23.get("model", {}).get("model_artifact", {})
    expected_calibrator = m24.get("calibration", {}).get("calibrator_artifact", {})
    model_contract = {
        **predictor.model_audit,
        "passed": bool(
            predictor.model_audit.get("passed")
            and model_before["sha256"] == expected_model.get("sha256")
            and predictor.model_audit.get("raw_feature_count") == 36
            and predictor.model_audit.get("encoded_feature_count") == 43
            and predictor.model_audit.get("deployment_tree_count") == 115
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
        ),
        "artifact": calibrator_before,
        "expected_artifact": expected_calibrator,
    }

    bundle = joblib.load(paths["model"])
    replayed, replay_contract = replay_frozen_model(data, bundle)
    replayed_test = replayed["test"]
    saved_predictions = pd.read_csv(paths["saved_predictions"])
    prediction_replay = audit_prediction_replay(saved_predictions, replayed_test)
    metrics = probability_metrics(
        replayed_test["y_true"].to_numpy(dtype=int),
        replayed_test["ct_win_probability"].to_numpy(dtype=float),
        n_bins=10,
    )
    metrics = {metric: float(metrics[metric]) for metric in METRIC_ORDER}
    metric_audit = audit_frozen_metrics(metrics, m24.get("metrics", {}))
    metric_sources = {
        "m22": m22.get("metrics"),
        "m23": m23.get("metrics"),
        "m24": m24.get("metrics"),
        "m25": m25.get("metrics"),
        "m26": m26.get("fixed_test_metrics"),
    }
    fixed_sources_match = all(
        audit_frozen_metrics(source or {}, m24.get("metrics", {}))["passed"]
        for source in metric_sources.values()
    )

    paired = pd.read_csv(paths["paired"])
    paired_uncertainty = audit_paired_uncertainty(paired, m24)
    robustness_calibration = {
        "passed": bool(
            m24.get("acceptance", {}).get("status") == "passed"
            and m24.get("checks", {}).get("global_bootstrap") is True
            and m24.get("checks", {}).get("group_outputs") is True
            and m24.get("checks", {}).get("source_stability") is True
            and m24.get("checks", {}).get("calibration_protocol") is True
            and m24.get("checks", {}).get("calibration_no_material_harm") is True
            and m24.get("calibration", {}).get("selected_method") == "uncalibrated"
        ),
        "selected_calibration": m24.get("calibration", {}).get("selected_method"),
        "global_assessment": m24.get("global_assessment"),
        "robustness": m24.get("robustness"),
    }
    explanation = bool(
        m25.get("acceptance", {}).get("status") == "passed"
        and all(m25.get("checks", {}).values())
        and m25.get("feature_audit", {}).get("all_feature_failures") == 0
        and m25.get("feature_audit", {}).get("top20_failures") == 0
    )
    prediction_interface = bool(
        m26.get("acceptance", {}).get("status") == "passed"
        and m26.get("acceptance", {}).get("ready_for_m27") is True
        and all(m26.get("checks", {}).values())
        and m26.get("validation_cases", {}).get("passed")
        == m26.get("validation_cases", {}).get("total")
        == 10
        and float(m26.get("json_csv_probability_difference", 1.0)) == 0.0
    )
    external = pd.read_csv(paths["external"])
    external_comparison = {
        "passed": bool(
            len(external) == 4
            and m26.get("external_comparison", {}).get("passed") is True
            and m26.get("external_comparison", {}).get("rows") == 4
        ),
        "rows": int(len(external)),
    }
    m14_prerequisite = bool(
        m14.get("stage") == "M14"
        and m14.get("status") == "passed"
        and m14.get("phase_1_pre_round_xgboost_complete") is True
        and m14.get("pre_round_rows") == 41074
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
    script_text = paths["script"].read_text(encoding="utf-8")
    reproduction = audit_reproduction_entrypoint(script_text)

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
        "m14_prerequisite": m14_prerequisite,
        "stage_chain": stage_chain["passed"],
        "required_artifacts": required_artifacts["passed"],
        "data_identity": data_identity["passed"] and artifact_integrity,
        "split_contract": split_contract["passed"],
        "model_contract": model_contract["passed"] and artifact_integrity,
        "calibrator_contract": calibrator_contract["passed"] and artifact_integrity,
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
        "stage": "M27",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "policy": "M22-M26 frozen; final replay only; no fit, tuning, or test selection",
        "acceptance": acceptance,
        "checks": checks,
        "required_artifacts": required_artifacts,
        "stage_chain": stage_chain,
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
        "next_stage": "M28 post-first-kill LightGBM controlled baseline",
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
        report_dir / "paired_lightgbm_vs_xgboost_bootstrap.csv", index=False
    )
    checks_frame.to_csv(report_dir / "m27_checks.csv", index=False)
    write_json(environment, report_dir / "runtime_environment.json")
    write_json(summary, report_dir / "m27_summary.json")
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests.get("output", ""), encoding="utf-8"
    )
    (report_dir / "source_compile_output.txt").write_text(
        source_compile.get("output", ""), encoding="utf-8"
    )
    (report_dir / "m27_pre_round_lightgbm_final_acceptance_report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )

    output_paths = sorted(
        [
            path
            for path in report_dir.iterdir()
            if path.is_file() and path.name != "m27_experiment_manifest.json"
        ],
        key=lambda path: path.name,
    )
    manifest = {
        "stage": "M27",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "powershell -File scripts/run_pre_round_lightgbm_pipeline.ps1",
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
    write_json(manifest, report_dir / "m27_experiment_manifest.json")
    if acceptance["status"] != "passed":
        raise RuntimeError("M27 acceptance failed; inspect m27_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run final acceptance for frozen pre-round LightGBM artifacts."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--report-dir", default="reports/esta_full_m27")
    parser.add_argument("--skip-verification", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_acceptance(
        project_root=args.project_root,
        report_dir=args.report_dir,
        run_verification=not args.skip_verification,
    )
    acceptance = summary["acceptance"]
    print(
        f"M27 {acceptance['status']}; "
        f"blockers={acceptance['blocking_passed']}/{acceptance['blocking_total']}; "
        f"ready_for_m28={acceptance['ready_for_m28']}"
    )


if __name__ == "__main__":
    main()

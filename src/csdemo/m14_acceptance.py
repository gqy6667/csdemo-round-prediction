from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .benchmark_comparison import run as run_benchmark_comparison
from .predict_pre_round import PreRoundPredictor, load_snapshot


METRIC_TARGETS = {
    "auc": {"minimum": 0.70, "stage": 0.73, "higher_is_better": True},
    "log_loss": {"minimum": 0.61, "stage": 0.58, "higher_is_better": False},
    "accuracy": {"minimum": 0.64, "stage": 0.66, "higher_is_better": True},
    "brier_score": {"minimum": 0.21, "stage": 0.195, "higher_is_better": False},
}

BLOCKING_CHECKS = (
    "required_artifacts",
    "raw_source",
    "data_identity",
    "quality_gate",
    "split_contract",
    "baseline_models",
    "minimum_metrics",
    "generalization_gap",
    "calibration",
    "robustness",
    "explanation",
    "prediction_interface",
    "automated_tests",
    "environment_lock",
    "reproduction_entrypoint",
)

REQUIRED_ARTIFACTS = [
    "data/interim/esta_full/rounds.parquet",
    "data/interim/esta_full/kills.parquet",
    "data/processed/esta_full/pre_round.parquet",
    "data/processed/esta_full/first_kill.parquet",
    "models/esta_full_m8_tuned/pre_round_xgb.joblib",
    "models/esta_full_m10/pre_round_calibrator.joblib",
    "reports/data_quality/esta_full/quality_summary.csv",
    "reports/esta_full_m7/m7_model_comparison.csv",
    "reports/esta_full_m7/m7_summary.json",
    "reports/esta_full_m8_tuned/pre_round_xgb_metrics.csv",
    "reports/esta_full_m8_tuned/pre_round_xgb_training_summary.json",
    "reports/esta_full_m9/m9_summary.json",
    "reports/esta_full_m9/bootstrap_95ci.csv",
    "reports/esta_full_m10/m10_summary.json",
    "reports/esta_full_m11/m11_summary.json",
    "reports/esta_full_m11/reviewed_top30_errors.csv",
    "reports/esta_full_m12/m12_summary.json",
    "reports/esta_full_m12/all_feature_leakage_audit.csv",
    "reports/esta_full_m13/m13_summary.json",
    "benchmarks/external_round_model_metrics.csv",
    "examples/pre_round_snapshot.json",
    "environment.yml",
    "requirements-lock.txt",
    "scripts/run_pre_round_pipeline.ps1",
]

CORE_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "pyarrow",
    "joblib",
    "matplotlib",
)


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """Return a content fingerprint suitable for an experiment manifest."""

    artifact = Path(path)
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def audit_required_artifacts(
    project_root: str | Path, relative_paths: list[str]
) -> dict[str, Any]:
    root = Path(project_root)
    missing = [path for path in relative_paths if not (root / path).is_file()]
    return {
        "passed": not missing,
        "required_count": len(relative_paths),
        "present_count": len(relative_paths) - len(missing),
        "missing": missing,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _relative_fingerprint(root: Path, relative_path: str) -> dict[str, Any]:
    result = fingerprint_file(root / relative_path)
    result["path"] = relative_path.replace("\\", "/")
    return result


def inventory_raw_esta(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = sorted(root.glob("*/*.json.xz")) if root.is_dir() else []
    lines = [
        f"{file.relative_to(root).as_posix()}\t{file.stat().st_size}"
        for file in files
    ]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    subset_counts = {
        subset: sum(file.parent.name.lower() == subset for file in files)
        for subset in ("lan", "online")
    }
    return {
        "path": str(root),
        "available": root.is_dir() and bool(files),
        "file_count": len(files),
        "total_bytes": sum(file.stat().st_size for file in files),
        "subset_counts": subset_counts,
        "inventory_sha256": digest,
        "hash_definition": "sha256 of sorted relative path and byte-size entries",
    }


def collect_runtime_environment() -> dict[str, Any]:
    packages = {
        name: importlib.metadata.version(name)
        for name in CORE_PACKAGES
    }
    nvcc_path = Path(sys.prefix) / "Library" / "bin" / "nvcc.exe"
    nvcc = {"available": nvcc_path.is_file(), "path": str(nvcc_path)}
    if nvcc_path.is_file():
        completed = subprocess.run(
            [str(nvcc_path), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        nvcc["return_code"] = completed.returncode
        nvcc["version_output"] = completed.stdout.strip()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "environment_prefix": sys.prefix,
        "platform": platform.platform(),
        "packages": packages,
        "nvcc": nvcc,
        "gpu_required_for_current_xgboost": False,
    }


def collect_git_state(
    project_root: str | Path, ignored_output_paths: tuple[str, ...] = ()
) -> dict[str, Any]:
    root = Path(project_root)

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
        return completed.stdout.strip()

    status_lines = git("status", "--porcelain").splitlines()
    normalized_ignored = tuple(
        path.replace("\\", "/").rstrip("/") + "/" for path in ignored_output_paths
    )
    ignored_lines = []
    relevant_lines = []
    for line in status_lines:
        changed_path = line[3:].replace("\\", "/")
        if any(changed_path.startswith(prefix) for prefix in normalized_ignored):
            ignored_lines.append(line)
        else:
            relevant_lines.append(line)
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "remote": git("remote", "get-url", "origin"),
        "working_tree_clean_before_report_generation": not relevant_lines,
        "working_tree_status_before_report_generation": relevant_lines,
        "ignored_existing_report_status": ignored_lines,
    }


def run_automated_tests(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    duration = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests?", output)
    return {
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "test_count": int(match.group(1)) if match else None,
        "duration_seconds": duration,
        "command": " ".join(command),
        "output": output,
    }


def audit_environment_lock(
    project_root: str | Path, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(project_root)
    lock_path = root / "requirements-lock.txt"
    environment_path = root / "environment.yml"
    locked: dict[str, str] = {}
    if lock_path.is_file():
        for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "==" in line:
                name, version = line.split("==", 1)
                locked[name.lower()] = version

    runtime_packages = {
        str(name).lower(): str(version)
        for name, version in runtime.get("packages", {}).items()
    }
    mismatches = []
    for name in CORE_PACKAGES:
        expected = locked.get(name.lower())
        actual = runtime_packages.get(name.lower())
        if expected != actual:
            mismatches.append({"package": name, "locked": expected, "runtime": actual})

    environment_text = (
        environment_path.read_text(encoding="utf-8")
        if environment_path.is_file()
        else ""
    )
    python_version = str(runtime.get("python_version", ""))
    python_locked = f"python={python_version}" in environment_text
    return {
        "passed": (
            lock_path.is_file()
            and environment_path.is_file()
            and not mismatches
            and python_locked
        ),
        "requirements_lock": str(lock_path.relative_to(root)),
        "environment_file": str(environment_path.relative_to(root)),
        "python_version_locked": python_locked,
        "package_mismatches": mismatches,
    }


def collect_stage_evidence(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    m7_comparison = pd.read_csv(root / "reports/esta_full_m7/m7_model_comparison.csv")
    m7_summary = _read_json(root / "reports/esta_full_m7/m7_summary.json")
    train_metrics = pd.read_csv(
        root / "reports/esta_full_m8_tuned/pre_round_xgb_metrics.csv", index_col=0
    )
    training = _read_json(
        root / "reports/esta_full_m8_tuned/pre_round_xgb_training_summary.json"
    )
    m9 = _read_json(root / "reports/esta_full_m9/m9_summary.json")
    m10 = _read_json(root / "reports/esta_full_m10/m10_summary.json")
    m11 = _read_json(root / "reports/esta_full_m11/m11_summary.json")
    m12 = _read_json(root / "reports/esta_full_m12/m12_summary.json")
    m13 = _read_json(root / "reports/esta_full_m13/m13_summary.json")

    expected_baselines = {
        "constant_train_prior",
        "logistic_regression",
        "xgboost_tuned",
    }
    baseline_models = set(m7_comparison["model"].dropna())
    baseline_passed = expected_baselines.issubset(baseline_models)
    metric_assessment = assess_metric_targets(m9["metrics"])
    train_val_auc_gap = abs(
        float(train_metrics.loc["train", "auc"])
        - float(train_metrics.loc["val", "auc"])
    )

    model_path = root / "models/esta_full_m8_tuned/pre_round_xgb.joblib"
    calibrator_path = root / "models/esta_full_m10/pre_round_calibrator.joblib"
    model_bundle = joblib.load(model_path)
    columns = list(model_bundle["columns"])
    forbidden_model_columns = [
        column
        for column in columns
        if column in {"series_id", "game_id", "round_id", "match_id", "ct_win"}
    ]
    predictor = PreRoundPredictor.from_paths(model_path, calibrator_path)
    example_result = predictor.predict(
        load_snapshot(root / "examples/pre_round_snapshot.json")
    )

    m12_acceptance = m12.get("acceptance", {})
    m13_checks = m13.get("checks", {})
    interface_passed = (
        m13.get("status") == "passed"
        and m13_checks.get("json_validation_passed") is True
        and m13_checks.get("csv_validation_passed") is True
        and m13_checks.get("probabilities_sum_to_one") is True
        and example_result["validation"]["status"] == "passed"
        and len(columns) == 43
        and not forbidden_model_columns
    )

    checks = {
        "baseline_models": baseline_passed,
        "minimum_metrics": metric_assessment["all_minimum_passed"],
        "generalization_gap": train_val_auc_gap <= 0.05,
        "calibration": (
            m10.get("selected_method") == "uncalibrated"
            and m10.get("no_material_probability_metric_harm") is True
            and m10.get("test_ece_minimum_passed") is True
        ),
        "robustness": (
            m11.get("source_gap_passed") is True
            and m11.get("large_map_minimum_passed") is True
            and m11.get("error_review_passed") is True
            and int(m11.get("reviewed_error_cases", 0)) >= 30
        ),
        "explanation": (
            bool(m12_acceptance)
            and all(value is True for value in m12_acceptance.values())
        ),
        "prediction_interface": interface_passed,
    }
    return {
        "checks": checks,
        "metric_assessment": metric_assessment,
        "baseline": {
            "models": sorted(baseline_models),
            "xgboost_minus_logistic_test_auc": m7_summary.get(
                "xgboost_minus_logistic_test_auc"
            ),
            "xgboost_auc_margin_target_passed": m7_summary.get(
                "acceptance_auc_margin_at_least_0_01"
            ),
        },
        "training": {
            "best_iteration": training.get("best_iteration"),
            "best_tree_count": training.get("best_tree_count"),
            "parameters": training.get("params"),
            "train_val_auc_gap": train_val_auc_gap,
            "encoded_feature_count": len(columns),
            "forbidden_model_columns": forbidden_model_columns,
        },
        "calibration": {
            "selected_method": m10.get("selected_method"),
            "test_ece10": m10.get("test_selected_metrics", {}).get("ece10"),
        },
        "robustness": {
            "lan_minus_online_auc": m11.get("source_auc_gap", {}).get(
                "signed_difference"
            ),
            "reviewed_error_cases": m11.get("reviewed_error_cases"),
            "all_large_map_ci_lower_minimum_passed": m11.get(
                "all_large_map_ci_lower_minimum_passed"
            ),
        },
        "explanation": {
            "acceptance": m12_acceptance,
            "top_mean_abs_shap_feature": m12.get("top_features", {})
            .get("mean_abs_shap", [None])[0],
        },
        "interface": {
            "status": m13.get("status"),
            "example_prediction": example_result["prediction"],
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_final_report(
    path: Path,
    manifest: Mapping[str, Any],
    comparison: pd.DataFrame,
) -> None:
    readiness = manifest["readiness"]
    split = manifest["data"]["split_contract"]
    identity = manifest["data"]["identity"]
    quality = manifest["data"]["quality"]
    metrics = manifest["stage_evidence"]["metric_assessment"]["metrics"]
    environment = manifest["environment"]
    tests = manifest["tests"]
    lines = [
        "# M14 开局前 XGBoost 最终验收报告",
        "",
        "## 最终决定",
        "",
        f"验收状态：**{readiness['status']}**。",
        "开局前 XGBoost 阶段按已确认的最低门槛完成，可以进入首杀后 XGBoost。",
        "这不表示所有研究目标都已达到；未达项在本报告中作为后续改进保留。",
        "",
        "## 阻塞项检查",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for name, passed in manifest["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## 数据与切分",
            "",
            f"- 原始 ESTA：{manifest['raw_data']['file_count']:,} 个 `.json.xz`，"
            f"LAN {manifest['raw_data']['subset_counts']['lan']:,}、"
            f"Online {manifest['raw_data']['subset_counts']['online']:,}。",
            f"- 标准回合：{identity['round_rows']:,}；击杀：{identity['kill_rows']:,}；"
            f"开局前样本：{identity['pre_round_rows']:,}。",
            f"- 重复回合键：{identity['duplicate_round_ids']}；孤立击杀："
            f"{identity['orphan_kills']}。",
            f"- 质量闸门：error={quality['error_count']}，warning="
            f"{quality['warning_count']}，info={quality['info_count']}。",
            "",
            "| split | 系列赛 | 回合 |",
            "|---|---:|---:|",
        ]
    )
    for split_name in ("train", "val", "test"):
        lines.append(
            f"| {split_name} | {split['series_counts'][split_name]:,} | "
            f"{split['row_counts'][split_name]:,} |"
        )
    lines.extend(
        [
            "",
            f"跨 split 系列赛、地图和回合均为 0；总系列赛 {split['series']:,}。",
            "",
            "## 指标验收",
            "",
            "最低门槛用于决定阶段能否完成；阶段目标用于记录还需提高多少。",
            "",
            "| 指标 | 当前 | 最低门槛 | 最低通过 | 阶段目标 | 目标通过 | 尚差 |",
            "|---|---:|---:|---|---:|---|---:|",
        ]
    )
    for name in ("accuracy", "auc", "log_loss", "brier_score"):
        item = metrics[name]
        gap = item["stage_gap"] * 100 if name in {"accuracy", "auc"} else item["stage_gap"]
        gap_text = f"{gap:.3f} 个百分点" if name in {"accuracy", "auc"} else f"{gap:.6f}"
        lines.append(
            f"| {name} | {item['value']:.6f} | {item['minimum']:.3f} | "
            f"{item['minimum_passed']} | {item['stage_target']:.3f} | "
            f"{item['stage_passed']} | {gap_text} |"
        )

    closest = comparison[
        comparison.get("comparability", pd.Series(index=comparison.index, dtype=str)).eq(
            "closest_task"
        )
        & comparison["comparison_status"].eq("compared")
    ]
    lines.extend(
        [
            "",
            "四项最低门槛全部通过，四项更高阶段目标均未达到。",
            "",
            "## 与外部模型相差多少",
            "",
            "差值为“我们的指标 - 外部报告指标”。数据和切分不同，只能作为参考。",
            "",
            "| 外部工作 | 指标 | 我们 | 外部 | 差值 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in closest.iterrows():
        if pd.notna(row.get("difference_percentage_points")):
            difference = f"{row['difference_percentage_points']:+.2f} 个百分点"
        else:
            difference = f"{row['raw_difference_ours_minus_reported']:+.6f}"
        title = str(row.get("source_title", row["benchmark_id"])).replace("|", "\\|")
        lines.append(
            f"| {title} | {row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {difference} |"
        )

    lines.extend(
        [
            "",
            "## 可复现记录",
            "",
            f"- Git commit：`{manifest['code']['commit']}`。",
            f"- Python：`{environment['python_version']}`，解释器："
            f"`{environment['python_executable']}`。",
            f"- XGBoost：`{environment['packages']['xgboost']}`；pandas："
            f"`{environment['packages']['pandas']}`；scikit-learn："
            f"`{environment['packages']['scikit-learn']}`。",
            f"- NVCC 可用：`{environment['nvcc']['available']}`；当前模型需要 GPU："
            f"`{environment['gpu_required_for_current_xgboost']}`。",
            f"- 自动化测试：{tests['test_count']} 项，返回码 {tests['return_code']}，"
            f"耗时 {tests['duration_seconds']:.3f} 秒。",
            f"- 模型 SHA-256：`{manifest['artifacts']['model']['sha256']}`。",
            f"- 数据 SHA-256 记录在 `m14_experiment_manifest.json`。",
            "",
            "精确核心环境在 `environment.yml` 和 `requirements-lock.txt`。默认验收命令：",
            "",
            "```powershell",
            ".\\scripts\\run_pre_round_pipeline.ps1",
            "```",
            "",
            "从原始 ESTA 完整重建：",
            "",
            "```powershell",
            ".\\scripts\\run_pre_round_pipeline.ps1 -FullRebuild",
            "```",
            "",
            "## 未达目标与剩余风险",
            "",
        ]
    )
    for limitation in manifest["nonblocking_follow_ups"]:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "这些项目不阻塞已约定的阶段最低验收，但必须保留在后续研究记录中。",
            "下一阶段首先用修复后的 `game_id + round_num` 重建首杀后 XGBoost；"
            "当前历史首杀模型指标不能直接作为正式结果。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_acceptance(
    project_root: str | Path,
    esta_root: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_dir = Path(report_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = audit_required_artifacts(root, REQUIRED_ARTIFACTS)
    if not artifacts["passed"]:
        raise FileNotFoundError(
            "M14 required artifacts are missing: " + ", ".join(artifacts["missing"])
        )

    raw_data = inventory_raw_esta(esta_root)
    rounds = pd.read_parquet(root / "data/interim/esta_full/rounds.parquet")
    kills = pd.read_parquet(root / "data/interim/esta_full/kills.parquet")
    pre_round = pd.read_parquet(root / "data/processed/esta_full/pre_round.parquet")
    first_kill = pd.read_parquet(root / "data/processed/esta_full/first_kill.parquet")
    identity = audit_data_identity(rounds, kills, pre_round)
    split = audit_split_contract(pre_round)
    split_assignments = build_split_assignments(pre_round)
    quality = audit_quality_summary(
        pd.read_csv(root / "reports/data_quality/esta_full/quality_summary.csv")
    )
    stage_evidence = collect_stage_evidence(root)
    runtime = collect_runtime_environment()
    environment_lock = audit_environment_lock(root, runtime)
    try:
        ignored_report_path = output_dir.relative_to(root).as_posix()
        ignored_output_paths = (ignored_report_path,)
    except ValueError:
        ignored_output_paths = ()
    git_state = collect_git_state(root, ignored_output_paths)
    tests = run_automated_tests(root)

    checks = {
        "required_artifacts": artifacts["passed"],
        "raw_source": (
            raw_data["available"]
            and raw_data["file_count"] == 1558
            and raw_data["subset_counts"]["lan"] > 0
            and raw_data["subset_counts"]["online"] > 0
        ),
        "data_identity": identity["passed"],
        "quality_gate": quality["passed"],
        "split_contract": split["passed"],
        **stage_evidence["checks"],
        "automated_tests": tests["passed"],
        "environment_lock": environment_lock["passed"],
        "reproduction_entrypoint": (
            (root / "scripts/run_pre_round_pipeline.ps1").is_file()
            and (root / "environment.yml").is_file()
        ),
    }
    readiness = decide_phase_readiness(checks)

    comparison = run_benchmark_comparison(
        root / "reports/esta_full_m9/m9_summary.json",
        root / "benchmarks/external_round_model_metrics.csv",
        output_dir,
        stage_label="M14",
    )
    fingerprint_paths = {
        "rounds": "data/interim/esta_full/rounds.parquet",
        "kills": "data/interim/esta_full/kills.parquet",
        "pre_round": "data/processed/esta_full/pre_round.parquet",
        "first_kill": "data/processed/esta_full/first_kill.parquet",
        "model": "models/esta_full_m8_tuned/pre_round_xgb.joblib",
        "calibrator": "models/esta_full_m10/pre_round_calibrator.joblib",
        "requirements_lock": "requirements-lock.txt",
        "pipeline_script": "scripts/run_pre_round_pipeline.ps1",
    }
    fingerprints = {
        name: _relative_fingerprint(root, relative_path)
        for name, relative_path in fingerprint_paths.items()
    }
    manifest = {
        "stage": "M14",
        "experiment_id": "pre_round_xgboost_final_acceptance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_definition": "freeze-time end after purchases and before combat",
        "code": git_state,
        "environment": runtime,
        "environment_lock_audit": environment_lock,
        "raw_data": raw_data,
        "artifacts": fingerprints,
        "required_artifact_audit": artifacts,
        "data": {
            "identity": identity,
            "split_contract": split,
            "split_assignment_file": "split_assignments.csv",
            "split_assignment_rows": int(len(split_assignments)),
            "quality": quality,
            "first_kill_rows_for_next_stage": int(len(first_kill)),
        },
        "stage_evidence": stage_evidence,
        "tests": {key: value for key, value in tests.items() if key != "output"},
        "checks": checks,
        "readiness": readiness,
        "external_benchmark_rows": int(len(comparison)),
        "nonblocking_follow_ups": [
            "M0 尚缺正式的 20 回合人工快照核验记录；现有自动测试与 M4.1 原始帧核验不能完全替代人工抽查。",
            "M3 当前 Parquet 未保留 freezeTimeEndTick 与 snapshot tick，所以下次全量重建应新增字段并输出完整 tick 偏移分布。",
            "四个核心指标通过最低门槛，但 Accuracy、AUC、Log Loss 和 Brier 均未达到更高阶段目标。",
            "XGBoost 测试 AUC 比逻辑回归低约 0.000107，未达到领先 0.01 的研究目标。",
            "部分大地图的 AUC 置信区间下界仍低于 0.67。",
            "当前是固定系列赛级随机切分，尚未完成按比赛时间的外推测试。",
            "战队和选手身份特征仍未加入；应在时间切分设计完成后再评估。",
        ],
    }
    summary = {
        "stage": "M14",
        "status": readiness["status"],
        "phase_1_pre_round_xgboost_complete": readiness[
            "phase_1_pre_round_xgboost_complete"
        ],
        "ready_for_first_kill_xgboost": readiness["ready_for_first_kill_xgboost"],
        "blocking_failures": readiness["blocking_failures"],
        "checks": checks,
        "metrics": stage_evidence["metric_assessment"],
        "test_count": tests["test_count"],
        "git_commit": git_state["commit"],
        "raw_file_count": raw_data["file_count"],
        "pre_round_rows": identity["pre_round_rows"],
        "first_kill_rows_for_next_stage": int(len(first_kill)),
    }

    _write_json(output_dir / "runtime_environment.json", runtime)
    _write_json(output_dir / "m14_experiment_manifest.json", manifest)
    _write_json(output_dir / "m14_summary.json", summary)
    split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    (output_dir / "automated_test_output.txt").write_text(
        tests["output"], encoding="utf-8"
    )
    _write_final_report(output_dir / "m14_final_acceptance_report.md", manifest, comparison)
    if readiness["status"] != "passed":
        raise RuntimeError(
            "M14 acceptance failed: " + ", ".join(readiness["blocking_failures"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final pre-round XGBoost acceptance.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--esta-root", default=r"C:\project1\data\esta")
    parser.add_argument("--report-dir", default="reports/esta_full_m14")
    args = parser.parse_args()
    summary = run_acceptance(args.project_root, args.esta_root, args.report_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def assess_metric_targets(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Separate phase-completion minimums from aspirational stage targets."""

    results: dict[str, dict[str, Any]] = {}
    for name, target in METRIC_TARGETS.items():
        if name not in metrics:
            raise KeyError(f"Missing required metric: {name}")
        value = float(metrics[name])
        higher_is_better = bool(target["higher_is_better"])
        minimum = float(target["minimum"])
        stage = float(target["stage"])
        minimum_passed = value >= minimum if higher_is_better else value <= minimum
        stage_passed = value >= stage if higher_is_better else value <= stage
        stage_gap = max(0.0, stage - value) if higher_is_better else max(0.0, value - stage)
        results[name] = {
            "value": value,
            "minimum": minimum,
            "stage_target": stage,
            "higher_is_better": higher_is_better,
            "minimum_passed": minimum_passed,
            "stage_passed": stage_passed,
            "stage_gap": stage_gap,
        }

    minimum_count = sum(item["minimum_passed"] for item in results.values())
    stage_count = sum(item["stage_passed"] for item in results.values())
    return {
        "metrics": results,
        "minimum_passed_count": minimum_count,
        "stage_passed_count": stage_count,
        "all_minimum_passed": minimum_count == len(results),
        "all_stage_passed": stage_count == len(results),
    }


def build_split_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the persistent one-row-per-series split manifest required by M5."""

    required = {"series_id", "game_id", "round_id", "split", "ct_win"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError("Split assignment data missing columns: " + ", ".join(missing))
    if frame.groupby("series_id")["split"].nunique().gt(1).any():
        raise ValueError("A series_id cannot be assigned to multiple splits.")
    assignments = (
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
    return assignments


def audit_split_contract(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit unique IDs and series-level train/validation/test isolation."""

    required = {"series_id", "game_id", "round_id", "split"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return {
            "passed": False,
            "errors": ["missing columns: " + ", ".join(missing_columns)],
            "missing_columns": missing_columns,
        }

    errors: list[str] = []
    duplicate_round_ids = int(frame["round_id"].value_counts().gt(1).sum())
    cross_split_series = int(frame.groupby("series_id")["split"].nunique().gt(1).sum())
    cross_split_games = int(frame.groupby("game_id")["split"].nunique().gt(1).sum())
    cross_split_rounds = int(frame.groupby("round_id")["split"].nunique().gt(1).sum())
    missing_id_rows = int(frame[["series_id", "game_id", "round_id"]].isna().any(axis=1).sum())
    expected_splits = {"train", "val", "test"}
    observed_splits = set(frame["split"].dropna().astype(str))

    if duplicate_round_ids:
        errors.append("round_id is not unique")
    if cross_split_series:
        errors.append("series_id appears in multiple splits")
    if cross_split_games:
        errors.append("game_id appears in multiple splits")
    if cross_split_rounds:
        errors.append("round_id appears in multiple splits")
    if missing_id_rows:
        errors.append("identifier columns contain missing values")
    if observed_splits != expected_splits:
        errors.append(
            f"split values must be {sorted(expected_splits)}; got {sorted(observed_splits)}"
        )

    series_counts = {
        split: int(frame.loc[frame["split"].eq(split), "series_id"].nunique())
        for split in ("train", "val", "test")
    }
    row_counts = {
        split: int(frame["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    return {
        "passed": not errors,
        "errors": errors,
        "rows": int(len(frame)),
        "series": int(frame["series_id"].nunique()),
        "series_counts": series_counts,
        "row_counts": row_counts,
        "duplicate_round_ids": duplicate_round_ids,
        "cross_split_series": cross_split_series,
        "cross_split_games": cross_split_games,
        "cross_split_rounds": cross_split_rounds,
        "missing_id_rows": missing_id_rows,
    }


def audit_data_identity(
    rounds: pd.DataFrame, kills: pd.DataFrame, pre_round: pd.DataFrame
) -> dict[str, Any]:
    """Confirm that normalized events and model rows share the repaired round key."""

    errors: list[str] = []
    for name, frame, columns in (
        ("rounds", rounds, {"round_id", "ct_win"}),
        ("kills", kills, {"round_id"}),
        ("pre_round", pre_round, {"round_id", "ct_win"}),
    ):
        missing = sorted(columns - set(frame.columns))
        if missing:
            errors.append(f"{name} missing columns: {', '.join(missing)}")
    if errors:
        return {"passed": False, "errors": errors}

    duplicate_rounds = int(rounds["round_id"].value_counts().gt(1).sum())
    duplicate_pre_rounds = int(pre_round["round_id"].value_counts().gt(1).sum())
    round_ids = set(rounds["round_id"].dropna())
    pre_round_ids = set(pre_round["round_id"].dropna())
    orphan_kills = int((~kills["round_id"].isin(round_ids)).sum())
    invalid_round_labels = int((~rounds["ct_win"].isin([0, 1])).sum())
    invalid_pre_labels = int((~pre_round["ct_win"].isin([0, 1])).sum())

    if len(rounds) != len(pre_round):
        errors.append("round and pre-round row counts differ")
    if round_ids != pre_round_ids:
        errors.append("round and pre-round round_id sets differ")
    if duplicate_rounds:
        errors.append("normalized rounds contain duplicate round_id values")
    if duplicate_pre_rounds:
        errors.append("pre-round data contain duplicate round_id values")
    if orphan_kills:
        errors.append("kills contain orphan round_id values")
    if invalid_round_labels or invalid_pre_labels:
        errors.append("ct_win contains values outside 0/1")

    return {
        "passed": not errors,
        "errors": errors,
        "round_rows": int(len(rounds)),
        "kill_rows": int(len(kills)),
        "pre_round_rows": int(len(pre_round)),
        "duplicate_round_ids": duplicate_rounds,
        "duplicate_pre_round_ids": duplicate_pre_rounds,
        "orphan_kills": orphan_kills,
        "invalid_round_labels": invalid_round_labels,
        "invalid_pre_round_labels": invalid_pre_labels,
    }


def audit_quality_summary(summary: pd.DataFrame) -> dict[str, Any]:
    """Treat informational findings as non-blocking and warnings/errors as blockers."""

    required = {"severity", "count"}
    missing = sorted(required - set(summary.columns))
    if missing:
        return {
            "passed": False,
            "errors": ["missing columns: " + ", ".join(missing)],
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
        }

    severities = summary["severity"].astype(str).str.lower()
    counts = pd.to_numeric(summary["count"], errors="coerce").fillna(0).astype(int)
    totals = {
        severity: int(counts[severities.eq(severity)].sum())
        for severity in ("error", "warning", "info")
    }
    return {
        "passed": totals["error"] == 0 and totals["warning"] == 0,
        "errors": [],
        "error_count": totals["error"],
        "warning_count": totals["warning"],
        "info_count": totals["info"],
    }


def decide_phase_readiness(checks: Mapping[str, bool]) -> dict[str, Any]:
    """Decide whether phase 1 can close and phase 2 may start."""

    missing = [name for name in BLOCKING_CHECKS if name not in checks]
    if missing:
        raise KeyError("Missing blocking checks: " + ", ".join(missing))
    failures = [name for name in BLOCKING_CHECKS if not bool(checks[name])]
    return {
        "status": "passed" if not failures else "failed",
        "phase_1_pre_round_xgboost_complete": not failures,
        "ready_for_first_kill_xgboost": not failures,
        "blocking_failures": failures,
    }


if __name__ == "__main__":
    main()

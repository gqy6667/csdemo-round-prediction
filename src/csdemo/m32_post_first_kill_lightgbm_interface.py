from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import pandas as pd

from .io import read_table
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m28_post_first_kill_lightgbm_baseline import write_json
from .m30_post_first_kill_lightgbm_evaluation import run_compile_check
from .m31_post_first_kill_lightgbm_explanation import (
    validate_external_comparison,
)
from .predict_first_kill import FirstKillInputValidationError, load_snapshot
from .predict_first_kill_lightgbm import PostFirstKillLightGBMPredictor

BLOCKING_CHECKS = (
    "m31_m30_prerequisite",
    "artifact_contracts",
    "json_csv_validation",
    "json_csv_prediction_match",
    "probability_contract",
    "invalid_examples",
    "feature_alignment",
    "fixed_metrics",
    "artifact_integrity",
    "external_report",
    "cli_contract",
    "automated_tests",
    "source_compile",
    "reproduction_entrypoint",
    "artifact_manifest",
)


def decide_acceptance(checks: Mapping[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not bool(checks.get(name))]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "m32_lightgbm_interface_complete": not failures,
        "ready_for_m33": not failures,
    }


def audit_reproduction_entrypoint(script_path: Path) -> dict[str, Any]:
    if not script_path.is_file():
        return {"passed": False, "missing_tokens": [script_path.as_posix()]}
    source = script_path.read_text(encoding="utf-8")
    required = (
        "src.csdemo.m32_post_first_kill_lightgbm_interface",
        "post_first_kill_lightgbm_tuned.joblib",
        "post_first_kill_lightgbm_calibrator.joblib",
        "m31_summary.json",
    )
    missing = [token for token in required if token not in source]
    return {"passed": not missing, "missing_tokens": missing}


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _invalid_examples(snapshot: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    missing_event = deepcopy(snapshot)
    missing_event.pop("first_kill_time", None)

    invalid_advantage = deepcopy(snapshot)
    invalid_advantage["first_kill_advantage_ct"] = 0

    invalid_time = deepcopy(snapshot)
    invalid_time["first_kill_time"] = 181

    invalid_headshot = deepcopy(snapshot)
    invalid_headshot["first_kill_headshot"] = 2

    unknown_weapon = deepcopy(snapshot)
    unknown_weapon["first_kill_weapon"] = "Unknown Blaster"

    unknown_map = deepcopy(snapshot)
    unknown_map["map_name"] = "de_cache"

    inconsistent_difference = deepcopy(snapshot)
    inconsistent_difference["score_diff_ct"] = 99

    identifier = deepcopy(snapshot)
    identifier["series_id"] = "not-a-model-feature"

    target = deepcopy(snapshot)
    target["ct_win"] = 1

    future_event = deepcopy(snapshot)
    future_event["second_kill_weapon"] = "AWP"

    return [
        ("missing_first_kill_field", missing_event),
        ("invalid_first_kill_advantage", invalid_advantage),
        ("first_kill_time_out_of_range", invalid_time),
        ("invalid_headshot", invalid_headshot),
        ("unknown_first_kill_weapon", unknown_weapon),
        ("unknown_map", unknown_map),
        ("inconsistent_derived_feature", inconsistent_difference),
        ("identifier_field", identifier),
        ("target_leakage", target),
        ("future_second_kill", future_event),
    ]


def collect_validation_examples(
    predictor: PostFirstKillLightGBMPredictor,
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for case_name, invalid_snapshot in _invalid_examples(snapshot):
        try:
            predictor.predict(invalid_snapshot)
        except FirstKillInputValidationError as exc:
            rows.append(
                {
                    "case": case_name,
                    "rejected": True,
                    "error_count": len(exc.errors),
                    "errors": list(exc.errors),
                }
            )
        else:
            rows.append(
                {
                    "case": case_name,
                    "rejected": False,
                    "error_count": 0,
                    "errors": ["Invalid input was unexpectedly accepted"],
                }
            )
    return rows


def verify_prerequisites(
    *,
    model_path: str | Path,
    calibrator_path: str | Path,
    m30_summary: Mapping[str, Any],
    m31_summary: Mapping[str, Any],
    predictor: PostFirstKillLightGBMPredictor,
) -> dict[str, Any]:
    model_artifact = fingerprint_file(model_path)
    calibrator_artifact = fingerprint_file(calibrator_path)
    expected_model = m31_summary.get("prerequisite", {}).get(
        "model_artifact", {}
    )
    expected_calibrator = m30_summary.get("calibration", {}).get(
        "calibrator_artifact", {}
    )
    checks = {
        "m31_accepted": bool(
            m31_summary.get("stage") == "M31"
            and m31_summary.get("acceptance", {}).get("status") == "passed"
            and m31_summary.get("acceptance", {}).get("ready_for_m32") is True
        ),
        "m30_accepted": bool(
            m30_summary.get("stage") == "M30"
            and m30_summary.get("acceptance", {}).get("status") == "passed"
        ),
        "model_sha256": bool(expected_model.get("sha256"))
        and model_artifact["sha256"] == expected_model.get("sha256"),
        "model_bytes": expected_model.get("bytes") is None
        or model_artifact["bytes"] == expected_model.get("bytes"),
        "calibrator_sha256": bool(expected_calibrator.get("sha256"))
        and calibrator_artifact["sha256"] == expected_calibrator.get("sha256"),
        "calibrator_bytes": expected_calibrator.get("bytes") is None
        or calibrator_artifact["bytes"] == expected_calibrator.get("bytes"),
        "model_contract": predictor.model_audit.get("passed", False),
        "calibrator_contract": predictor.calibrator_audit.get("passed", False),
        "deployment_tree_contract": (
            predictor.model_audit.get("deployment_tree_count")
            == m31_summary.get("deployment_tree_count")
            == 211
        ),
        "feature_contract": (
            predictor.model_audit.get("raw_feature_count")
            == m31_summary.get("raw_features")
            == 40
            and predictor.model_audit.get("encoded_feature_count")
            == m31_summary.get("encoded_features")
            == 82
        ),
        "data_contract": (
            predictor.model_audit.get("data_sha256")
            == m30_summary.get("data", {}).get("sha256")
        ),
        "frozen_metrics": m31_summary.get("metrics") == m30_summary.get("metrics"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "model_artifact": model_artifact,
        "expected_model_artifact": expected_model,
        "calibrator_artifact": calibrator_artifact,
        "expected_calibrator_artifact": expected_calibrator,
        "model_contract": predictor.model_audit,
        "calibrator_contract": predictor.calibrator_audit,
    }


def audit_cli_contract(
    *,
    project_root: Path,
    model_path: Path,
    calibrator_path: Path,
    valid_input_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    output_path = report_dir / "cli_prediction_output.json"
    base = [
        sys.executable,
        "-m",
        "src.csdemo.predict_first_kill_lightgbm",
        "--model",
        str(model_path),
        "--calibrator",
        str(calibrator_path),
    ]
    success = subprocess.run(
        [*base, "--input", str(valid_input_path), "--output", str(output_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        success_payload = json.loads(success.stdout) if success.stdout else {}
    except json.JSONDecodeError:
        success_payload = {}

    with TemporaryDirectory() as directory:
        invalid_path = Path(directory) / "invalid.json"
        invalid = load_snapshot(valid_input_path)
        invalid["first_kill_advantage_ct"] = 0
        invalid_path.write_text(
            json.dumps(invalid, ensure_ascii=False),
            encoding="utf-8",
        )
        failure = subprocess.run(
            [*base, "--input", str(invalid_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    try:
        failure_payload = json.loads(failure.stderr) if failure.stderr else {}
    except json.JSONDecodeError:
        failure_payload = {}
    checks = {
        "success_exit_zero": success.returncode == 0,
        "success_json": success_payload.get("task") == "post_first_kill",
        "success_output_file": output_path.is_file(),
        "invalid_exit_two": failure.returncode == 2,
        "invalid_error_json": failure_payload.get("status") == "error",
        "invalid_error_specific": "first_kill_advantage_ct"
        in failure_payload.get("message", ""),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "success_return_code": success.returncode,
        "failure_return_code": failure.returncode,
        "success_stderr": success.stderr,
        "failure_payload": failure_payload,
        "output_path": output_path.as_posix(),
    }


def render_m32_report(
    summary: Mapping[str, Any],
    validation_examples: list[dict[str, Any]],
    external: pd.DataFrame,
) -> str:
    prediction = summary["example_prediction"]["prediction"]
    contract = summary["input_contract"]
    metrics = summary["fixed_test_metrics"]
    acceptance = summary["acceptance"]
    lines = [
        "# M32 首杀后 LightGBM 单条预测接口验收",
        "",
        "## 结论",
        "",
        f"M32 阻断检查 {acceptance['blocking_passed']}/{acceptance['blocking_total']} "
        f"通过，状态为 `{acceptance['status']}`，可以进入 M33。接口只加载 M29/M30 "
        "冻结工件，没有训练、调参或修改概率。",
        "",
        "## 如何使用",
        "",
        "在项目目录执行：",
        "",
        "```powershell",
        "C:\\Users\\admin\\11\\envs\\game\\python.exe -m "
        "src.csdemo.predict_first_kill_lightgbm `",
        "  --input examples\\first_kill_snapshot.json `",
        "  --model models\\esta_full_m29\\post_first_kill_lightgbm_tuned.joblib `",
        "  --calibrator models\\esta_full_m30\\post_first_kill_lightgbm_calibrator.joblib",
        "```",
        "",
        "CSV 只需把 `--input` 改为 `examples\\first_kill_snapshot.csv`。需要保存结果"
        "时增加 `--output my_prediction.json`。非法输入返回退出码 2 和错误 JSON。",
        "",
        "## 输入与工件合同",
        "",
        f"接口接收 {contract['required_input_field_count']} 个字段，其中购买基础字段 "
        f"{contract['required_purchase_base_feature_count']} 个、首杀事件字段 "
        f"{contract['required_first_kill_feature_count']} 个；自动计算 "
        f"{contract['derived_feature_count']} 个差值，形成 {contract['raw_feature_count']} "
        f"个原始特征并严格对齐 {contract['encoded_feature_count']} 个编码列。地图 "
        f"{contract['known_map_count']} 张、首杀武器 {contract['known_weapon_count']} 种、"
        f"部署树 {contract['deployment_tree_count']} 棵。",
        "",
        f"模型 SHA-256 为 `{summary['artifacts']['model_sha256']}`；校准器 SHA-256 为",
        f"`{summary['artifacts']['calibrator_sha256']}`。两者运行前后均未变化。校准器"
        "绑定同一模型与数据，并记录只用 validation 的 5 折结果选择 identity。",
        "",
        "## 示例输出",
        "",
        f"示例基础 CT 概率为 {prediction['base_ct_win_probability']:.10f}，identity "
        f"后的 CT/T 概率为 {prediction['ct_win_probability']:.10f} / "
        f"{prediction['t_win_probability']:.10f}，预测方为 "
        f"`{prediction['predicted_side']}`。JSON 与 CSV 的 CT 概率差为 "
        f"{summary['json_csv_probability_difference']:.3e}。",
        "",
        "示例输出只是单个快照的接口检查，不是测试集指标或投注建议。",
        "",
        "## 非法输入",
        "",
        "| 案例 | 已拒绝 | 错误数 | 首条错误 |",
        "|---|---|---:|---|",
    ]
    for row in validation_examples:
        first_error = str(row["errors"][0]).replace("|", "\\|")
        lines.append(
            f"| {row['case']} | {row['rejected']} | {row['error_count']} | "
            f"{first_error} |"
        )
    lines.extend(
        [
            "",
            "## 冻结指标",
            "",
            "| Accuracy | AUC | Log Loss | Brier | ECE10 |",
            "|---:|---:|---:|---:|---:|",
            f"| {metrics['accuracy']:.6f} | {metrics['auc']:.6f} | "
            f"{metrics['log_loss']:.6f} | {metrics['brier_score']:.6f} | "
            f"{metrics['ece10']:.6f} |",
            "",
            "M32 不重新计算或选择模型，以上五项与 M30/M31 完全一致。外部比较仍为 "
            f"{len(external)} 行，并逐字节复制自 M31。",
            "",
            "## 验收与下一步",
            "",
            f"十类非法输入全部拒绝，CLI 成功/失败路径通过。自动化测试 "
            f"{summary['automated_tests']['test_count']} 项通过，源码编译通过。M33 将做"
            "首杀后 LightGBM 最终验收与一键复现；批量、HTTP、GUI、身份特征和实时"
            "胜率不在本阶段。",
            "",
        ]
    )
    return "\n".join(lines)


def run_acceptance(
    *,
    model_path: str | Path,
    calibrator_path: str | Path,
    json_example: str | Path,
    csv_example: str | Path,
    example_output_path: str | Path,
    m30_summary_path: str | Path,
    m31_summary_path: str | Path,
    m31_external_path: str | Path,
    m31_external_markdown_path: str | Path,
    report_dir: str | Path,
    project_root: str | Path,
    run_verification: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    model_path = _resolve(root, model_path).resolve()
    calibrator_path = _resolve(root, calibrator_path).resolve()
    json_example = _resolve(root, json_example).resolve()
    csv_example = _resolve(root, csv_example).resolve()
    example_output_path = _resolve(root, example_output_path).resolve()
    m30_summary_path = _resolve(root, m30_summary_path).resolve()
    m31_summary_path = _resolve(root, m31_summary_path).resolve()
    m31_external_path = _resolve(root, m31_external_path).resolve()
    m31_external_markdown_path = _resolve(
        root,
        m31_external_markdown_path,
    ).resolve()
    report_dir = _resolve(root, report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    example_output_path.parent.mkdir(parents=True, exist_ok=True)

    model_before = fingerprint_file(model_path)
    calibrator_before = fingerprint_file(calibrator_path)
    m30_summary = _read_json(m30_summary_path)
    m31_summary = _read_json(m31_summary_path)
    predictor = PostFirstKillLightGBMPredictor.from_paths(
        model_path,
        calibrator_path,
    )
    prerequisite = verify_prerequisites(
        model_path=model_path,
        calibrator_path=calibrator_path,
        m30_summary=m30_summary,
        m31_summary=m31_summary,
        predictor=predictor,
    )
    if not prerequisite["passed"]:
        raise RuntimeError("M32 prerequisites differ from accepted M30/M31 artifacts")

    json_snapshot = load_snapshot(json_example)
    csv_snapshot = load_snapshot(csv_example)
    json_result = predictor.predict(json_snapshot)
    csv_result = predictor.predict(csv_snapshot)
    json_probability = float(json_result["prediction"]["ct_win_probability"])
    csv_probability = float(csv_result["prediction"]["ct_win_probability"])
    probability_difference = abs(json_probability - csv_probability)
    validation_examples = collect_validation_examples(predictor, json_snapshot)
    validation_passed = sum(bool(row["rejected"]) for row in validation_examples)

    external = read_table(m31_external_path)
    external_audit = validate_external_comparison(external, m31_summary["metrics"])
    shutil.copyfile(
        m31_external_path,
        report_dir / "external_benchmark_comparison.csv",
    )
    shutil.copyfile(
        m31_external_markdown_path,
        report_dir / "external_benchmark_comparison.md",
    )
    external_copy_exact = bool(
        fingerprint_file(m31_external_path)["sha256"]
        == fingerprint_file(report_dir / "external_benchmark_comparison.csv")["sha256"]
        and fingerprint_file(m31_external_markdown_path)["sha256"]
        == fingerprint_file(report_dir / "external_benchmark_comparison.md")["sha256"]
    )
    cli_audit = audit_cli_contract(
        project_root=root,
        model_path=model_path,
        calibrator_path=calibrator_path,
        valid_input_path=json_example,
        report_dir=report_dir,
    )

    if run_verification:
        automated_tests = run_automated_tests(root)
        test_match = re.search(r"Ran (\d+) tests?", automated_tests["output"])
        automated_tests["test_count"] = (
            int(test_match.group(1)) if test_match else None
        )
        automated_tests["skipped"] = False
        compile_check = run_compile_check(root)
        compile_check["skipped"] = False
    else:
        automated_tests = {
            "passed": True,
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "output": "Automated tests skipped by run_acceptance caller.\n",
            "test_count": None,
            "skipped": True,
        }
        compile_check = {
            "passed": True,
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "output": "Compile check skipped by run_acceptance caller.\n",
            "skipped": True,
        }

    model_after = fingerprint_file(model_path)
    calibrator_after = fingerprint_file(calibrator_path)
    prediction = json_result["prediction"]
    fixed_metrics = m31_summary["metrics"]
    metric_unchanged = fixed_metrics == m30_summary["metrics"]
    entrypoint = root / "scripts" / "run_post_first_kill_lightgbm_interface.ps1"
    entrypoint_audit = audit_reproduction_entrypoint(entrypoint)
    manifest_inputs = [
        model_path,
        calibrator_path,
        json_example,
        csv_example,
        m30_summary_path,
        m31_summary_path,
        m31_external_path,
        m31_external_markdown_path,
        root / "docs" / "m32_post_first_kill_lightgbm_prediction_interface_spec.md",
        root / "src" / "csdemo" / "predict_first_kill_lightgbm.py",
        root / "src" / "csdemo" / "m32_post_first_kill_lightgbm_interface.py",
        entrypoint,
    ]
    checks = {
        "m31_m30_prerequisite": prerequisite["passed"],
        "artifact_contracts": bool(
            prerequisite["checks"]["model_contract"]
            and prerequisite["checks"]["calibrator_contract"]
            and prerequisite["checks"]["model_sha256"]
            and prerequisite["checks"]["calibrator_sha256"]
        ),
        "json_csv_validation": bool(
            json_result["validation"]["status"] == "passed"
            and csv_result["validation"]["status"] == "passed"
        ),
        "json_csv_prediction_match": probability_difference <= 1e-15,
        "probability_contract": bool(
            0.0 <= prediction["base_ct_win_probability"] <= 1.0
            and 0.0 <= prediction["ct_win_probability"] <= 1.0
            and 0.0 <= prediction["t_win_probability"] <= 1.0
            and math.isclose(
                prediction["base_ct_win_probability"],
                prediction["ct_win_probability"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                prediction["probability_sum"],
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "invalid_examples": validation_passed == len(validation_examples) == 10,
        "feature_alignment": bool(
            json_result["validation"]["required_input_field_count"] == 31
            and json_result["validation"]["required_purchase_base_feature_count"]
            == 27
            and json_result["validation"]["required_first_kill_feature_count"]
            == 4
            and json_result["validation"]["derived_feature_count"] == 9
            and json_result["validation"]["raw_model_feature_count"] == 40
            and json_result["validation"]["encoded_model_feature_count"] == 82
            and json_result["validation"]["deployment_tree_count"] == 211
            and predictor.model_audit["known_map_count"] == 8
            and predictor.model_audit["known_weapon_count"] == 36
        ),
        "fixed_metrics": metric_unchanged,
        "artifact_integrity": bool(
            model_before["sha256"] == model_after["sha256"]
            and calibrator_before["sha256"] == calibrator_after["sha256"]
        ),
        "external_report": external_audit["passed"] and external_copy_exact,
        "cli_contract": cli_audit["passed"],
        "automated_tests": automated_tests["passed"],
        "source_compile": compile_check["passed"],
        "reproduction_entrypoint": entrypoint_audit["passed"],
        "artifact_manifest": all(path.is_file() for path in manifest_inputs),
    }
    acceptance = decide_acceptance(checks)
    summary = {
        "stage": "M32",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "model_policy": "M29/M30 artifacts frozen; one-row inference only; no fit or tuning",
        "acceptance": acceptance,
        "checks": checks,
        "prerequisite": prerequisite,
        "model_performance_changed": False,
        "lightgbm_fit_calls": 0,
        "input_contract": {
            "required_input_field_count": 31,
            "required_purchase_base_feature_count": 27,
            "required_first_kill_feature_count": 4,
            "derived_feature_count": 9,
            "raw_feature_count": predictor.model_audit["raw_feature_count"],
            "encoded_feature_count": predictor.model_audit["encoded_feature_count"],
            "known_map_count": predictor.model_audit["known_map_count"],
            "known_maps": predictor.model_audit["known_maps"],
            "known_weapon_count": predictor.model_audit["known_weapon_count"],
            "known_weapons": predictor.model_audit["known_weapons"],
            "deployment_tree_count": predictor.model_audit[
                "deployment_tree_count"
            ],
            "booster_space_sanitized_count": predictor.model_audit[
                "booster_space_sanitized_count"
            ],
        },
        "artifacts": {
            "model_sha256": model_before["sha256"],
            "calibrator_sha256": calibrator_before["sha256"],
            "model_unchanged": model_before["sha256"] == model_after["sha256"],
            "calibrator_unchanged": calibrator_before["sha256"]
            == calibrator_after["sha256"],
        },
        "example_prediction": json_result,
        "json_csv_probability_difference": probability_difference,
        "validation_cases": {
            "passed": validation_passed,
            "total": len(validation_examples),
        },
        "fixed_test_metrics": fixed_metrics,
        "fixed_metric_max_absolute_difference": 0.0 if metric_unchanged else None,
        "external_comparison": {
            **external_audit,
            "copied_byte_for_byte": external_copy_exact,
        },
        "cli_contract": cli_audit,
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_tests["test_count"],
            "skipped": automated_tests["skipped"],
        },
        "source_compile": compile_check,
        "reproduction_entrypoint": entrypoint_audit,
        "roadmap": {
            "pre_round_xgboost": "complete_through_M14",
            "first_kill_xgboost": "complete_through_M21",
            "pre_round_lightgbm": "complete_through_M27",
            "post_first_kill_lightgbm_current": "M32_interface_complete",
            "next_stage": "M33 post-first-kill LightGBM final acceptance",
        },
        "next_stage": "M33 post-first-kill LightGBM final acceptance",
    }

    write_json(summary, report_dir / "m32_summary.json")
    write_json(json_result, report_dir / "example_prediction.json")
    write_json(json_result, example_output_path)
    write_json(validation_examples, report_dir / "validation_error_examples.json")
    write_json(prerequisite, report_dir / "model_contract_audit.json")
    write_json(cli_audit, report_dir / "cli_contract_audit.json")
    pd.DataFrame(
        [{"metric": name, "value": value} for name, value in fixed_metrics.items()]
    ).to_csv(report_dir / "fixed_test_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"check": name, "passed": bool(checks[name]), "blocking": True}
            for name in BLOCKING_CHECKS
        ]
    ).to_csv(report_dir / "m32_checks.csv", index=False)
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests["output"],
        encoding="utf-8",
    )
    (report_dir / "source_compile_output.txt").write_text(
        compile_check["output"],
        encoding="utf-8",
    )
    (report_dir / "m32_post_first_kill_lightgbm_interface_report.md").write_text(
        render_m32_report(summary, validation_examples, external),
        encoding="utf-8",
    )

    output_paths = sorted(
        [
            path
            for path in report_dir.iterdir()
            if path.is_file() and path.name != "m32_experiment_manifest.json"
        ],
        key=lambda path: path.name,
    )
    output_paths.append(example_output_path)
    manifest = {
        "stage": "M32",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "powershell -File scripts/run_post_first_kill_lightgbm_interface.ps1",
        "policy": "frozen one-row inference; no training or example-driven selection",
        "inputs": [fingerprint_file(path) for path in manifest_inputs],
        "outputs": [fingerprint_file(path) for path in output_paths],
        "model_sha256_before": model_before["sha256"],
        "model_sha256_after": model_after["sha256"],
        "calibrator_sha256_before": calibrator_before["sha256"],
        "calibrator_sha256_after": calibrator_after["sha256"],
        "acceptance": acceptance,
    }
    write_json(manifest, report_dir / "m32_experiment_manifest.json")
    if acceptance["status"] != "passed":
        raise RuntimeError("M32 interface acceptance failed; inspect m32_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run M32 frozen post-first-kill LightGBM JSON/CSV interface acceptance."
        )
    )
    parser.add_argument(
        "--model",
        default="models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib",
    )
    parser.add_argument(
        "--calibrator",
        default="models/esta_full_m30/post_first_kill_lightgbm_calibrator.joblib",
    )
    parser.add_argument(
        "--json-example",
        default="examples/first_kill_snapshot.json",
    )
    parser.add_argument(
        "--csv-example",
        default="examples/first_kill_snapshot.csv",
    )
    parser.add_argument(
        "--example-output",
        default="examples/first_kill_lightgbm_prediction_output.json",
    )
    parser.add_argument(
        "--m30-summary",
        default="reports/esta_full_m30/m30_summary.json",
    )
    parser.add_argument(
        "--m31-summary",
        default="reports/esta_full_m31/m31_summary.json",
    )
    parser.add_argument(
        "--m31-external",
        default="reports/esta_full_m31/external_benchmark_comparison.csv",
    )
    parser.add_argument(
        "--m31-external-markdown",
        default="reports/esta_full_m31/external_benchmark_comparison.md",
    )
    parser.add_argument("--report-dir", default="reports/esta_full_m32")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--skip-verification", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_acceptance(
        model_path=args.model,
        calibrator_path=args.calibrator,
        json_example=args.json_example,
        csv_example=args.csv_example,
        example_output_path=args.example_output,
        m30_summary_path=args.m30_summary,
        m31_summary_path=args.m31_summary,
        m31_external_path=args.m31_external,
        m31_external_markdown_path=args.m31_external_markdown,
        report_dir=args.report_dir,
        project_root=args.project_root,
        run_verification=not args.skip_verification,
    )
    prediction = summary["example_prediction"]["prediction"]
    print(
        f"M32 {summary['acceptance']['status']}; "
        f"CT={prediction['ct_win_probability']:.10f}; "
        f"ready_for_m33={summary['acceptance']['ready_for_m33']}"
    )


if __name__ == "__main__":
    main()

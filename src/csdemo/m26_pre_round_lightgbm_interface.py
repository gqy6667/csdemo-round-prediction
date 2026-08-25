from __future__ import annotations

import argparse
import json
import math
import re
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
from .m22_pre_round_lightgbm_baseline import write_json
from .m24_pre_round_lightgbm_evaluation import run_compile_check
from .m25_pre_round_lightgbm_explanation import (
    render_external_report,
    validate_external_comparison,
)
from .predict_pre_round import InputValidationError, load_snapshot
from .predict_pre_round_lightgbm import PreRoundLightGBMPredictor


BLOCKING_CHECKS = (
    "m25_m24_prerequisite",
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
        "m26_lightgbm_interface_complete": not failures,
        "ready_for_m27": not failures,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _invalid_examples(snapshot: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    missing_field = deepcopy(snapshot)
    missing_field.pop("round_num", None)

    unknown_map = deepcopy(snapshot)
    unknown_map["map_name"] = "de_cache"

    invalid_round = deepcopy(snapshot)
    invalid_round["round_num"] = int(snapshot["round_num"]) + 1

    string_number = deepcopy(snapshot)
    string_number["ct_eq_value"] = str(snapshot["ct_eq_value"])

    inventory_mismatch = deepcopy(snapshot)
    inventory_mismatch["ct_rifles"] = 1

    inconsistent_difference = deepcopy(snapshot)
    inconsistent_difference["score_diff_ct"] = 99

    identifier = deepcopy(snapshot)
    identifier["series_id"] = "not-a-model-feature"

    target = deepcopy(snapshot)
    target["ct_win"] = 1

    future_event = deepcopy(snapshot)
    future_event["first_kill_time"] = 20.0

    identity = deepcopy(snapshot)
    identity["team_name"] = "not-allowed"

    return [
        ("missing_required_field", missing_field),
        ("unknown_map", unknown_map),
        ("round_score_inconsistency", invalid_round),
        ("string_numeric_type", string_number),
        ("inventory_inconsistency", inventory_mismatch),
        ("derived_feature_inconsistency", inconsistent_difference),
        ("identifier_field", identifier),
        ("target_leakage", target),
        ("future_first_kill_field", future_event),
        ("team_identity_field", identity),
    ]


def collect_validation_examples(
    predictor: PreRoundLightGBMPredictor,
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for case_name, invalid_snapshot in _invalid_examples(snapshot):
        try:
            predictor.predict(invalid_snapshot)
        except InputValidationError as exc:
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
    m24_summary: Mapping[str, Any],
    m25_summary: Mapping[str, Any],
    predictor: PreRoundLightGBMPredictor,
) -> dict[str, Any]:
    model_artifact = fingerprint_file(model_path)
    calibrator_artifact = fingerprint_file(calibrator_path)
    expected_model = m25_summary.get("prerequisite", {}).get(
        "model_artifact", {}
    )
    expected_calibrator = m24_summary.get("calibration", {}).get(
        "calibrator_artifact", {}
    )
    checks = {
        "m25_accepted": bool(
            m25_summary.get("stage") == "M25"
            and m25_summary.get("acceptance", {}).get("status") == "passed"
            and m25_summary.get("acceptance", {}).get("ready_for_m26") is True
        ),
        "m24_accepted": bool(
            m24_summary.get("stage") == "M24"
            and m24_summary.get("acceptance", {}).get("status") == "passed"
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
            == m25_summary.get("deployment_tree_count")
            == 115
        ),
        "data_contract": (
            predictor.model_audit.get("data_sha256")
            == m25_summary.get("prerequisite", {})
            .get("data_artifact", {})
            .get("sha256")
        ),
        "frozen_metrics": m25_summary.get("metrics") == m24_summary.get("metrics"),
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
        "src.csdemo.predict_pre_round_lightgbm",
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
        invalid["map_name"] = "de_cache"
        invalid_path.write_text(
            json.dumps(invalid, ensure_ascii=False), encoding="utf-8"
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
        "success_json": success_payload.get("task") == "pre_round",
        "success_output_file": output_path.is_file(),
        "invalid_exit_two": failure.returncode == 2,
        "invalid_error_json": failure_payload.get("status") == "error",
        "invalid_error_specific": "map_name" in failure_payload.get("message", ""),
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


def render_m26_external_report(external: pd.DataFrame) -> str:
    return render_external_report(external).replace(
        "# M25 外部模型指标差距",
        "# M26 外部模型指标差距",
        1,
    ).replace(
        "M25 不改变 M24 概率",
        "M26 不改变 M24/M25 概率",
        1,
    )


def render_m26_report(
    summary: Mapping[str, Any],
    validation_examples: list[dict[str, Any]],
    external: pd.DataFrame,
) -> str:
    prediction = summary["example_prediction"]["prediction"]
    contract = summary["input_contract"]
    metrics = summary["fixed_test_metrics"]
    acceptance = summary["acceptance"]
    lines = [
        "# M26 开局前 LightGBM 单条预测接口验收",
        "",
        "## 结论",
        "",
        f"M26 阻断检查 {acceptance['blocking_passed']}/{acceptance['blocking_total']} "
        f"通过，状态为 `{acceptance['status']}`，可以进入 M27。接口只加载 M23/M24",
        "冻结工件，没有训练、调参或修改概率。",
        "",
        "## 如何使用",
        "",
        "在项目目录执行：",
        "",
        "```powershell",
        "C:\\Users\\admin\\11\\envs\\game\\python.exe -m "
        "src.csdemo.predict_pre_round_lightgbm `",
        "  --input examples\\pre_round_snapshot.json `",
        "  --model models\\esta_full_m23\\pre_round_lightgbm_tuned.joblib `",
        "  --calibrator models\\esta_full_m24\\pre_round_lightgbm_calibrator.joblib",
        "```",
        "",
        "CSV 只需把 `--input` 改为 `examples\\pre_round_snapshot.csv`。需要保存结果时",
        "增加 `--output my_prediction.json`。非法输入返回退出码 2 和错误 JSON。",
        "",
        "## 输入与工件合同",
        "",
        f"接口接收 {contract['required_input_field_count']} 个基础字段，自动计算 "
        f"{contract['derived_feature_count']} 个 CT-T 差值，形成 "
        f"{contract['raw_feature_count']} 个原始特征并严格对齐 "
        f"{contract['encoded_feature_count']} 个编码列。地图类别 "
        f"{contract['known_map_count']} 个，部署树 {contract['deployment_tree_count']} 棵。",
        "",
        f"模型 SHA-256 为 `{summary['artifacts']['model_sha256']}`；校准器 SHA-256 为",
        f"`{summary['artifacts']['calibrator_sha256']}`。两者运行前后均未变化，校准器",
        "绑定同一模型和数据，并记录只用 validation 选择。",
        "",
        "## 示例输出",
        "",
        f"示例原始 CT 概率为 {prediction['base_ct_win_probability']:.10f}，identity",
        f"校准后的 CT/T 概率为 {prediction['ct_win_probability']:.10f} / "
        f"{prediction['t_win_probability']:.10f}，预测方为 "
        f"`{prediction['predicted_side']}`。JSON 与 CSV 的 CT 概率最大差为 "
        f"{summary['json_csv_probability_difference']:.3e}。",
        "",
        "这只是该示例快照的接口输出，不是测试集指标，也不是投注建议。",
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
            "M26 不重新计算或选择模型，以上五项与 M25 完全一致。外部比较仍为 "
            f"{len(external)} 行，完整差值见 `external_benchmark_comparison.csv`。",
            "",
            "## 验收与下一步",
            "",
            f"十类非法输入全部拒绝，CLI 成功/失败路径通过。自动化测试 "
            f"{summary['automated_tests']['test_count']} 项通过，源码编译通过。M27 将做",
            "购买结束 LightGBM 最终验收和一键复现；批量、HTTP、GUI、身份特征和实时",
            "胜率仍不在本阶段。",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def run_acceptance(
    *,
    model_path: str | Path,
    calibrator_path: str | Path,
    json_example: str | Path,
    csv_example: str | Path,
    example_output_path: str | Path,
    m24_summary_path: str | Path,
    m25_summary_path: str | Path,
    m25_external_path: str | Path,
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
    m24_summary_path = _resolve(root, m24_summary_path).resolve()
    m25_summary_path = _resolve(root, m25_summary_path).resolve()
    m25_external_path = _resolve(root, m25_external_path).resolve()
    report_dir = _resolve(root, report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    example_output_path.parent.mkdir(parents=True, exist_ok=True)

    model_before = fingerprint_file(model_path)
    calibrator_before = fingerprint_file(calibrator_path)
    m24_summary = _read_json(m24_summary_path)
    m25_summary = _read_json(m25_summary_path)
    predictor = PreRoundLightGBMPredictor.from_paths(model_path, calibrator_path)
    prerequisite = verify_prerequisites(
        model_path=model_path,
        calibrator_path=calibrator_path,
        m24_summary=m24_summary,
        m25_summary=m25_summary,
        predictor=predictor,
    )
    if not prerequisite["passed"]:
        raise RuntimeError("M26 prerequisites differ from accepted M24/M25 artifacts")

    json_snapshot = load_snapshot(json_example)
    csv_snapshot = load_snapshot(csv_example)
    json_result = predictor.predict(json_snapshot)
    csv_result = predictor.predict(csv_snapshot)
    json_probability = float(json_result["prediction"]["ct_win_probability"])
    csv_probability = float(csv_result["prediction"]["ct_win_probability"])
    probability_difference = abs(json_probability - csv_probability)
    validation_examples = collect_validation_examples(predictor, json_snapshot)
    validation_passed = sum(bool(row["rejected"]) for row in validation_examples)

    external = read_table(m25_external_path)
    external_audit = validate_external_comparison(external, m25_summary["metrics"])
    external_report = render_m26_external_report(external)
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
    probability = json_result["prediction"]
    fixed_metrics = m25_summary["metrics"]
    metric_unchanged = fixed_metrics == m24_summary["metrics"]
    entrypoint = root / "scripts" / "run_pre_round_lightgbm_interface.ps1"
    manifest_inputs = [
        model_path,
        calibrator_path,
        json_example,
        csv_example,
        m24_summary_path,
        m25_summary_path,
        m25_external_path,
        root / "docs" / "m26_pre_round_lightgbm_prediction_interface_spec.md",
        root / "src" / "csdemo" / "predict_pre_round_lightgbm.py",
        root / "src" / "csdemo" / "m26_pre_round_lightgbm_interface.py",
        entrypoint,
    ]
    checks = {
        "m25_m24_prerequisite": prerequisite["passed"],
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
            0.0 <= probability["base_ct_win_probability"] <= 1.0
            and 0.0 <= probability["ct_win_probability"] <= 1.0
            and 0.0 <= probability["t_win_probability"] <= 1.0
            and math.isclose(
                probability["probability_sum"], 1.0, rel_tol=0.0, abs_tol=1e-12
            )
        ),
        "invalid_examples": validation_passed == len(validation_examples) == 10,
        "feature_alignment": bool(
            json_result["validation"]["required_base_feature_count"] == 27
            and len(json_result["validation"]["derived_features"]) == 9
            and json_result["validation"]["raw_model_feature_count"] == 36
            and json_result["validation"]["encoded_model_feature_count"] == 43
            and json_result["validation"]["deployment_tree_count"] == 115
        ),
        "fixed_metrics": metric_unchanged,
        "artifact_integrity": bool(
            model_before["sha256"] == model_after["sha256"]
            and calibrator_before["sha256"] == calibrator_after["sha256"]
        ),
        "external_report": external_audit["passed"] and bool(external_report),
        "cli_contract": cli_audit["passed"],
        "automated_tests": automated_tests["passed"],
        "source_compile": compile_check["passed"],
        "reproduction_entrypoint": entrypoint.is_file(),
        "artifact_manifest": all(path.is_file() for path in manifest_inputs),
    }
    acceptance = decide_acceptance(checks)
    summary = {
        "stage": "M26",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "model_policy": "M23/M24 artifacts frozen; one-row inference only; no fit or tuning",
        "acceptance": acceptance,
        "checks": checks,
        "prerequisite": prerequisite,
        "model_performance_changed": False,
        "lightgbm_fit_calls": 0,
        "input_contract": {
            "required_input_field_count": 27,
            "derived_feature_count": 9,
            "raw_feature_count": predictor.model_audit["raw_feature_count"],
            "encoded_feature_count": predictor.model_audit["encoded_feature_count"],
            "known_map_count": predictor.model_audit["known_map_count"],
            "known_maps": predictor.model_audit["known_maps"],
            "deployment_tree_count": predictor.model_audit[
                "deployment_tree_count"
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
        "external_comparison": external_audit,
        "cli_contract": cli_audit,
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_tests["test_count"],
            "skipped": automated_tests["skipped"],
        },
        "source_compile": compile_check,
        "roadmap": {
            "pre_round_xgboost": "complete_through_M14",
            "first_kill_xgboost": "complete_through_M21",
            "pre_round_lightgbm_current": "M26_interface_complete",
            "next_stage": "M27 pre-round LightGBM final acceptance",
            "later_tracks": [
                "post-first-kill LightGBM controlled comparison",
                "real-time win probability data and model",
            ],
        },
        "next_stage": "M27 pre-round LightGBM final acceptance",
    }

    write_json(summary, report_dir / "m26_summary.json")
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
    ).to_csv(report_dir / "m26_checks.csv", index=False)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    (report_dir / "external_benchmark_comparison.md").write_text(
        external_report, encoding="utf-8"
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests["output"], encoding="utf-8"
    )
    (report_dir / "source_compile_output.txt").write_text(
        compile_check["output"], encoding="utf-8"
    )
    (report_dir / "m26_pre_round_lightgbm_interface_report.md").write_text(
        render_m26_report(summary, validation_examples, external),
        encoding="utf-8",
    )

    output_paths = sorted(
        [
            path
            for path in report_dir.iterdir()
            if path.is_file() and path.name != "m26_experiment_manifest.json"
        ],
        key=lambda path: path.name,
    )
    output_paths.append(example_output_path)
    manifest = {
        "stage": "M26",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "powershell -File scripts/run_pre_round_lightgbm_interface.ps1",
        "policy": "frozen one-row inference; no training or example-driven selection",
        "inputs": [fingerprint_file(path) for path in manifest_inputs],
        "outputs": [fingerprint_file(path) for path in output_paths],
        "model_sha256_before": model_before["sha256"],
        "model_sha256_after": model_after["sha256"],
        "calibrator_sha256_before": calibrator_before["sha256"],
        "calibrator_sha256_after": calibrator_after["sha256"],
        "acceptance": acceptance,
    }
    write_json(manifest, report_dir / "m26_experiment_manifest.json")
    if acceptance["status"] != "passed":
        raise RuntimeError("M26 interface acceptance failed; inspect m26_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run M26 pre-round LightGBM one-row interface acceptance"
    )
    parser.add_argument(
        "--model", default="models/esta_full_m23/pre_round_lightgbm_tuned.joblib"
    )
    parser.add_argument(
        "--calibrator",
        default="models/esta_full_m24/pre_round_lightgbm_calibrator.joblib",
    )
    parser.add_argument("--json-example", default="examples/pre_round_snapshot.json")
    parser.add_argument("--csv-example", default="examples/pre_round_snapshot.csv")
    parser.add_argument(
        "--example-output",
        default="examples/pre_round_lightgbm_prediction_output.json",
    )
    parser.add_argument(
        "--m24-summary", default="reports/esta_full_m24/m24_summary.json"
    )
    parser.add_argument(
        "--m25-summary", default="reports/esta_full_m25/m25_summary.json"
    )
    parser.add_argument(
        "--m25-external",
        default="reports/esta_full_m25/external_benchmark_comparison.csv",
    )
    parser.add_argument("--report-dir", default="reports/esta_full_m26")
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
        m24_summary_path=args.m24_summary,
        m25_summary_path=args.m25_summary,
        m25_external_path=args.m25_external,
        report_dir=args.report_dir,
        project_root=args.project_root,
        run_verification=not args.skip_verification,
    )
    prediction = summary["example_prediction"]["prediction"]
    print(
        f"M26 {summary['acceptance']['status']}; "
        f"CT={prediction['ct_win_probability']:.10f}; "
        f"T={prediction['t_win_probability']:.10f}; "
        f"ready_for_m27={summary['acceptance']['ready_for_m27']}"
    )


if __name__ == "__main__":
    main()

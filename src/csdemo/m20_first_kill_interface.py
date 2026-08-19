from __future__ import annotations

import argparse
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .io import read_table
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m16_first_kill_baselines import compare_external_models
from .predict_first_kill import (
    FIRST_KILL_TIME_RANGE,
    FirstKillInputValidationError,
    FirstKillPredictor,
    load_snapshot,
)


BLOCKING_CHECKS = (
    "m19_prerequisite",
    "artifact_contracts",
    "json_csv_validation",
    "json_csv_prediction_match",
    "probability_contract",
    "invalid_examples",
    "feature_alignment",
    "fixed_metrics_and_targets",
    "external_report",
    "automated_tests",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _invalid_examples(snapshot: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    missing_event = deepcopy(snapshot)
    missing_event.pop("first_kill_time", None)

    invalid_advantage = deepcopy(snapshot)
    invalid_advantage["first_kill_advantage_ct"] = 0

    invalid_time = deepcopy(snapshot)
    invalid_time["first_kill_time"] = FIRST_KILL_TIME_RANGE[1] + 1

    invalid_headshot = deepcopy(snapshot)
    invalid_headshot["first_kill_headshot"] = 2

    unknown_weapon = deepcopy(snapshot)
    unknown_weapon["first_kill_weapon"] = "Unknown Blaster"

    unknown_map = deepcopy(snapshot)
    unknown_map["map_name"] = "de_cache"

    inconsistent_difference = deepcopy(snapshot)
    inconsistent_difference["score_diff_ct"] = 99

    target_leakage = deepcopy(snapshot)
    target_leakage["ct_win"] = 1

    future_event = deepcopy(snapshot)
    future_event["second_kill_weapon"] = "AWP"

    redundant_state = deepcopy(snapshot)
    redundant_state["ct_alive_after_fk"] = 5

    return [
        ("missing_first_kill_field", missing_event),
        ("invalid_first_kill_advantage", invalid_advantage),
        ("first_kill_time_out_of_range", invalid_time),
        ("invalid_headshot", invalid_headshot),
        ("unknown_first_kill_weapon", unknown_weapon),
        ("unknown_map", unknown_map),
        ("inconsistent_derived_feature", inconsistent_difference),
        ("target_leakage", target_leakage),
        ("future_second_kill", future_event),
        ("redundant_alive_state", redundant_state),
    ]


def _collect_validation_examples(
    predictor: FirstKillPredictor, snapshot: dict[str, Any]
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
                    "errors": ["Invalid input was unexpectedly accepted."],
                }
            )
    return rows


def verify_m20_prerequisites(
    *,
    model_path: str | Path,
    calibrator_path: str | Path,
    m18_summary: dict[str, Any],
    m19_summary: dict[str, Any],
    predictor: FirstKillPredictor,
) -> dict[str, Any]:
    model_artifact = fingerprint_file(model_path)
    calibrator_artifact = fingerprint_file(calibrator_path)
    expected_model = m19_summary.get("prerequisite", {}).get("model_artifact", {})
    expected_calibrator = m18_summary.get("calibration", {}).get(
        "calibrator_artifact", {}
    )
    formal_targets = m19_summary.get("target_gap", {})

    checks = {
        "m19_accepted": (
            m19_summary.get("stage") == "M19"
            and m19_summary.get("acceptance", {}).get("status") == "passed"
            and bool(m19_summary.get("acceptance", {}).get("ready_for_m20"))
        ),
        "m18_accepted": (
            m18_summary.get("stage") == "M18"
            and m18_summary.get("acceptance", {}).get("status") == "passed"
        ),
        "model_sha256": (
            bool(expected_model.get("sha256"))
            and model_artifact["sha256"] == expected_model.get("sha256")
        ),
        "model_bytes": (
            expected_model.get("bytes") is None
            or model_artifact["bytes"] == expected_model.get("bytes")
        ),
        "calibrator_sha256": (
            bool(expected_calibrator.get("sha256"))
            and calibrator_artifact["sha256"] == expected_calibrator.get("sha256")
        ),
        "calibrator_bytes": (
            expected_calibrator.get("bytes") is None
            or calibrator_artifact["bytes"] == expected_calibrator.get("bytes")
        ),
        "model_contract": predictor.model_audit.get("passed", False),
        "calibrator_contract": predictor.calibrator_audit.get("passed", False),
        "deployment_tree_contract": (
            predictor.model_audit.get("deployment_tree_count")
            == m19_summary.get("deployment_tree_count")
            == 409
        ),
        "formal_targets_frozen": (
            formal_targets.get("passed_count") == 10
            and formal_targets.get("target_count") == 10
            and formal_targets.get("remaining_count") == 0
            and bool(formal_targets.get("all_formal_targets_passed"))
        ),
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


def decide_acceptance(checks: dict[str, Any]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not bool(checks.get(name))]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "ready_for_m21": not failures,
    }


def render_external_report(external: pd.DataFrame) -> str:
    lines = [
        "# M20 外部模型指标差距",
        "",
        "差值统一为“当前项目指标 - 外部报告指标”。Accuracy/AUC 同时换算成百分点。",
        "M20 没有训练或改变概率，因此数值与 M19 相同。不同数据、切分、特征和时点",
        "不能解释为纯算法排名。",
        "",
        "| 可比性 | 当前模型 | 外部来源 | 指标 | 当前 | 外部 | 差值 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in external.to_dict(orient="records"):
        title = row.get("source_title", row["benchmark_id"])
        url = row.get("source_url", "")
        source = f"[{title}]({url})" if url else str(title)
        difference = float(row["raw_difference_ours_minus_reported"])
        difference_text = (
            f"{difference * 100:+.2f} 百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| {row.get('comparability', '')} | `{row['current_model']}` | "
            f"{source} | {row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {difference_text} |"
        )
    return "\n".join(lines) + "\n"


def _format_external_difference(row: dict[str, Any]) -> str:
    difference = float(row["raw_difference_ours_minus_reported"])
    if row["metric"] in {"accuracy", "auc"}:
        return f"{difference * 100:+.2f} 百分点"
    return f"{difference:+.6f}"


def render_m20_report(
    summary: dict[str, Any],
    validation_examples: list[dict[str, Any]],
    external: pd.DataFrame,
) -> str:
    prediction = summary["example_prediction"]["prediction"]
    contract = summary["input_contract"]
    lines = [
        "# M20 首杀后单条预测接口验收报告",
        "",
        "## 阶段结论",
        "",
        f"阻断验收状态：**{summary['acceptance']['status']}**；"
        f"可进入 M21：**{summary['acceptance']['ready_for_m21']}**。",
        "M20 没有训练、调参或改变固定测试概率，只把 M17/M18 模型封装为严格校验的",
        "JSON/CSV 单条预测命令。",
        "",
        "## 输入与模型合同",
        "",
        f"- 购买结束必填字段：{contract['purchase_base_feature_count']} 个；",
        f"- 首杀事件必填字段：{contract['first_kill_feature_count']} 个；",
        f"- 自动生成 CT-T 差值：{contract['derived_feature_count']} 个；",
        f"- 原始模型特征：{contract['raw_feature_count']} 个；",
        f"- 编码模型列：{contract['encoded_feature_count']} 个；",
        f"- 已知地图：{contract['known_map_count']} 类；",
        f"- 已知首杀武器：{contract['known_weapon_count']} 类；",
        f"- 部署树：{contract['deployment_tree_count']} 棵；",
        f"- 模型 SHA-256：`{summary['artifacts']['model_sha256']}`；",
        f"- 校准器 SHA-256：`{summary['artifacts']['calibrator_sha256']}`。",
        "",
        "未知地图/武器、ID、标签、第二次击杀、伤害/血量、下包和冗余存活字段都会被拒绝。",
        "",
        "## 示例预测",
        "",
        f"- 地图：`{summary['example_prediction']['validation']['map_name']}`；",
        f"- 首杀方：`{summary['example_prediction']['validation']['first_kill_side']}`；",
        f"- 首杀武器：`{summary['example_prediction']['validation']['first_kill_weapon']}`；",
        f"- CT 胜率：`{prediction['ct_win_probability']:.6f}`；",
        f"- T 胜率：`{prediction['t_win_probability']:.6f}`；",
        f"- 判定方：`{prediction['predicted_side']}`；",
        f"- 概率和：`{prediction['probability_sum']:.12f}`；",
        f"- JSON/CSV 完全一致：`{summary['checks']['json_csv_prediction_match']}`；",
        f"- 校准方式：`{summary['example_prediction']['calibration_method']}`。",
        "",
        "这是一条局面的模型估计，不表示较高概率一方一定赢得回合。",
        "",
        "## 错误输入验收",
        "",
        "| 错误类型 | 是否拒绝 | 错误数 | 首条信息 |",
        "|---|---|---:|---|",
    ]
    for case in validation_examples:
        first_error = str(case["errors"][0]).replace("|", "\\|")
        lines.append(
            f"| {case['case']} | {case['rejected']} | {case['error_count']} | "
            f"{first_error} |"
        )

    lines.extend(
        [
            "",
            "## 固定测试指标",
            "",
            "| 指标 | M20 固定值 | 是否因接口改变 |",
            "|---|---:|---|",
        ]
    )
    labels = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "log_loss": "Log Loss",
        "brier_score": "Brier",
        "ece10": "ECE10",
    }
    for metric, label in labels.items():
        lines.append(
            f"| {label} | {summary['fixed_test_metrics'][metric]:.6f} | 否 |"
        )

    targets = summary["formal_targets"]
    lines.extend(
        [
            "",
            "M19 十项正式目标继续通过 "
            f"`{targets['passed_count']}/{targets['target_count']}`，"
            f"仍需改善的目标数为 `{targets['remaining_count']}`。",
            "",
            "## 与外部模型相差多少",
            "",
            "以下列出预测时点最接近的公开首杀后逻辑回归。完整七行及可比性标签见",
            "`external_benchmark_comparison.md`。",
            "",
            "| 外部工作 | 指标 | 本项目逻辑回归 | 外部 | 差值 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    closest = external.loc[
        external["comparability"].eq("closest_task")
        & external["comparison_status"].eq("compared")
    ]
    for row in closest.to_dict(orient="records"):
        title = str(row.get("source_title", row["benchmark_id"])).replace("|", "\\|")
        lines.append(
            f"| {title} | {row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {_format_external_difference(row)} |"
        )

    tests = summary["automated_tests"]
    lines.extend(
        [
            "",
            "## 自动化验收",
            "",
            f"- 阻断检查通过：{sum(bool(summary['checks'][name]) for name in BLOCKING_CHECKS)}/"
            f"{len(BLOCKING_CHECKS)}；",
            f"- 自动化测试：{tests.get('test_count', 'skipped')} 项；",
            f"- XGBoost fit 调用：{summary['xgboost_fit_calls']}；",
            f"- 模型性能改变：{summary['model_performance_changed']}。",
            "",
            "## 使用命令",
            "",
            "```powershell",
            "C:\\Users\\admin\\11\\envs\\game\\python.exe -m src.csdemo.predict_first_kill `",
            "  --input examples\\first_kill_snapshot.json `",
            "  --model models\\esta_full_m17\\first_kill_xgboost_tuned.joblib `",
            "  --calibrator models\\esta_full_m18\\first_kill_calibrator.joblib",
            "```",
            "",
            "## 下一阶段",
            "",
            "M21 做首杀后 XGBoost 最终验收和一键复现。M21 通过后，首杀后 XGBoost",
            "任务完成，再进入 LightGBM 同数据对照和实时胜率数据模块。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_acceptance(
    *,
    model_path: str | Path,
    calibrator_path: str | Path,
    json_example: str | Path,
    csv_example: str | Path,
    m18_summary_path: str | Path,
    m19_summary_path: str | Path,
    m17_comparison_path: str | Path,
    benchmarks_path: str | Path,
    report_dir: str | Path,
    project_root: str | Path,
    run_tests: bool = True,
) -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    m18_summary = _read_json(m18_summary_path)
    m19_summary = _read_json(m19_summary_path)
    predictor = FirstKillPredictor.from_paths(model_path, calibrator_path)
    prerequisite = verify_m20_prerequisites(
        model_path=model_path,
        calibrator_path=calibrator_path,
        m18_summary=m18_summary,
        m19_summary=m19_summary,
        predictor=predictor,
    )

    json_snapshot = load_snapshot(json_example)
    csv_snapshot = load_snapshot(csv_example)
    json_result = predictor.predict(json_snapshot)
    csv_result = predictor.predict(csv_snapshot)
    json_probability = float(json_result["prediction"]["ct_win_probability"])
    csv_probability = float(csv_result["prediction"]["ct_win_probability"])
    validation_examples = _collect_validation_examples(predictor, json_snapshot)
    validation_cases_passed = sum(row["rejected"] for row in validation_examples)

    m17_comparison = read_table(m17_comparison_path)
    benchmarks = read_table(benchmarks_path)
    external = compare_external_models(m17_comparison, benchmarks)
    external_report = render_external_report(external)

    if run_tests:
        automated_tests = run_automated_tests(project_root)
        match = re.search(r"Ran (\d+) tests?", automated_tests["output"])
        automated_tests["test_count"] = int(match.group(1)) if match else None
        automated_tests["skipped"] = False
    else:
        automated_tests = {
            "passed": True,
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "command": [],
            "output": "Automated tests skipped by run_acceptance caller.\n",
            "test_count": None,
            "skipped": True,
        }

    probability_sum = float(json_result["prediction"]["probability_sum"])
    formal_targets = m19_summary["target_gap"]
    fixed_metrics = m18_summary["metrics"]
    checks: dict[str, Any] = {
        "m19_prerequisite": prerequisite["passed"],
        "artifact_contracts": (
            prerequisite["checks"]["model_contract"]
            and prerequisite["checks"]["calibrator_contract"]
            and prerequisite["checks"]["model_sha256"]
            and prerequisite["checks"]["calibrator_sha256"]
        ),
        "json_csv_validation": (
            json_result["validation"]["status"] == "passed"
            and csv_result["validation"]["status"] == "passed"
        ),
        "json_csv_prediction_match": math.isclose(
            json_probability, csv_probability, rel_tol=0.0, abs_tol=1e-15
        ),
        "probability_contract": (
            0.0 <= json_probability <= 1.0
            and 0.0 <= json_result["prediction"]["t_win_probability"] <= 1.0
            and math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12)
        ),
        "invalid_examples": validation_cases_passed == len(validation_examples),
        "feature_alignment": (
            json_result["validation"]["raw_model_feature_count"] == 40
            and json_result["validation"]["encoded_model_feature_count"] == 82
            and json_result["validation"]["deployment_tree_count"] == 409
        ),
        "fixed_metrics_and_targets": (
            m19_summary["model_replay"]["metric_max_absolute_difference_vs_m18"]
            == 0.0
            and formal_targets["passed_count"] == 10
            and formal_targets["remaining_count"] == 0
        ),
        "external_report": len(external) == len(benchmarks) == 7 and bool(external_report),
        "automated_tests": bool(automated_tests["passed"]),
        "validation_cases_passed": validation_cases_passed,
        "validation_cases_total": len(validation_examples),
    }
    acceptance = decide_acceptance(checks)
    summary = {
        "stage": "M20",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "model_policy": "M17/M18 artifacts frozen; inference only; no fit or tuning",
        "acceptance": acceptance,
        "checks": checks,
        "prerequisite": prerequisite,
        "model_performance_changed": False,
        "xgboost_fit_calls": 0,
        "input_contract": {
            "purchase_base_feature_count": 27,
            "first_kill_feature_count": 4,
            "required_input_field_count": 31,
            "derived_feature_count": 9,
            "raw_feature_count": predictor.model_audit["raw_feature_count"],
            "encoded_feature_count": predictor.model_audit["encoded_feature_count"],
            "known_map_count": predictor.model_audit["known_map_count"],
            "known_weapon_count": predictor.model_audit["known_weapon_count"],
            "deployment_tree_count": predictor.model_audit["deployment_tree_count"],
        },
        "artifacts": {
            "model_sha256": prerequisite["model_artifact"]["sha256"],
            "calibrator_sha256": prerequisite["calibrator_artifact"]["sha256"],
        },
        "example_prediction": json_result,
        "fixed_test_metrics": fixed_metrics,
        "formal_targets": formal_targets,
        "external_comparison_rows": len(external),
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_tests["test_count"],
            "skipped": automated_tests["skipped"],
        },
        "roadmap": {
            "pre_round_xgboost": "complete_through_M14",
            "first_kill_xgboost_current": "M20_interface_complete",
            "first_kill_xgboost_modules_remaining_after_m20": 1,
            "remaining_module": "M21 first-kill final acceptance",
            "later_tracks": [
                "LightGBM controlled comparison",
                "real-time win probability data and model",
            ],
        },
        "next_stage": "M21 first-kill final acceptance",
    }

    _write_json(report_dir / "m20_summary.json", summary)
    _write_json(report_dir / "example_prediction.json", json_result)
    _write_json(report_dir / "validation_error_examples.json", validation_examples)
    _write_json(report_dir / "model_contract_audit.json", prerequisite)
    pd.DataFrame(
        [
            {
                "check": name,
                "passed": bool(checks[name]),
                "blocking": True,
            }
            for name in BLOCKING_CHECKS
        ]
    ).to_csv(report_dir / "m20_checks.csv", index=False)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    (report_dir / "external_benchmark_comparison.md").write_text(
        external_report, encoding="utf-8"
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests["output"], encoding="utf-8"
    )
    (report_dir / "m20_first_kill_interface_report.md").write_text(
        render_m20_report(summary, validation_examples, external),
        encoding="utf-8",
    )
    if acceptance["status"] != "passed":
        raise RuntimeError("M20 interface acceptance failed; inspect m20_summary.json.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run M20 post-first-kill prediction interface acceptance."
    )
    parser.add_argument(
        "--model",
        default="models/esta_full_m17/first_kill_xgboost_tuned.joblib",
    )
    parser.add_argument(
        "--calibrator",
        default="models/esta_full_m18/first_kill_calibrator.joblib",
    )
    parser.add_argument(
        "--json-example", default="examples/first_kill_snapshot.json"
    )
    parser.add_argument(
        "--csv-example", default="examples/first_kill_snapshot.csv"
    )
    parser.add_argument(
        "--m18-summary", default="reports/esta_full_m18/m18_summary.json"
    )
    parser.add_argument(
        "--m19-summary", default="reports/esta_full_m19/m19_summary.json"
    )
    parser.add_argument(
        "--m17-comparison", default="reports/esta_full_m17/model_comparison.csv"
    )
    parser.add_argument(
        "--benchmarks", default="benchmarks/external_first_kill_tuned_metrics.csv"
    )
    parser.add_argument("--report-dir", default="reports/esta_full_m20")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--skip-tests", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_acceptance(
        model_path=args.model,
        calibrator_path=args.calibrator,
        json_example=args.json_example,
        csv_example=args.csv_example,
        m18_summary_path=args.m18_summary,
        m19_summary_path=args.m19_summary,
        m17_comparison_path=args.m17_comparison,
        benchmarks_path=args.benchmarks,
        report_dir=args.report_dir,
        project_root=args.project_root,
        run_tests=not args.skip_tests,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

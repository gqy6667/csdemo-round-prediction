from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from .benchmark_comparison import load_current_metrics, run as run_benchmarks
from .predict_pre_round import (
    BASE_FEATURES,
    DIFFERENCE_FEATURES,
    InputValidationError,
    PreRoundPredictor,
    load_snapshot,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _invalid_examples(snapshot: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    missing = deepcopy(snapshot)
    missing.pop("round_num", None)

    wrong_type = deepcopy(snapshot)
    wrong_type["ct_cash"] = "3500"

    out_of_range = deepcopy(snapshot)
    out_of_range["ct_helmets"] = 6

    unknown_map = deepcopy(snapshot)
    unknown_map["map_name"] = "de_cache"

    inconsistent_difference = deepcopy(snapshot)
    inconsistent_difference["score_diff_ct"] = 99

    return [
        ("missing_required_field", missing),
        ("wrong_numeric_type", wrong_type),
        ("out_of_range", out_of_range),
        ("unknown_map", unknown_map),
        ("inconsistent_derived_feature", inconsistent_difference),
    ]


def _collect_validation_examples(
    predictor: PreRoundPredictor, snapshot: dict[str, Any]
) -> list[dict[str, Any]]:
    results = []
    for case_name, invalid_snapshot in _invalid_examples(snapshot):
        try:
            predictor.predict(invalid_snapshot)
        except InputValidationError as exc:
            results.append(
                {
                    "case": case_name,
                    "rejected": True,
                    "error_count": len(exc.errors),
                    "errors": list(exc.errors),
                }
            )
        else:
            results.append(
                {
                    "case": case_name,
                    "rejected": False,
                    "error_count": 0,
                    "errors": ["Invalid input was unexpectedly accepted."],
                }
            )
    return results


def _format_external_difference(row: pd.Series) -> str:
    percentage_points = row.get("difference_percentage_points")
    if pd.notna(percentage_points):
        return f"{percentage_points:+.2f} 个百分点"
    return f"{row['raw_difference_ours_minus_reported']:+.6f}"


def _write_report(
    path: Path,
    summary: dict[str, Any],
    validation_examples: list[dict[str, Any]],
    comparison: pd.DataFrame,
) -> None:
    prediction = summary["example_prediction"]["prediction"]
    checks = summary["checks"]
    lines = [
        "# M13 独立预测接口验收报告",
        "",
        "## 阶段结论",
        "",
        f"状态：**{summary['status']}**。M8 调优模型和 M10 校准选择均未重新训练，",
        "所以 M9 固定测试集指标不变；本阶段只把模型封装成可校验、可重复使用的单回合接口。",
        "",
        "## 输入与预处理",
        "",
        f"- 用户输入基础字段：{summary['input_contract']['base_feature_count']} 个。",
        f"- 接口自动计算 CT-T 差值：{summary['input_contract']['derived_feature_count']} 个。",
        f"- 独热编码前模型字段：{summary['input_contract']['pre_encoding_feature_count']} 个。",
        f"- 按保存的模型列对齐后：{summary['input_contract']['encoded_feature_count']} 个。",
        f"- 已知地图：{', '.join(summary['input_contract']['known_maps'])}。",
        "- 时间点定义：购买结束、冻结时间结束、第一次交火之前。",
        "",
        "训练阶段的 `prepare_features()` 被推理接口直接复用；地图类别编码完成后，",
        "再按模型 bundle 中保存的 43 列重排，缺少的独热列补 0。",
        "",
        "## 示例结果",
        "",
        f"- CT 胜率：`{prediction['ct_win_probability']:.6f}`",
        f"- T 胜率：`{prediction['t_win_probability']:.6f}`",
        f"- 判定方：`{prediction['predicted_side']}`",
        f"- 两个概率之和：`{prediction['probability_sum']:.12f}`",
        f"- JSON/CSV 结果一致：`{checks['json_csv_prediction_match']}`",
        f"- 校准方式：`{summary['example_prediction']['calibration_method']}`",
        "",
        "概率表示模型在当前数据和特征下的估计，不代表比赛一定按较高概率一方获胜。",
        "",
        "## 错误输入验收",
        "",
        "| 错误类型 | 是否拒绝 | 返回错误数 | 首条信息 |",
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
            "| 指标 | M13 当前值 | 是否因接口阶段改变 |",
            "|---|---:|---|",
        ]
    )
    metric_labels = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "log_loss": "Log Loss",
        "brier_score": "Brier Score",
        "ece10": "ECE10",
    }
    for metric, label in metric_labels.items():
        if metric in summary["fixed_test_metrics"]:
            lines.append(
                f"| {label} | {summary['fixed_test_metrics'][metric]:.6f} | 否 |"
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
            "## 与外部模型相差多少",
            "",
            "差值固定为“我们的指标 - 外部报告指标”。以下只列预测时点最接近的公开结果；",
            "数据集、年份与切分方式不同，所以这是参考差距，不是受控模型排名。",
            "",
            "| 外部工作 | 指标 | 我们 | 外部 | 差值 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in closest.iterrows():
        source = str(row.get("source_title", row["benchmark_id"])).replace("|", "\\|")
        lines.append(
            f"| {source} | {row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {_format_external_difference(row)} |"
        )

    lines.extend(
        [
            "",
            "完整的可比性分组、来源链接和所有数值差见",
            "`external_benchmark_comparison.md`。",
            "",
            "## 使用命令",
            "",
            "```powershell",
            "C:\\Users\\admin\\11\\envs\\game\\python.exe -m src.csdemo.predict_pre_round `",
            "  --input examples\\pre_round_snapshot.json `",
            "  --model models\\esta_full_m8_tuned\\pre_round_xgb.joblib `",
            "  --calibrator models\\esta_full_m10\\pre_round_calibrator.joblib",
            "```",
            "",
            "## 下一阶段",
            "",
            "M14 做开局前 XGBoost 最终验收：从干净环境按文档复跑关键命令、整理未达目标",
            "指标和剩余风险，然后决定先优化特征还是开始首杀后模型。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_acceptance(
    *,
    model_path: str | Path,
    calibrator_path: str | Path,
    json_example: str | Path,
    csv_example: str | Path,
    metrics_path: str | Path,
    benchmarks_path: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = PreRoundPredictor.from_paths(model_path, calibrator_path)
    json_snapshot = load_snapshot(json_example)
    csv_snapshot = load_snapshot(csv_example)
    json_result = predictor.predict(json_snapshot)
    csv_result = predictor.predict(csv_snapshot)

    json_probability = json_result["prediction"]["ct_win_probability"]
    csv_probability = csv_result["prediction"]["ct_win_probability"]
    probability_sum = json_result["prediction"]["probability_sum"]
    validation_examples = _collect_validation_examples(predictor, json_snapshot)
    validation_cases_passed = sum(case["rejected"] for case in validation_examples)

    checks = {
        "json_validation_passed": json_result["validation"]["status"] == "passed",
        "csv_validation_passed": csv_result["validation"]["status"] == "passed",
        "json_csv_prediction_match": math.isclose(
            json_probability, csv_probability, rel_tol=0.0, abs_tol=1e-15
        ),
        "probabilities_in_range": (
            0.0 <= json_probability <= 1.0
            and 0.0 <= json_result["prediction"]["t_win_probability"] <= 1.0
        ),
        "probabilities_sum_to_one": math.isclose(
            probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "validation_cases_passed": validation_cases_passed,
        "validation_cases_total": len(validation_examples),
    }
    required_boolean_checks = (
        "json_validation_passed",
        "csv_validation_passed",
        "json_csv_prediction_match",
        "probabilities_in_range",
        "probabilities_sum_to_one",
    )
    passed = all(checks[name] for name in required_boolean_checks) and (
        validation_cases_passed == len(validation_examples)
    )

    metrics = load_current_metrics(metrics_path)
    comparison = run_benchmarks(
        metrics_path,
        benchmarks_path,
        output_dir,
        stage_label="M13",
    )
    summary = {
        "stage": "M13",
        "status": "passed" if passed else "failed",
        "snapshot_definition": "purchase end, before combat",
        "model_performance_changed": False,
        "input_contract": {
            "base_feature_count": len(BASE_FEATURES),
            "derived_feature_count": len(DIFFERENCE_FEATURES),
            "pre_encoding_feature_count": len(BASE_FEATURES) + len(DIFFERENCE_FEATURES),
            "encoded_feature_count": len(predictor.columns),
            "known_maps": sorted(predictor.known_maps),
        },
        "checks": checks,
        "example_prediction": json_result,
        "fixed_test_metrics": metrics,
        "external_benchmark_rows": len(comparison),
    }
    _write_json(output_dir / "m13_summary.json", summary)
    _write_json(output_dir / "example_prediction.json", json_result)
    _write_json(output_dir / "validation_error_examples.json", validation_examples)
    _write_report(
        output_dir / "m13_interface_report.md",
        summary,
        validation_examples,
        comparison,
    )
    if not passed:
        raise RuntimeError("M13 interface acceptance checks failed; inspect m13_summary.json.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M13 prediction interface acceptance.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument("--json-example", required=True)
    parser.add_argument("--csv-example", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--benchmarks", required=True)
    parser.add_argument("--report-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_acceptance(
        model_path=args.model,
        calibrator_path=args.calibrator,
        json_example=args.json_example,
        csv_example=args.csv_example,
        metrics_path=args.metrics,
        benchmarks_path=args.benchmarks,
        report_dir=args.report_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

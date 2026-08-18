from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_BENCHMARK_COLUMNS = {
    "benchmark_id",
    "metric",
    "reported_value",
    "direction",
}
VALID_DIRECTIONS = {"higher", "lower"}
PROPORTION_METRICS = {"accuracy", "auc", "f1", "precision", "recall", "ece10"}
COMPARABILITY_LABELS = {
    "closest_task": "任务时点最接近",
    "partial": "部分可比",
    "not_comparable": "不可直接比较",
}


def load_current_metrics(path: str | Path) -> dict[str, float]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError("Metrics JSON must contain a metrics object or metric values.")
    return {
        str(name): float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }


def _validate_benchmarks(benchmarks: pd.DataFrame) -> None:
    missing = REQUIRED_BENCHMARK_COLUMNS - set(benchmarks.columns)
    if missing:
        raise KeyError(f"Benchmark table is missing columns: {sorted(missing)}")
    if benchmarks.empty:
        raise ValueError("Benchmark table must contain at least one row.")
    if benchmarks["benchmark_id"].isna().any() or benchmarks["benchmark_id"].duplicated().any():
        raise ValueError("benchmark_id values must be present and unique.")
    invalid_directions = set(benchmarks["direction"].dropna()) - VALID_DIRECTIONS
    if invalid_directions:
        raise ValueError(
            "Benchmark direction must be higher or lower; found "
            f"{sorted(invalid_directions)}."
        )
    reported = pd.to_numeric(benchmarks["reported_value"], errors="coerce")
    if reported.isna().any() or not np.isfinite(reported).all():
        raise ValueError("reported_value values must be finite numbers.")


def compare_benchmarks(
    current_metrics: dict[str, float], benchmarks: pd.DataFrame
) -> pd.DataFrame:
    """Compare current metrics with externally reported point estimates.

    ``raw_difference_ours_minus_reported`` always means current minus external.
    ``performance_advantage_ours`` is positive only when the current model performs
    better after accounting for whether a metric should be high or low.
    """

    _validate_benchmarks(benchmarks)
    result = benchmarks.copy()
    result["reported_value"] = pd.to_numeric(result["reported_value"])
    result["current_value"] = result["metric"].map(current_metrics)
    available = result["current_value"].notna()
    result["comparison_status"] = np.where(
        available, "compared", "current_metric_unavailable"
    )
    result["raw_difference_ours_minus_reported"] = (
        result["current_value"] - result["reported_value"]
    )
    higher = result["direction"].eq("higher")
    result["performance_advantage_ours"] = np.where(
        available,
        np.where(
            higher,
            result["raw_difference_ours_minus_reported"],
            -result["raw_difference_ours_minus_reported"],
        ),
        np.nan,
    )
    result["ours_performs_better"] = pd.array(
        result["performance_advantage_ours"].gt(0).where(available), dtype="boolean"
    )
    result["difference_percentage_points"] = np.where(
        available & result["metric"].isin(PROPORTION_METRICS),
        result["raw_difference_ours_minus_reported"] * 100,
        np.nan,
    )
    return result


def _format_difference(row: pd.Series) -> str:
    if row["comparison_status"] != "compared":
        return "当前未计算"
    if np.isfinite(row["difference_percentage_points"]):
        return f"{row['difference_percentage_points']:+.2f} 个百分点"
    return f"{row['raw_difference_ours_minus_reported']:+.6f}"


def _performance_text(row: pd.Series) -> str:
    if row["comparison_status"] != "compared":
        return "不可计算"
    if row.get("comparability") == "not_comparable":
        return "仅数值差，不判断优劣"
    advantage = row["performance_advantage_ours"]
    if np.isclose(advantage, 0.0):
        return "相同"
    return f"我们的模型{'较好' if advantage > 0 else '较差'} {abs(advantage):.6f}"


def write_markdown_report(
    comparison: pd.DataFrame,
    current_metrics: dict[str, float],
    path: str | Path,
    *,
    stage_label: str,
) -> None:
    lines = [
        f"# {stage_label} 外部模型指标对照",
        "",
        "差值统一定义为 `我们的指标 - 外部报告指标`。Accuracy/AUC 的差值同时换算为",
        "百分点；`performance_advantage_ours` 已按指标方向换算，正数才表示我们的模型更好。",
        "不同数据集、年代、特征和切分会影响结果，因此这些数字不是同一排行榜。",
        "",
        "## 当前模型",
        "",
        "| 指标 | 当前值 |",
        "|---|---:|",
    ]
    for metric in ("accuracy", "auc", "log_loss", "brier_score", "ece10"):
        if metric in current_metrics:
            lines.append(f"| {metric} | {current_metrics[metric]:.6f} |")

    group_order = ("closest_task", "partial", "not_comparable")
    for comparability in group_order:
        if "comparability" not in comparison.columns:
            continue
        group = comparison[comparison["comparability"].eq(comparability)]
        if group.empty:
            continue
        lines.extend(
            [
                "",
                f"## {COMPARABILITY_LABELS[comparability]}",
                "",
                "| 外部工作 | 模型 | 时点 | 指标 | 我们 | 外部 | 原始差值 | 方向修正后 |",
                "|---|---|---|---|---:|---:|---:|---|",
            ]
        )
        for _, row in group.iterrows():
            title = str(row.get("source_title", row["benchmark_id"]))
            url = row.get("source_url", "")
            source = f"[{title}]({url})" if isinstance(url, str) and url else title
            current = (
                f"{row['current_value']:.6f}"
                if row["comparison_status"] == "compared"
                else "未计算"
            )
            lines.append(
                f"| {source} | {row.get('model', '')} | "
                f"{row.get('prediction_point', '')} | {row['metric']} | {current} | "
                f"{row['reported_value']:.6f} | {_format_difference(row)} | "
                f"{_performance_text(row)} |"
            )

    if "notes" in comparison.columns:
        notes = comparison[
            ["benchmark_id", "notes"]
        ].drop_duplicates().dropna(subset=["notes"])
        if not notes.empty:
            lines.extend(["", "## 可比性说明", ""])
            for _, row in notes.iterrows():
                lines.append(f"- `{row['benchmark_id']}`：{row['notes']}")

    lines.extend(
        [
            "",
            "## 使用规则",
            "",
            "后续阶段报告继续使用同一张结构化基准表并重新生成本报告。只有在预测时点、",
            "数据集、划分单位和评价代码都一致时，才允许把差值解释为模型本身的优劣。",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    metrics_path: str | Path,
    benchmarks_path: str | Path,
    report_dir: str | Path,
    *,
    stage_label: str = "M11",
) -> pd.DataFrame:
    current_metrics = load_current_metrics(metrics_path)
    benchmarks = pd.read_csv(benchmarks_path)
    comparison = compare_benchmarks(current_metrics, benchmarks)
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "external_benchmark_comparison.csv", index=False)
    write_markdown_report(
        comparison,
        current_metrics,
        output_dir / "external_benchmark_comparison.md",
        stage_label=stage_label,
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current model metrics with published external results."
    )
    parser.add_argument("--metrics", required=True, help="JSON file containing metrics.")
    parser.add_argument("--benchmarks", required=True, help="External benchmark CSV.")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--stage-label", default="M11")
    args = parser.parse_args()

    comparison = run(
        args.metrics,
        args.benchmarks,
        args.report_dir,
        stage_label=args.stage_label,
    )
    columns = [
        "benchmark_id",
        "metric",
        "current_value",
        "reported_value",
        "raw_difference_ours_minus_reported",
        "performance_advantage_ours",
        "comparability",
    ]
    print(comparison[[column for column in columns if column in comparison]].to_string(index=False))


if __name__ == "__main__":
    main()

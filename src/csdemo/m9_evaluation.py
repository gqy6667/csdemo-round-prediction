from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from .config import LABEL_COL, RANDOM_STATE, REPORT_DIR
from .io import read_table
from .metrics import probability_metrics
from .train_xgb import prepare_xy

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_ORDER = ("accuracy", "auc", "log_loss", "brier_score", "ece10")
TARGETS = {
    "auc": {"minimum": 0.70, "stage": 0.73, "excellent": 0.75, "higher_is_better": True},
    "log_loss": {"minimum": 0.61, "stage": 0.58, "excellent": 0.56, "higher_is_better": False},
    "accuracy": {"minimum": 0.64, "stage": 0.66, "excellent": 0.68, "higher_is_better": True},
    "brier_score": {"minimum": 0.21, "stage": 0.195, "excellent": 0.185, "higher_is_better": False},
}


def calibration_table(y_true, probability, *, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    proba = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(proba):
        raise ValueError("y_true and probability must have the same non-zero length")
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("probability values must be finite and between 0 and 1")

    bin_ids = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        mean_probability = float(proba[mask].mean()) if mask.any() else float("nan")
        observed_rate = float(y[mask].mean()) if mask.any() else float("nan")
        rows.append(
            {
                "bin": bin_id + 1,
                "lower_bound": bin_id / n_bins,
                "upper_bound": (bin_id + 1) / n_bins,
                "count": int(mask.sum()),
                "mean_probability": mean_probability,
                "observed_ct_win_rate": observed_rate,
                "absolute_gap": abs(mean_probability - observed_rate),
            }
        )
    return pd.DataFrame(rows)


def confusion_counts(y_true, probability, *, threshold: float = 0.5) -> dict[str, int]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    pred = (np.asarray(probability, dtype=float).reshape(-1) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def bootstrap_metric_intervals(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    required = {"series_id", "y_true", "ct_win_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Bootstrap predictions are missing columns: {sorted(missing)}")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    y = predictions["y_true"].to_numpy(dtype=int)
    proba = predictions["ct_win_probability"].to_numpy(dtype=float)
    series = predictions["series_id"].astype(str).to_numpy()
    unique_series = np.unique(series)
    group_positions = [np.flatnonzero(series == value) for value in unique_series]
    point = probability_metrics(y, proba, n_bins=10)

    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in METRIC_ORDER}
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(group_positions), size=len(group_positions))
        positions = np.concatenate([group_positions[index] for index in chosen])
        values = probability_metrics(y[positions], proba[positions], n_bins=10)
        for metric in METRIC_ORDER:
            if np.isfinite(values[metric]):
                samples[metric].append(values[metric])

    rows = []
    for metric in METRIC_ORDER:
        values = np.asarray(samples[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "successful_bootstraps": int(len(values)),
                "bootstrap_unit": "series_id",
            }
        )
    return pd.DataFrame(rows)


def threshold_metrics(y_true, probability) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    rows = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        threshold_value = round(float(threshold), 1)
        counts = confusion_counts(y, probability, threshold=threshold_value)
        tn, fp = counts["true_negative"], counts["false_positive"]
        fn, tp = counts["false_negative"], counts["true_positive"]
        rows.append(
            {
                "threshold": threshold_value,
                **counts,
                "accuracy": (tn + tp) / len(y),
                "precision_ct": tp / (tp + fp) if tp + fp else float("nan"),
                "recall_ct": tp / (tp + fn) if tp + fn else float("nan"),
                "specificity_t": tn / (tn + fp) if tn + fp else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_value, group in predictions.groupby(group_column, dropna=False):
        metrics = probability_metrics(
            group["y_true"], group["ct_win_probability"], n_bins=10
        )
        rows.append(
            {
                group_column: group_value,
                "rounds": int(len(group)),
                "series": int(group["series_id"].nunique()),
                "ct_win_rate": float(group["y_true"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("rounds", ascending=False)


def make_predictions(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    if "model" not in bundle or "columns" not in bundle:
        raise KeyError("Model bundle must contain model and columns.")
    test_df = df[df["split"].eq("test")].copy()
    if test_df.empty:
        raise ValueError("Evaluation data has no test rows.")

    x_test, y_test = prepare_xy(test_df)
    x_test = x_test.reindex(columns=bundle["columns"], fill_value=0)
    probability = bundle["model"].predict_proba(x_test)[:, 1]
    metadata = [
        column
        for column in ("series_id", "game_id", "round_id", "map_name", "round_num")
        if column in test_df.columns
    ]
    predictions = test_df[metadata].reset_index(drop=True)
    if "game_id" in predictions.columns:
        predictions["source_subset"] = (
            predictions["game_id"].astype(str).str.split(":", n=1).str[0]
        )
    predictions["y_true"] = y_test.to_numpy(dtype=int)
    predictions["ct_win_probability"] = probability
    predictions["t_win_probability"] = 1.0 - probability
    predictions["predicted_label"] = (probability >= 0.5).astype(int)
    predictions["correct"] = predictions["predicted_label"].eq(predictions["y_true"])
    return predictions


def probability_distribution(predictions: pd.DataFrame, *, n_bins: int = 20) -> pd.DataFrame:
    proba = predictions["ct_win_probability"].to_numpy()
    bin_ids = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    rows = []
    for label in (0, 1):
        label_mask = predictions["y_true"].to_numpy() == label
        denominator = int(label_mask.sum())
        for bin_id in range(n_bins):
            count = int((label_mask & (bin_ids == bin_id)).sum())
            rows.append(
                {
                    "actual_label": label,
                    "bin": bin_id + 1,
                    "lower_bound": bin_id / n_bins,
                    "upper_bound": (bin_id + 1) / n_bins,
                    "count": count,
                    "fraction_within_label": count / denominator,
                }
            )
    return pd.DataFrame(rows)


def target_assessment(metrics: dict[str, float]) -> dict:
    assessment = {}
    for metric, targets in TARGETS.items():
        value = metrics[metric]
        higher = targets["higher_is_better"]
        assessment[metric] = {
            "value": value,
            "minimum_passed": value >= targets["minimum"] if higher else value <= targets["minimum"],
            "stage_passed": value >= targets["stage"] if higher else value <= targets["stage"],
            "excellent_passed": value >= targets["excellent"] if higher else value <= targets["excellent"],
            **targets,
        }
    return assessment


def save_plots(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    distribution: pd.DataFrame,
    report_dir: Path,
) -> None:
    y = predictions["y_true"].to_numpy()
    proba = predictions["ct_win_probability"].to_numpy()
    fpr, tpr, _ = roc_curve(y, proba)
    auc = probability_metrics(y, proba)["auc"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(fpr, tpr, color="#176B87", linewidth=2, label=f"XGBoost AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    counts = confusion_counts(y, proba)
    matrix = np.array(
        [[counts["true_negative"], counts["false_positive"]],
         [counts["false_negative"], counts["true_positive"]]]
    )
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            text_color = "white" if matrix[row, column] > matrix.max() / 2 else "#1F2937"
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color=text_color,
            )
    ax.set_xticks([0, 1], labels=["Predicted T", "Predicted CT"])
    ax.set_yticks([0, 1], labels=["Actual T", "Actual CT"])
    ax.set_title("Confusion matrix at threshold 0.5")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for label, color, name in ((0, "#C44E52", "Actual T win"), (1, "#176B87", "Actual CT win")):
        part = distribution[distribution["actual_label"].eq(label)]
        centers = (part["lower_bound"] + part["upper_bound"]) / 2
        ax.step(centers, part["fraction_within_label"], where="mid", color=color, linewidth=2, label=name)
    ax.set(xlabel="Predicted CT win probability", ylabel="Fraction within actual class", title="Probability distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "probability_distribution.png", dpi=160)
    plt.close(fig)

    valid = calibration[calibration["count"].gt(0)]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.plot(valid["mean_probability"], valid["observed_ct_win_rate"], color="#176B87", marker="o", linewidth=2, label="XGBoost")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted CT probability", ylabel="Observed CT win rate", title="Reliability curve")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(report_dir / "reliability_curve.png", dpi=160)
    plt.close(fig)


def write_report(summary: dict, intervals: pd.DataFrame, path: Path) -> None:
    indexed = intervals.set_index("metric")
    assessment = summary["target_assessment"]
    minimum_passed = sum(item["minimum_passed"] for item in assessment.values())
    stage_passed = sum(item["stage_passed"] for item in assessment.values())
    lines = [
        "# M9 Unified Evaluation Report",
        "",
        "The fixed M8 XGBoost model was evaluated once on the fixed test split.",
        "Confidence intervals use series-level bootstrap resampling.",
        "",
        "| Metric | Point | 95% CI | Minimum | Stage target | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric in ("auc", "log_loss", "accuracy", "brier_score"):
        row = indexed.loc[metric]
        target = assessment[metric]
        result = "minimum passed" if target["minimum_passed"] else "minimum not met"
        comparator = ">=" if target["higher_is_better"] else "<="
        lines.append(
            f"| {metric} | {row['point_estimate']:.6f} | "
            f"[{row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}] | "
            f"{comparator} {target['minimum']:.3f} | "
            f"{comparator} {target['stage']:.3f} | {result} |"
        )
    lines.extend(
        [
            "",
            f"Test rounds: {summary['test_rounds']:,}; series: {summary['test_series']:,}.",
            f"Bootstrap repetitions: {summary['bootstrap_samples']:,}; seed: {summary['seed']}.",
            "",
            f"Minimum thresholds passed: {minimum_passed}/4; "
            f"stage targets passed: {stage_passed}/4.",
            "No parameter or decision threshold was selected using these test results.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_path: str | Path,
    model_path: str | Path,
    report_dir: str | Path,
    *,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
):
    df = read_table(data_path)
    if LABEL_COL not in df.columns:
        raise KeyError(f"Evaluation data must contain label column: {LABEL_COL}")
    bundle = joblib.load(model_path)
    predictions = make_predictions(df, bundle)
    intervals = bootstrap_metric_intervals(
        predictions, n_bootstrap=n_bootstrap, seed=seed
    )
    metrics = probability_metrics(
        predictions["y_true"], predictions["ct_win_probability"], n_bins=10
    )
    calibration = calibration_table(
        predictions["y_true"], predictions["ct_win_probability"], n_bins=10
    )
    distribution = probability_distribution(predictions, n_bins=20)
    thresholds = threshold_metrics(
        predictions["y_true"], predictions["ct_win_probability"]
    )
    counts = confusion_counts(
        predictions["y_true"], predictions["ct_win_probability"]
    )
    fpr, tpr, roc_thresholds = roc_curve(
        predictions["y_true"], predictions["ct_win_probability"]
    )
    roc = pd.DataFrame(
        {"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": roc_thresholds}
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    intervals.to_csv(report_dir / "bootstrap_95ci.csv", index=False)
    calibration.to_csv(report_dir / "calibration_table.csv", index=False)
    distribution.to_csv(report_dir / "probability_distribution.csv", index=False)
    thresholds.to_csv(report_dir / "threshold_metrics.csv", index=False)
    roc.to_csv(report_dir / "roc_curve.csv", index=False)
    grouped_metrics(predictions, "map_name").to_csv(
        report_dir / "metrics_by_map.csv", index=False
    )
    grouped_metrics(predictions, "source_subset").to_csv(
        report_dir / "metrics_by_source.csv", index=False
    )
    pd.DataFrame(
        [
            {"actual_label": actual, "predicted_label": predicted, "count": int(count)}
            for actual, predicted, count in (
                (0, 0, counts["true_negative"]),
                (0, 1, counts["false_positive"]),
                (1, 0, counts["false_negative"]),
                (1, 1, counts["true_positive"]),
            )
        ]
    ).to_csv(report_dir / "confusion_matrix.csv", index=False)

    summary = {
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "model_path": str(model_path),
        "test_rounds": int(len(predictions)),
        "test_series": int(predictions["series_id"].nunique()),
        "bootstrap_unit": "series_id",
        "bootstrap_samples": n_bootstrap,
        "seed": seed,
        "metrics": metrics,
        "confusion_matrix_threshold_0_5": counts,
        "target_assessment": target_assessment(metrics),
    }
    with (report_dir / "m9_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_report(summary, intervals, report_dir / "m9_evaluation_report.md")
    save_plots(predictions, calibration, distribution, report_dir)
    return predictions, intervals, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M9 unified test-set evaluation.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-dir", default=str(REPORT_DIR / "m9"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    _, intervals, _ = run(
        args.data,
        args.model,
        args.report_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    print(intervals.round(6).to_string(index=False))


if __name__ == "__main__":
    main()

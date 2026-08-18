from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    SigmoidCalibrator,
    make_calibrator,
)
from .config import LABEL_COL, MODEL_DIR, REPORT_DIR
from .io import read_table
from .m9_evaluation import calibration_table, make_predictions
from .metrics import probability_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")


def group_fold_assignments(groups, *, n_splits: int = 5) -> np.ndarray:
    groups = np.asarray(groups).reshape(-1)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("n_splits cannot exceed the number of unique groups")
    assignments = np.full(len(groups), -1, dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(groups), 1))
    for fold, (_, holdout) in enumerate(splitter.split(dummy, groups=groups)):
        assignments[holdout] = fold
    if (assignments < 0).any():
        raise RuntimeError("Not every validation row received a fold assignment")
    return assignments


def cross_validated_comparison(
    validation_predictions: pd.DataFrame, *, n_splits: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = validation_predictions["y_true"].to_numpy(dtype=int)
    raw = validation_predictions["ct_win_probability"].to_numpy(dtype=float)
    folds = group_fold_assignments(
        validation_predictions["series_id"].astype(str), n_splits=n_splits
    )
    output = validation_predictions[
        ["series_id", "game_id", "round_id", "y_true"]
    ].copy()
    output["fold"] = folds

    rows = []
    for method in CALIBRATION_METHODS:
        calibrated = np.full(len(validation_predictions), np.nan, dtype=float)
        if method == "uncalibrated":
            calibrated[:] = raw
        else:
            for fold in range(n_splits):
                holdout = folds == fold
                calibrator = make_calibrator(method).fit(raw[~holdout], y[~holdout])
                calibrated[holdout] = calibrator.predict(raw[holdout])
        if not np.isfinite(calibrated).all():
            raise RuntimeError(f"OOF calibration produced invalid values: {method}")
        output[f"probability_{method}"] = calibrated
        rows.append({"method": method, **probability_metrics(y, calibrated, n_bins=10)})
    return pd.DataFrame(rows), output


def select_calibration_method(validation_comparison: pd.DataFrame) -> str:
    required = {"method", "log_loss", "brier_score"}
    missing = required - set(validation_comparison.columns)
    if missing:
        raise KeyError(f"Validation comparison is missing columns: {sorted(missing)}")
    ranked = validation_comparison.sort_values(
        ["log_loss", "brier_score", "method"], ascending=[True, True, True]
    )
    return str(ranked.iloc[0]["method"])


def fit_full_calibrators(validation_predictions: pd.DataFrame) -> dict:
    raw = validation_predictions["ct_win_probability"].to_numpy(dtype=float)
    y = validation_predictions["y_true"].to_numpy(dtype=int)
    return {
        method: make_calibrator(method).fit(raw, y)
        for method in CALIBRATION_METHODS
    }


def evaluate_test_calibrators(
    test_predictions: pd.DataFrame, calibrators: dict, selected_method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = test_predictions["y_true"].to_numpy(dtype=int)
    raw = test_predictions["ct_win_probability"].to_numpy(dtype=float)
    output = test_predictions[
        [
            "series_id",
            "game_id",
            "round_id",
            "map_name",
            "source_subset",
            "y_true",
        ]
    ].copy()
    rows = []
    raw_metrics = None
    for method in CALIBRATION_METHODS:
        probability = calibrators[method].predict(raw)
        output[f"probability_{method}"] = probability
        metrics = probability_metrics(y, probability, n_bins=10)
        if method == "uncalibrated":
            raw_metrics = metrics
        rows.append(
            {
                "method": method,
                "selected_by_validation": method == selected_method,
                **metrics,
            }
        )
    assert raw_metrics is not None
    comparison = pd.DataFrame(rows)
    for metric in ("log_loss", "brier_score", "ece10", "accuracy", "auc"):
        comparison[f"{metric}_change_vs_uncalibrated"] = (
            comparison[metric] - raw_metrics[metric]
        )
    return comparison, output


def calibration_curves(test_output: pd.DataFrame) -> pd.DataFrame:
    curves = []
    for method in CALIBRATION_METHODS:
        table = calibration_table(
            test_output["y_true"],
            test_output[f"probability_{method}"],
            n_bins=10,
        )
        table.insert(0, "method", method)
        curves.append(table)
    return pd.concat(curves, ignore_index=True)


def save_reliability_plot(
    curves: pd.DataFrame, selected_method: str, path: Path
) -> None:
    colors = {
        "uncalibrated": "#6B7280",
        "sigmoid": "#176B87",
        "isotonic": "#C44E52",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot(
        [0, 1],
        [0, 1],
        color="#111827",
        linestyle="--",
        linewidth=1,
        label="Perfect calibration",
    )
    for method in CALIBRATION_METHODS:
        part = curves[curves["method"].eq(method) & curves["count"].gt(0)]
        selected = method == selected_method
        label = f"{method} (selected)" if selected else method
        ax.plot(
            part["mean_probability"],
            part["observed_ct_win_rate"],
            color=colors[method],
            marker="o",
            linewidth=2.5 if selected else 1.5,
            alpha=1.0 if selected else 0.75,
            label=label,
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted CT probability",
        ylabel="Observed CT win rate",
        title="M10 test reliability comparison",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_summary(
    selected_method: str,
    validation_comparison: pd.DataFrame,
    test_comparison: pd.DataFrame,
    *,
    n_splits: int,
) -> dict:
    selected = test_comparison[test_comparison["method"].eq(selected_method)].iloc[0]
    raw = test_comparison[test_comparison["method"].eq("uncalibrated")].iloc[0]
    log_loss_change = float(selected["log_loss"] - raw["log_loss"])
    brier_change = float(selected["brier_score"] - raw["brier_score"])
    ece_change = float(selected["ece10"] - raw["ece10"])
    return {
        "task": "pre_round",
        "selection_data": "validation only",
        "selection_method": "grouped out-of-fold log_loss, then brier_score",
        "validation_group_column": "series_id",
        "validation_folds": n_splits,
        "selected_method": selected_method,
        "validation_oof_comparison": json.loads(
            validation_comparison.to_json(orient="records")
        ),
        "test_selected_metrics": {
            metric: float(selected[metric])
            for metric in ("accuracy", "auc", "log_loss", "brier_score", "ece10")
        },
        "test_changes_vs_uncalibrated": {
            "log_loss": log_loss_change,
            "brier_score": brier_change,
            "ece10": ece_change,
        },
        "test_ece_minimum_passed": bool(selected["ece10"] <= 0.04),
        "test_ece_stage_passed": bool(selected["ece10"] <= 0.03),
        "no_material_probability_metric_harm": bool(
            log_loss_change <= 0.002 and brier_change <= 0.001
        ),
    }


def write_report(
    validation_comparison: pd.DataFrame,
    test_comparison: pd.DataFrame,
    summary: dict,
    path: Path,
) -> None:
    lines = [
        "# M10 Probability Calibration Report",
        "",
        "Calibration method selection used grouped out-of-fold validation predictions only.",
        f"Selected method: **{summary['selected_method']}**.",
        "",
        "| Split | Method | Log Loss | Brier | ECE10 | AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, table in (
        ("validation_oof", validation_comparison),
        ("test", test_comparison),
    ):
        for _, row in table.iterrows():
            lines.append(
                f"| {split} | {row['method']} | {row['log_loss']:.6f} | "
                f"{row['brier_score']:.6f} | {row['ece10']:.6f} | "
                f"{row['auc']:.6f} |"
            )
    changes = summary["test_changes_vs_uncalibrated"]
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"Selected test ECE <= 0.04: {summary['test_ece_minimum_passed']}.",
            f"Selected test ECE <= 0.03: {summary['test_ece_stage_passed']}.",
            f"Test Log Loss change vs raw: {changes['log_loss']:+.6f}.",
            f"Test Brier change vs raw: {changes['brier_score']:+.6f}.",
            f"Test ECE change vs raw: {changes['ece10']:+.6f}.",
            "The test set was not used to select the calibration method.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_path: str | Path,
    base_model_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path,
    *,
    n_splits: int = 5,
):
    df = read_table(data_path)
    if LABEL_COL not in df.columns:
        raise KeyError(f"Calibration data must contain label column: {LABEL_COL}")
    base_bundle = joblib.load(base_model_path)
    validation = make_predictions(df, base_bundle, split="val")
    test = make_predictions(df, base_bundle, split="test")

    validation_comparison, validation_oof = cross_validated_comparison(
        validation, n_splits=n_splits
    )
    selected_method = select_calibration_method(validation_comparison)
    calibrators = fit_full_calibrators(validation)
    test_comparison, test_output = evaluate_test_calibrators(
        test, calibrators, selected_method
    )
    curves = calibration_curves(test_output)
    summary = build_summary(
        selected_method,
        validation_comparison,
        test_comparison,
        n_splits=n_splits,
    )

    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "calibrator": calibrators[selected_method],
            "method": selected_method,
            "base_model_path": str(base_model_path),
            "selection_data": "validation only",
            "validation_folds": n_splits,
        },
        model_dir / "pre_round_calibrator.joblib",
    )
    validation_comparison.to_csv(
        report_dir / "validation_oof_comparison.csv", index=False
    )
    validation_oof.to_csv(report_dir / "validation_oof_predictions.csv", index=False)
    test_comparison.to_csv(report_dir / "test_calibration_comparison.csv", index=False)
    test_output.to_csv(report_dir / "calibrated_test_predictions.csv", index=False)
    curves.to_csv(report_dir / "calibration_curves.csv", index=False)
    with (report_dir / "m10_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_report(
        validation_comparison,
        test_comparison,
        summary,
        report_dir / "m10_calibration_report.md",
    )
    save_reliability_plot(
        curves, selected_method, report_dir / "reliability_comparison.png"
    )
    return validation_comparison, test_comparison, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M10 probability calibration.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-dir", default=str(MODEL_DIR / "m10"))
    parser.add_argument("--report-dir", default=str(REPORT_DIR / "m10"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    validation, test, summary = run(
        args.data,
        args.base_model,
        args.model_dir,
        args.report_dir,
        n_splits=args.folds,
    )
    print("Validation OOF")
    print(validation.round(6).to_string(index=False))
    print("\nTest")
    print(test.round(6).to_string(index=False))
    print(f"\nSelected by validation: {summary['selected_method']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import LABEL_COL, MODEL_DIR, RANDOM_STATE, REPORT_DIR
from .io import read_table
from .metrics import probability_metrics
from .train_xgb import align_columns, make_model, prepare_xy


MODEL_NAMES = ("constant_train_prior", "logistic_regression", "xgboost_tuned")


def make_constant_model() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def make_logistic_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def evaluate_model(model, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    return probability_metrics(y, model.predict_proba(x)[:, 1], n_bins=10)


def prepare_splits(df: pd.DataFrame):
    if "split" not in df.columns:
        raise KeyError("M7 data must contain the fixed split column from make_dataset.")

    prepared = {}
    for split in ("train", "val", "test"):
        split_df = df[df["split"].eq(split)]
        if split_df.empty:
            raise ValueError(f"M7 data has no rows for split: {split}")
        prepared[split] = prepare_xy(split_df)

    reference = prepared["train"][0]
    for split in ("val", "test"):
        x, y = prepared[split]
        prepared[split] = (align_columns(reference, x), y)
    return prepared


def fit_models(prepared):
    x_train, y_train = prepared["train"]
    x_val, y_val = prepared["val"]

    constant = make_constant_model().fit(x_train, y_train)
    logistic = make_logistic_model().fit(x_train, y_train)
    xgboost = make_model(task="pre_round")
    xgboost.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return {
        "constant_train_prior": constant,
        "logistic_regression": logistic,
        "xgboost_tuned": xgboost,
    }


def compare_models(models, prepared) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_NAMES:
        model = models[model_name]
        for split in ("train", "val", "test"):
            x, y = prepared[split]
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    **evaluate_model(model, x, y),
                }
            )
    return pd.DataFrame(rows)


def build_summary(comparison: pd.DataFrame, prepared, xgboost) -> dict:
    test = comparison[comparison["split"].eq("test")].set_index("model")
    auc_margin = float(
        test.loc["xgboost_tuned", "auc"]
        - test.loc["logistic_regression", "auc"]
    )
    return {
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "feature_count": int(prepared["train"][0].shape[1]),
        "row_counts": {split: int(len(values[1])) for split, values in prepared.items()},
        "train_ct_win_rate": float(prepared["train"][1].mean()),
        "xgboost_best_iteration": getattr(xgboost, "best_iteration", None),
        "xgboost_minus_logistic_test_auc": auc_margin,
        "acceptance_auc_margin_at_least_0_01": bool(auc_margin >= 0.01),
    }


def write_report(comparison: pd.DataFrame, summary: dict, path: Path) -> None:
    test = comparison[comparison["split"].eq("test")].set_index("model")
    lines = [
        "# M7 Baseline Comparison Report",
        "",
        "All models use the same fixed 70/20/10 split, encoded feature columns,",
        "test rows, classification threshold (0.5), and probability metrics.",
        "",
        "| Model | Accuracy | AUC | Log Loss | Brier | ECE10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        row = test.loc[name]
        lines.append(
            f"| {name} | {row['accuracy']:.6f} | {row['auc']:.6f} | "
            f"{row['log_loss']:.6f} | {row['brier_score']:.6f} | "
            f"{row['ece10']:.6f} |"
        )
    passed = "PASS" if summary["acceptance_auc_margin_at_least_0_01"] else "NOT MET"
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"XGBoost minus logistic-regression test AUC: "
            f"{summary['xgboost_minus_logistic_test_auc']:.6f}.",
            f"Target (at least 0.01): **{passed}**.",
            "",
            "A NOT MET result is retained as an honest experimental result, as required",
            "by the M7 specification; it is not a pipeline failure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_path: str | Path, model_dir: str | Path, report_dir: str | Path):
    df = read_table(data_path)
    if LABEL_COL not in df.columns:
        raise KeyError(f"M7 data must contain label column: {LABEL_COL}")

    prepared = prepare_splits(df)
    models = fit_models(prepared)
    comparison = compare_models(models, prepared)
    summary = build_summary(comparison, prepared, models["xgboost_tuned"])

    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    columns = list(prepared["train"][0].columns)
    for model_name, model in models.items():
        joblib.dump(
            {"model": model, "columns": columns, "model_name": model_name},
            model_dir / f"pre_round_{model_name}.joblib",
        )
    comparison.to_csv(report_dir / "m7_model_comparison.csv", index=False)
    with (report_dir / "m7_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_report(comparison, summary, report_dir / "m7_baseline_report.md")
    return comparison, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", default=str(MODEL_DIR / "m7"))
    parser.add_argument("--report-dir", default=str(REPORT_DIR / "m7"))
    args = parser.parse_args()

    comparison, summary = run(args.data, args.model_dir, args.report_dir)
    print(comparison[comparison["split"].eq("test")].round(6).to_string(index=False))
    print(
        "XGBoost - logistic test AUC: "
        f"{summary['xgboost_minus_logistic_test_auc']:.6f}"
    )


if __name__ == "__main__":
    main()

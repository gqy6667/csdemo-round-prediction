from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from .config import LABEL_COL, MODEL_DIR, RANDOM_STATE, REPORT_DIR
from .io import read_table
from .metrics import probability_metrics
from .schema import ID_COLUMNS


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # IDs control joins and splits; they must never become predictive features.
    drop_cols = set(ID_COLUMNS) | {"match_id", "split", LABEL_COL}
    x = df.drop(columns=[c for c in drop_cols if c in df.columns])
    categorical = x.select_dtypes(exclude="number").columns
    if len(categorical):
        x[categorical] = x[categorical].astype("string").fillna("__MISSING__")
    return pd.get_dummies(x)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = prepare_features(df)
    y = df[LABEL_COL].astype(int)
    return x, y


def align_columns(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    return other.reindex(columns=reference.columns, fill_value=0)


def evaluate(model: XGBClassifier, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(x)[:, 1]
    return probability_metrics(y, proba, n_bins=10)


def make_model(task: str = "pre_round", **overrides) -> XGBClassifier:
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if task == "pre_round":
        params.update(
            {
                "n_estimators": 3000,
                "max_depth": 2,
                "min_child_weight": 3,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "early_stopping_rounds": 100,
            }
        )
    elif task != "first_kill":
        raise ValueError(f"Unknown XGBoost task: {task}")
    params.update(overrides)
    return XGBClassifier(**params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["pre_round", "first_kill"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    df = read_table(args.data)
    if "split" not in df.columns:
        raise KeyError("Training data must contain a split column from make_dataset.")

    train_df = df[df["split"].eq("train")]
    val_df = df[df["split"].eq("val")]
    test_df = df[df["split"].eq("test")]

    x_train, y_train = prepare_xy(train_df)
    x_val, y_val = prepare_xy(val_df)
    x_test, y_test = prepare_xy(test_df)
    x_val = align_columns(x_train, x_val)
    x_test = align_columns(x_train, x_test)

    model = make_model(task=args.task)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    metrics = {
        "train": evaluate(model, x_train, y_train),
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
    }

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best_iteration = getattr(model, "best_iteration", None)
    bundle = {
        "model": model,
        "columns": list(x_train.columns),
        "params": model.get_params(),
        "best_iteration": best_iteration,
    }
    joblib.dump(bundle, model_dir / f"{args.task}_xgb.joblib")
    pd.DataFrame(metrics).T.to_csv(report_dir / f"{args.task}_xgb_metrics.csv")

    training_summary = {
        "task": args.task,
        "best_iteration": best_iteration,
        "best_tree_count": best_iteration + 1 if best_iteration is not None else None,
        "params": model.get_params(),
    }
    with (report_dir / f"{args.task}_xgb_training_summary.json").open(
        "w", encoding="utf-8"
    ) as fh:
        json.dump(training_summary, fh, indent=2)

    evaluation_history = model.evals_result()
    if evaluation_history:
        history = next(iter(evaluation_history.values()))
        pd.DataFrame(history).to_csv(
            report_dir / f"{args.task}_xgb_training_history.csv", index_label="iteration"
        )

    print(pd.DataFrame(metrics).T.round(4))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from .config import MODEL_DIR, RANDOM_STATE, REPORT_DIR
from .io import read_table
from .metrics import probability_metrics
from .train_xgb import align_columns, prepare_xy


LIGHTGBM_BASE_PARAMS = {
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}
EARLY_STOPPING_ROUNDS = 100


def make_model(**overrides) -> LGBMClassifier:
    params = {
        **LIGHTGBM_BASE_PARAMS,
        "objective": "binary",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "device_type": "cpu",
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    params.update(overrides)
    return LGBMClassifier(**params)


def evaluate(model: LGBMClassifier, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probability = model.predict_proba(x)[:, 1]
    return probability_metrics(y, probability, n_bins=10)


def fit_with_validation(
    model: LGBMClassifier,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> LGBMClassifier:
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            log_evaluation(period=0),
        ],
    )
    return model


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

    model = make_model()
    fit_with_validation(model, x_train, y_train, x_val, y_val)

    metrics = {
        "train": evaluate(model, x_train, y_train),
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
    }

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "task": args.task,
        "model_name": "lightgbm_baseline",
        "columns": list(x_train.columns),
        "params": model.get_params(),
        "best_iteration": getattr(model, "best_iteration_", None),
    }
    joblib.dump(bundle, model_dir / f"{args.task}_lgbm.joblib")
    pd.DataFrame(metrics).T.to_csv(report_dir / f"{args.task}_lgbm_metrics.csv")

    history = getattr(model, "evals_result_", {})
    if history:
        validation = next(iter(history.values()))
        pd.DataFrame(validation).to_csv(
            report_dir / f"{args.task}_lgbm_training_history.csv",
            index_label="iteration",
        )
    summary = {
        "task": args.task,
        "best_iteration": getattr(model, "best_iteration_", None),
        "params": model.get_params(),
    }
    with (report_dir / f"{args.task}_lgbm_training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(pd.DataFrame(metrics).T.round(4))


if __name__ == "__main__":
    main()

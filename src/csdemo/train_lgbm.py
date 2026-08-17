from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .config import LABEL_COL, MODEL_DIR, RANDOM_STATE, REPORT_DIR
from .io import read_table
from .schema import ID_COLUMNS


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = set(ID_COLUMNS) | {"match_id", "split", LABEL_COL}
    x = df.drop(columns=[c for c in drop_cols if c in df.columns])
    x = pd.get_dummies(x, dummy_na=True)
    y = df[LABEL_COL].astype(int)
    return x, y


def align_columns(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    return other.reindex(columns=reference.columns, fill_value=0)


def evaluate(model: LGBMClassifier, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba),
    }
    if y.nunique() == 2:
        metrics["auc"] = roc_auc_score(y, proba)
    return metrics


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

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="binary_logloss")

    metrics = {
        "train": evaluate(model, x_train, y_train),
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
    }

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle = {"model": model, "columns": list(x_train.columns)}
    joblib.dump(bundle, model_dir / f"{args.task}_lgbm.joblib")
    pd.DataFrame(metrics).T.to_csv(report_dir / f"{args.task}_lgbm_metrics.csv")

    print(pd.DataFrame(metrics).T.round(4))


if __name__ == "__main__":
    main()

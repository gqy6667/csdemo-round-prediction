from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import LABEL_COL
from .io import read_table
from .schema import PRE_ROUND_FEATURES, PRE_ROUND_FEATURE_GROUPS
from .train_xgb import align_columns, evaluate, make_model, prepare_xy


ABLATION_GROUPS = ("score", "economy", "weapons")

ABLATION_VARIANTS = {
    "without_score": PRE_ROUND_FEATURE_GROUPS["score"],
    "without_economy": PRE_ROUND_FEATURE_GROUPS["economy"],
    "without_weapons": PRE_ROUND_FEATURE_GROUPS["weapons"],
    "without_map": ["map_name"],
    "without_cash": ["ct_cash", "t_cash", "cash_diff_ct"],
    "without_equipment_value": [
        "ct_eq_value",
        "t_eq_value",
        "eq_value_diff_ct",
    ],
    "without_armor_utility": PRE_ROUND_FEATURE_GROUPS["armor_utility"],
    "without_differences": [
        feature for feature in PRE_ROUND_FEATURES if feature.endswith("_diff_ct")
    ],
}


def drop_feature_group(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    if group_name not in PRE_ROUND_FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group: {group_name}")
    return df.drop(
        columns=[c for c in PRE_ROUND_FEATURE_GROUPS[group_name] if c in df.columns]
    )


def feature_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in PRE_ROUND_FEATURES:
        if feature not in df.columns:
            rows.append(
                {
                    "feature": feature,
                    "dtype": "missing",
                    "missing_count": len(df),
                    "nunique": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "nonzero_rate": None,
                }
            )
            continue

        series = df[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = numeric.notna().sum() == series.notna().sum()
        rows.append(
            {
                "feature": feature,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "nunique": int(series.nunique(dropna=False)),
                "min": numeric.min() if is_numeric else None,
                "max": numeric.max() if is_numeric else None,
                "mean": numeric.mean() if is_numeric else None,
                "nonzero_rate": float(numeric.ne(0).mean()) if is_numeric else None,
            }
        )
    return pd.DataFrame(rows)


def prepare_splits(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    train_df = df[df["split"].eq("train")]
    val_df = df[df["split"].eq("val")]
    test_df = df[df["split"].eq("test")]

    x_train, y_train = prepare_xy(train_df)
    x_val, y_val = prepare_xy(val_df)
    x_test, y_test = prepare_xy(test_df)
    return (
        x_train,
        y_train,
        align_columns(x_train, x_val),
        y_val,
        align_columns(x_train, x_test),
        y_test,
    )


def make_ablation_model():
    # Full row/column sampling isolates feature removal from sampling variance.
    return make_model(
        task="pre_round",
        n_estimators=500,
        max_depth=4,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=1.0,
        early_stopping_rounds=None,
    )


def train_variant(df: pd.DataFrame):
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_splits(df)
    model = make_ablation_model()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    metrics = {
        "train": evaluate(model, x_train, y_train),
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
    }
    return model, metrics, (x_train, y_train, x_val, y_val, x_test, y_test)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M6 feature profiling, importance, map metrics, and ablations."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    df = read_table(args.data)
    if "split" not in df.columns:
        raise KeyError("M6 data must contain the fixed train/val/test split.")
    if LABEL_COL not in df.columns:
        raise KeyError(f"M6 data must contain label column {LABEL_COL}.")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    feature_profile(df).to_csv(report_dir / "feature_profile.csv", index=False)

    variants = [("all_features", [])] + list(ABLATION_VARIANTS.items())
    metric_rows = []
    baseline_model = None
    baseline_split_data = None

    for variant_name, removed_features in variants:
        variant_df = df.drop(
            columns=[feature for feature in removed_features if feature in df.columns]
        )
        model, metrics, split_data = train_variant(variant_df)
        for split_name, values in metrics.items():
            metric_rows.append(
                {
                    "variant": variant_name,
                    "removed_features": ",".join(removed_features) or "none",
                    "split": split_name,
                    "feature_count_after_encoding": split_data[0].shape[1],
                    **values,
                }
            )
        if not removed_features:
            baseline_model = model
            baseline_split_data = split_data

    metrics_df = pd.DataFrame(metric_rows)
    test_auc = metrics_df[metrics_df["split"].eq("test")].set_index("variant")["auc"]
    metrics_df["test_auc_change_vs_all"] = metrics_df["variant"].map(
        test_auc - test_auc.loc["all_features"]
    )
    metrics_df.to_csv(report_dir / "ablation_metrics.csv", index=False)

    assert baseline_model is not None and baseline_split_data is not None
    x_train, _, _, _, x_test, y_test = baseline_split_data
    importance = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance": baseline_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(report_dir / "feature_importance.csv", index=False)

    test_df = df[df["split"].eq("test")]
    map_rows = []
    for map_name, map_df in test_df.groupby("map_name"):
        map_metrics = evaluate(
            baseline_model,
            x_test.loc[map_df.index],
            y_test.loc[map_df.index],
        )
        map_rows.append({"map_name": map_name, "rounds": len(map_df), **map_metrics})
    pd.DataFrame(map_rows).sort_values("rounds", ascending=False).to_csv(
        report_dir / "test_metrics_by_map.csv", index=False
    )

    print(metrics_df[metrics_df["split"].eq("test")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

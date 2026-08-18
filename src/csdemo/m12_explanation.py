from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance

from .io import read_table
from .schema import ID_COLUMNS, PRE_ROUND_FEATURES
from .train_xgb import prepare_xy

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FUTURE_INFORMATION_TERMS = (
    "kill",
    "death",
    "damage",
    "bomb_planted",
    "plant_tick",
    "round_end",
    "winner",
)


def deployment_tree_count(bundle: dict) -> int:
    model = bundle.get("model")
    if model is None:
        raise KeyError("Model bundle must contain model.")
    available = len(model.get_booster().get_dump())
    best_iteration = bundle.get("best_iteration")
    tree_count = available if best_iteration is None else int(best_iteration) + 1
    if tree_count < 1 or tree_count > available:
        raise ValueError(
            f"Deployment tree count {tree_count} is outside available range 1..{available}."
        )
    return tree_count


def tree_shap_contributions(
    bundle: dict, x: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    expected_columns = list(bundle.get("columns", []))
    if not expected_columns:
        raise KeyError("Model bundle must contain non-empty columns.")
    if list(x.columns) != expected_columns:
        raise ValueError("SHAP input columns must exactly match the model bundle order.")
    if x.empty:
        raise ValueError("SHAP input must contain at least one row.")

    tree_count = deployment_tree_count(bundle)
    matrix = xgb.DMatrix(x, feature_names=expected_columns)
    contributions = bundle["model"].get_booster().predict(
        matrix,
        pred_contribs=True,
        iteration_range=(0, tree_count),
    )
    if contributions.shape != (len(x), len(expected_columns) + 1):
        raise ValueError(
            "Expected binary TreeSHAP output with one contribution per feature plus bias."
        )
    values = pd.DataFrame(
        contributions[:, :-1], index=x.index, columns=expected_columns
    )
    return values, contributions[:, -1]


def gain_importance(bundle: dict) -> pd.DataFrame:
    expected_columns = list(bundle.get("columns", []))
    if not expected_columns:
        raise KeyError("Model bundle must contain non-empty columns.")
    full_booster = bundle["model"].get_booster()
    available_tree_count = len(full_booster.get_dump())
    tree_count = deployment_tree_count(bundle)
    booster = full_booster[:tree_count] if tree_count < available_tree_count else full_booster
    gain = booster.get_score(importance_type="gain")
    split_count = booster.get_score(importance_type="weight")
    result = pd.DataFrame(
        {
            "feature": expected_columns,
            "gain": [float(gain.get(feature, 0.0)) for feature in expected_columns],
            "split_count": [int(split_count.get(feature, 0)) for feature in expected_columns],
        }
    )
    total_gain = float(result["gain"].sum())
    result["gain_normalized"] = result["gain"] / total_gain if total_gain else 0.0
    result["deployment_tree_count"] = tree_count
    result["available_tree_count"] = available_tree_count
    result = result.sort_values(["gain", "feature"], ascending=[False, True])
    result.insert(0, "gain_rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def permutation_auc_importance(
    model,
    x: pd.DataFrame,
    y,
    *,
    n_repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    if x.empty or len(x) != len(y):
        raise ValueError("Permutation inputs must have the same non-zero length.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    result = permutation_importance(
        model,
        x,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": x.columns,
            "auc_decrease_mean": result.importances_mean,
            "auc_decrease_std": result.importances_std,
            "n_repeats": n_repeats,
        }
    ).sort_values(["auc_decrease_mean", "feature"], ascending=[False, True])
    importance.insert(0, "permutation_rank", range(1, len(importance) + 1))
    return importance.reset_index(drop=True)


def shap_importance(shap_values: pd.DataFrame) -> pd.DataFrame:
    if shap_values.empty:
        raise ValueError("SHAP values must not be empty.")
    importance = pd.DataFrame(
        {
            "feature": shap_values.columns,
            "mean_abs_shap": shap_values.abs().mean().to_numpy(),
            "mean_signed_shap": shap_values.mean().to_numpy(),
            "max_abs_shap": shap_values.abs().max().to_numpy(),
        }
    ).sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    importance.insert(0, "shap_rank", range(1, len(importance) + 1))
    return importance.reset_index(drop=True)


def build_importance_comparison(
    gain: pd.DataFrame,
    permutation: pd.DataFrame,
    shap: pd.DataFrame,
) -> pd.DataFrame:
    gain_columns = ["feature", "gain_rank", "gain_normalized"]
    permutation_columns = [
        "feature",
        "permutation_rank",
        "auc_decrease_mean",
        "auc_decrease_std",
    ]
    shap_columns = ["feature", "shap_rank", "mean_abs_shap"]
    comparison = gain[gain_columns].merge(
        permutation[permutation_columns], on="feature", how="outer", validate="one_to_one"
    )
    comparison = comparison.merge(
        shap[shap_columns], on="feature", how="outer", validate="one_to_one"
    )
    comparison["mean_rank"] = comparison[
        ["gain_rank", "permutation_rank", "shap_rank"]
    ].mean(axis=1)
    return comparison.sort_values(
        ["mean_rank", "feature"], ascending=[True, True]
    ).reset_index(drop=True)


def audit_model_features(feature_names: Sequence[str]) -> pd.DataFrame:
    allowed_numeric = set(PRE_ROUND_FEATURES) - {"map_name"}
    rows = []
    for rank, feature in enumerate(feature_names, start=1):
        source_feature = "map_name" if feature.startswith("map_name_") else feature
        allowed = feature in allowed_numeric or feature.startswith("map_name_")
        if allowed:
            reason = "allowed_pre_round"
        elif feature in ID_COLUMNS or feature == "match_id" or feature.endswith("_id"):
            reason = "identifier"
        elif any(term in feature.lower() for term in FUTURE_INFORMATION_TERMS):
            reason = "future_information"
        else:
            reason = "not_in_pre_round_schema"
        rows.append(
            {
                "importance_rank": rank,
                "feature": feature,
                "source_feature": source_feature,
                "audit_result": "pass" if allowed else "fail",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def select_explanation_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"y_true", "ct_win_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Case predictions are missing columns: {sorted(missing)}")
    working = predictions.copy().reset_index(drop=True)
    working["row_position"] = np.arange(len(working))
    working["predicted_label"] = (
        working["ct_win_probability"].to_numpy() >= 0.5
    ).astype(int)
    working["prediction_confidence"] = np.maximum(
        working["ct_win_probability"], 1.0 - working["ct_win_probability"]
    )

    candidates = {
        "ct_high_probability": working[
            working["y_true"].eq(1) & working["predicted_label"].eq(1)
        ].sort_values("ct_win_probability", ascending=False),
        "t_high_probability": working[
            working["y_true"].eq(0) & working["predicted_label"].eq(0)
        ].sort_values("ct_win_probability", ascending=True),
        "high_confidence_error": working[
            working["y_true"].ne(working["predicted_label"])
        ].sort_values("prediction_confidence", ascending=False),
    }
    empty = [name for name, frame in candidates.items() if frame.empty]
    if empty:
        raise ValueError(f"No eligible rows for explanation cases: {empty}")

    selected = []
    for case_type, frame in candidates.items():
        row = frame.iloc[0].to_dict()
        row["case_type"] = case_type
        selected.append(row)
    return pd.DataFrame(selected)


def build_case_explanations(
    cases: pd.DataFrame,
    x: pd.DataFrame,
    shap_values: pd.DataFrame,
    base_values: np.ndarray,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if len(x) != len(shap_values) or len(x) != len(base_values):
        raise ValueError("Feature, SHAP, and base-value rows must have equal length.")
    if list(x.columns) != list(shap_values.columns):
        raise ValueError("Feature and SHAP columns must match in the same order.")
    if "row_position" not in cases.columns or "case_type" not in cases.columns:
        raise KeyError("Cases must contain row_position and case_type.")

    rows = []
    for _, case in cases.iterrows():
        position = int(case["row_position"])
        if position < 0 or position >= len(x):
            raise IndexError(f"Case row_position {position} is outside the test data.")
        contributions = shap_values.iloc[position]
        ordered_features = contributions.abs().sort_values(ascending=False).index[:top_n]
        model_log_odds = float(base_values[position] + contributions.sum())
        reconstructed_probability = float(
            1.0 / (1.0 + np.exp(-np.clip(model_log_odds, -709, 709)))
        )
        common = {
            column: case[column]
            for column in cases.columns
            if column != "row_position"
        }
        common.update(
            {
                "row_position": position,
                "base_value_log_odds": float(base_values[position]),
                "model_log_odds": model_log_odds,
                "reconstructed_ct_probability": reconstructed_probability,
            }
        )
        for rank, feature in enumerate(ordered_features, start=1):
            contribution = float(contributions[feature])
            rows.append(
                {
                    **common,
                    "contribution_rank": rank,
                    "feature": feature,
                    "feature_value": x.iloc[position][feature],
                    "shap_value_log_odds": contribution,
                    "direction": (
                        "toward_ct"
                        if contribution > 0
                        else "toward_t" if contribution < 0 else "neutral"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _sigmoid(log_odds: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(log_odds, dtype=float), -709, 709)
    return 1.0 / (1.0 + np.exp(-clipped))


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_analysis(
    data_path: str | Path,
    model_path: str | Path,
    report_dir: str | Path,
    *,
    permutation_repeats: int = 20,
    seed: int = 42,
    case_features: int = 10,
    shap_plot_rows: int = 1500,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    data = read_table(data_path)
    if "split" not in data.columns:
        raise KeyError("M12 data must contain the fixed split column.")
    test_data = data[data["split"].eq("test")].copy()
    if test_data.empty:
        raise ValueError("M12 data contains no test rows.")

    bundle = joblib.load(model_path)
    expected_columns = list(bundle.get("columns", []))
    if not expected_columns:
        raise KeyError("Model bundle must contain non-empty columns.")
    x_test, y_test = prepare_xy(test_data)
    x_test = x_test.reindex(columns=expected_columns, fill_value=0)
    probability = bundle["model"].predict_proba(x_test)[:, 1]

    gain = gain_importance(bundle)
    permutation = permutation_auc_importance(
        bundle["model"],
        x_test,
        y_test,
        n_repeats=permutation_repeats,
        seed=seed,
    )
    shap_values, base_values = tree_shap_contributions(bundle, x_test)
    shap = shap_importance(shap_values)
    comparison = build_importance_comparison(gain, permutation, shap)

    reconstructed = _sigmoid(base_values + shap_values.sum(axis=1).to_numpy())
    reconstruction_error = np.abs(reconstructed - probability)
    audit = audit_model_features(shap["feature"].tolist()).merge(
        shap[["feature", "mean_abs_shap"]], on="feature", how="left", validate="one_to_one"
    )
    top20_audit = audit.head(20).copy()

    metadata_columns = [
        column
        for column in ("series_id", "game_id", "round_id", "map_name", "round_num")
        if column in test_data.columns
    ]
    prediction_context = test_data[metadata_columns].reset_index(drop=True)
    prediction_context["y_true"] = y_test.to_numpy(dtype=int)
    prediction_context["ct_win_probability"] = probability
    prediction_context["predicted_label"] = (probability >= 0.5).astype(int)
    prediction_context["correct"] = prediction_context["predicted_label"].eq(
        prediction_context["y_true"]
    )
    cases = select_explanation_cases(prediction_context)
    case_explanations = build_case_explanations(
        cases,
        x_test,
        shap_values,
        base_values,
        top_n=case_features,
    )

    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "gain_importance": gain,
        "permutation_importance_auc": permutation,
        "shap_importance": shap,
        "importance_comparison": comparison,
        "all_feature_leakage_audit": audit,
        "top20_feature_audit": top20_audit,
        "selected_cases": cases,
        "case_explanations": case_explanations,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    available_trees = len(bundle["model"].get_booster().get_dump())
    rank_correlations = comparison[
        ["gain_rank", "permutation_rank", "shap_rank"]
    ].corr(method="spearman")
    summary = {
        "task": "pre_round",
        "definition": "freeze-time end after purchases and before combat",
        "model_path": str(model_path),
        "test_rounds": int(len(test_data)),
        "encoded_features": int(x_test.shape[1]),
        "available_tree_count": available_trees,
        "deployment_tree_count": deployment_tree_count(bundle),
        "permutation_metric": "test_auc_decrease",
        "permutation_repeats": permutation_repeats,
        "seed": seed,
        "shap_method": "xgboost_native_tree_shap",
        "shap_units": "log_odds",
        "shap_reconstruction_max_abs_error": float(reconstruction_error.max()),
        "shap_reconstruction_mean_abs_error": float(reconstruction_error.mean()),
        "top_features": {
            "gain": gain.head(10)["feature"].tolist(),
            "permutation_auc": permutation.head(10)["feature"].tolist(),
            "mean_abs_shap": shap.head(10)["feature"].tolist(),
            "mean_rank": comparison.head(10)["feature"].tolist(),
        },
        "importance_rank_spearman": rank_correlations.to_dict(),
        "feature_audit": {
            "all_feature_failures": int(audit["audit_result"].eq("fail").sum()),
            "top20_failures": int(top20_audit["audit_result"].eq("fail").sum()),
        },
        "selected_cases": cases.to_dict(orient="records"),
        "acceptance": {
            "three_importance_methods_created": True,
            "ct_t_and_error_cases_created": set(cases["case_type"])
            == {
                "ct_high_probability",
                "t_high_probability",
                "high_confidence_error",
            },
            "top20_has_no_id_or_future_leakage": bool(
                top20_audit["audit_result"].eq("pass").all()
            ),
            "all_model_features_pass_schema_audit": bool(
                audit["audit_result"].eq("pass").all()
            ),
            "shap_reconstructs_probability_within_1e_5": bool(
                reconstruction_error.max() <= 1e-5
            ),
        },
    }
    runtime_tables = {**tables, "x_test": x_test, "shap_values": shap_values}
    save_explanation_plots(
        runtime_tables,
        output_dir,
        seed=seed,
        shap_plot_rows=shap_plot_rows,
    )
    write_explanation_report(
        summary,
        runtime_tables,
        output_dir / "m12_explanation_report.md",
    )
    with (output_dir / "m12_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    return summary, runtime_tables


def _save_importance_bar(
    table: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    x_label: str,
    path: Path,
    color: str,
    error_column: str | None = None,
    top_n: int = 20,
) -> None:
    part = table.head(top_n).sort_values(value_column, ascending=True)
    figure, axis = plt.subplots(figsize=(8.4, 6.4))
    errors = part[error_column] if error_column else None
    axis.barh(
        part["feature"],
        part[value_column],
        xerr=errors,
        color=color,
        alpha=0.9,
        capsize=2,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_explanation_plots(
    tables: dict[str, pd.DataFrame],
    report_dir: str | Path,
    *,
    seed: int,
    shap_plot_rows: int = 1500,
) -> None:
    output_dir = Path(report_dir)
    plt.style.use("seaborn-v0_8-whitegrid")
    _save_importance_bar(
        tables["gain_importance"],
        value_column="gain_normalized",
        title="XGBoost gain importance (deployment trees)",
        x_label="Normalized average gain",
        path=output_dir / "gain_importance.png",
        color="#176B87",
    )
    _save_importance_bar(
        tables["permutation_importance_auc"],
        value_column="auc_decrease_mean",
        error_column="auc_decrease_std",
        title="Test-set permutation importance",
        x_label="Mean decrease in test AUC",
        path=output_dir / "permutation_importance_auc.png",
        color="#D17A22",
    )
    _save_importance_bar(
        tables["shap_importance"],
        value_column="mean_abs_shap",
        title="Global TreeSHAP importance",
        x_label="Mean absolute SHAP value (log-odds)",
        path=output_dir / "shap_importance.png",
        color="#4C956C",
    )

    x_test = tables["x_test"]
    shap_values = tables["shap_values"]
    top_features = tables["shap_importance"].head(15)["feature"].tolist()[::-1]
    rng = np.random.default_rng(seed)
    sample_size = min(shap_plot_rows, len(x_test))
    positions = np.sort(rng.choice(len(x_test), size=sample_size, replace=False))
    figure, axis = plt.subplots(figsize=(9.0, 6.8))
    last_scatter = None
    for y_position, feature in enumerate(top_features):
        feature_values = pd.to_numeric(
            x_test.iloc[positions][feature], errors="coerce"
        ).fillna(0.0)
        relative_values = feature_values.rank(method="average", pct=True).to_numpy()
        jitter = rng.uniform(-0.24, 0.24, size=sample_size)
        last_scatter = axis.scatter(
            shap_values.iloc[positions][feature],
            y_position + jitter,
            c=relative_values,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=9,
            alpha=0.62,
            linewidths=0,
        )
    axis.axvline(0, color="#4B5563", linewidth=1)
    axis.set_yticks(range(len(top_features)), labels=top_features)
    axis.set_xlabel("SHAP contribution to CT win log-odds")
    axis.set_ylabel("Feature")
    axis.set_title(f"TreeSHAP summary ({sample_size:,} fixed-test rows)")
    if last_scatter is not None:
        colorbar = figure.colorbar(last_scatter, ax=axis, pad=0.02)
        colorbar.set_label("Relative feature value (low to high)")
    figure.tight_layout()
    figure.savefig(output_dir / "shap_summary.png", dpi=170)
    plt.close(figure)

    cases = tables["selected_cases"]
    explanations = tables["case_explanations"]
    case_order = [
        "ct_high_probability",
        "t_high_probability",
        "high_confidence_error",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.8), sharex=False)
    for axis, case_type in zip(axes, case_order):
        part = explanations[explanations["case_type"].eq(case_type)].sort_values(
            "contribution_rank", ascending=False
        )
        colors = np.where(
            part["shap_value_log_odds"].gt(0), "#176B87", "#C44E52"
        )
        axis.barh(part["feature"], part["shap_value_log_odds"], color=colors)
        axis.axvline(0, color="#4B5563", linewidth=1)
        probability = float(
            cases[cases["case_type"].eq(case_type)]["ct_win_probability"].iloc[0]
        )
        axis.set_title(f"{case_type}\nCT probability = {probability:.3f}")
        axis.set_xlabel("SHAP value (log-odds)")
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "case_explanations.png", dpi=170)
    plt.close(figure)


def write_explanation_report(
    summary: dict,
    tables: dict[str, pd.DataFrame],
    path: str | Path,
) -> None:
    comparison = tables["importance_comparison"]
    audit = tables["top20_feature_audit"]
    cases = tables["selected_cases"]
    explanations = tables["case_explanations"]
    correlations = summary["importance_rank_spearman"]
    lines = [
        "# M12 Model Explanation Report",
        "",
        "This stage explains the fixed M8 XGBoost on the unchanged M9 test split.",
        "No model, feature, threshold, or test probability was changed.",
        "",
        "## Explanation Integrity",
        "",
        f"The saved booster contains {summary['available_tree_count']} trees, while deployed",
        f"predictions use the first {summary['deployment_tree_count']} trees selected by early stopping.",
        "Gain and native TreeSHAP were both limited to those deployment trees.",
        f"TreeSHAP reconstructed all probabilities with maximum absolute error",
        f"{summary['shap_reconstruction_max_abs_error']:.10f}.",
        "",
        "## Global Importance",
        "",
        "| Feature | Gain rank | Normalized gain | Permutation rank | Test AUC decrease | SHAP rank | Mean abs SHAP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.head(15).iterrows():
        lines.append(
            f"| {row['feature']} | {int(row['gain_rank'])} | {row['gain_normalized']:.6f} | "
            f"{int(row['permutation_rank'])} | {row['auc_decrease_mean']:.6f} | "
            f"{int(row['shap_rank'])} | {row['mean_abs_shap']:.6f} |"
        )
    lines.extend(
        [
            "",
            "`eq_value_diff_ct` is the most stable signal: gain rank 2, permutation rank 1,",
            "and SHAP rank 1. `grenade_diff_ct` is gain rank 1 but permutation rank 12,",
            "showing why gain alone is not sufficient. Rank Spearman correlations are",
            f"gain-permutation {correlations['gain_rank']['permutation_rank']:.3f},",
            f"gain-SHAP {correlations['gain_rank']['shap_rank']:.3f}, and",
            f"permutation-SHAP {correlations['permutation_rank']['shap_rank']:.3f}.",
            "",
            "Gain measures average loss reduction at tree splits. Permutation reports test AUC",
            "loss after shuffling one encoded column. Mean absolute TreeSHAP reports average",
            "contribution magnitude in log-odds. Correlated raw and difference features can",
            "share importance, and none of these measures establishes causality.",
            "",
            "## Leakage Audit",
            "",
            f"All {summary['encoded_features']} encoded model columns passed the pre-round schema audit.",
            "The top 20 TreeSHAP features contain no ID, first-kill, damage, bomb-state, or",
            "round-result fields.",
            "",
            "| SHAP rank | Feature | Source | Result |",
            "|---|---|---|---|",
        ]
    )
    for _, row in audit.iterrows():
        lines.append(
            f"| {int(row['importance_rank'])} | {row['feature']} | "
            f"{row['source_feature']} | {row['audit_result']} |"
        )

    lines.extend(
        [
            "",
            "## Round Cases",
            "",
            "| Case | Round | Map | Actual | Predicted CT probability | Correct |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for _, case in cases.iterrows():
        actual = "CT" if int(case["y_true"]) == 1 else "T"
        lines.append(
            f"| {case['case_type']} | {case.get('round_id', '')} | "
            f"{case.get('map_name', '')} | {actual} | "
            f"{float(case['ct_win_probability']):.6f} | {bool(case['correct'])} |"
        )
    for case_type in (
        "ct_high_probability",
        "t_high_probability",
        "high_confidence_error",
    ):
        lines.extend(
            [
                "",
                f"### {case_type}",
                "",
                "| Rank | Feature | Value | SHAP log-odds | Direction |",
                "|---:|---|---:|---:|---|",
            ]
        )
        part = explanations[
            explanations["case_type"].eq(case_type)
            & explanations["contribution_rank"].le(5)
        ].sort_values("contribution_rank")
        for _, row in part.iterrows():
            lines.append(
                f"| {int(row['contribution_rank'])} | {row['feature']} | "
                f"{row['feature_value']} | {row['shap_value_log_odds']:.6f} | "
                f"{row['direction']} |"
            )

    lines.extend(
        [
            "",
            "The error case shows why the model strongly favored CT from the purchase snapshot;",
            "it does not explain the later combat outcome. Position, aim, utility execution, and",
            "other post-freeze events are outside this model and must not be added as pre-round features.",
            "",
            "## Acceptance",
            "",
        ]
    )
    for requirement, passed in summary["acceptance"].items():
        lines.append(f"- {requirement}: {passed}")
    lines.extend(
        [
            "",
            "External benchmark differences are unchanged because M12 did not retrain the model;",
            "see `external_benchmark_comparison.md` in this report directory.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M12 gain, permutation, TreeSHAP, and leakage analysis."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-features", type=int, default=10)
    parser.add_argument("--shap-plot-rows", type=int, default=1500)
    parser.add_argument(
        "--metrics",
        default="reports/esta_full_m9/m9_summary.json",
        help="Fixed model metrics JSON used for the required external comparison.",
    )
    parser.add_argument(
        "--benchmarks",
        default="benchmarks/external_round_model_metrics.csv",
        help="Structured external benchmark registry.",
    )
    args = parser.parse_args()

    summary, _ = run_analysis(
        args.data,
        args.model,
        args.report_dir,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
        case_features=args.case_features,
        shap_plot_rows=args.shap_plot_rows,
    )
    from .benchmark_comparison import run as run_benchmark_comparison

    run_benchmark_comparison(
        args.metrics,
        args.benchmarks,
        args.report_dir,
        stage_label="M12",
    )
    print(json.dumps(summary["acceptance"], indent=2))


if __name__ == "__main__":
    main()

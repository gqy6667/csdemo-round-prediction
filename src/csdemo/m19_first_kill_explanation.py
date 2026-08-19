from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .io import read_table
from .m12_explanation import (
    build_case_explanations,
    build_importance_comparison,
    deployment_tree_count,
    gain_importance,
    permutation_auc_importance,
    save_explanation_plots,
    select_explanation_cases,
    shap_importance,
    tree_shap_contributions,
)
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m16_first_kill_baselines import (
    FIRST_KILL_MODEL_FEATURES,
    REPORT_METRICS,
    audit_training_data,
    canonical_feature_names,
    compare_external_models,
    prepare_profile_splits,
    write_json,
)
from .m18_first_kill_evaluation import render_external_report
from .metrics import probability_metrics
from .schema import ID_COLUMNS, PRE_ROUND_FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIRST_KILL_EVENT_FEATURES = tuple(FIRST_KILL_MODEL_FEATURES)
CATEGORICAL_FEATURES = ("map_name", "first_kill_weapon")
HIGHER_IS_BETTER = {"accuracy", "auc"}

TARGET_DEFINITIONS = (
    ("test_accuracy", "Test Accuracy", "higher", 0.700),
    ("test_auc", "Test AUC", "higher", 0.780),
    ("test_log_loss", "Test Log Loss", "lower", 0.550),
    ("test_brier", "Test Brier", "lower", 0.185),
    ("test_ece10", "Test ECE10", "lower", 0.030),
    ("auc_ci_lower", "AUC 95% CI lower", "higher", 0.790),
    ("log_loss_ci_upper", "Log Loss 95% CI upper", "lower", 0.540),
    ("source_auc_gap", "LAN-online absolute AUC gap", "lower", 0.040),
    ("large_map_min_auc", "Large-map minimum AUC", "higher", 0.770),
    (
        "large_map_min_auc_ci_lower",
        "Large-map minimum AUC CI lower",
        "higher",
        0.700,
    ),
)

BLOCKING_CHECKS = (
    "m18_prerequisite",
    "model_replay",
    "importance_methods",
    "feature_mapping_and_leakage",
    "shap_reconstruction",
    "case_explanations",
    "target_and_internal_gaps",
    "external_report",
    "automated_tests",
)


def map_encoded_feature_to_source(
    encoded_feature: str,
    raw_features: Sequence[str],
) -> str:
    raw = list(raw_features)
    if encoded_feature in raw:
        return encoded_feature
    matches = [
        feature
        for feature in CATEGORICAL_FEATURES
        if feature in raw and encoded_feature.startswith(f"{feature}_")
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Encoded feature {encoded_feature!r} does not map to one raw feature"
    )


def _failed_feature_reason(feature: str) -> str:
    lower = feature.lower()
    if feature in ID_COLUMNS or feature == "match_id" or feature.endswith("_id"):
        return "identifier"
    if feature in {"ct_win", "y_true", "label", "split"}:
        return "label_or_split"
    if any(
        term in lower
        for term in (
            "winner",
            "round_end",
            "second_kill",
            "damage",
            "health",
            "bomb_planted",
            "plant_tick",
        )
    ):
        return "future_information"
    return "not_in_post_first_kill_contract"


def audit_post_first_kill_features(
    encoded_features: Sequence[str],
    raw_features: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for rank, encoded in enumerate(encoded_features, start=1):
        try:
            source = map_encoded_feature_to_source(encoded, raw_features)
            allowed = True
            reason = (
                "allowed_first_kill_event"
                if source in FIRST_KILL_EVENT_FEATURES
                else "allowed_purchase_end"
            )
            availability = (
                "first_valid_enemy_kill"
                if source in FIRST_KILL_EVENT_FEATURES
                else "purchase_end_pre_combat"
            )
        except ValueError:
            source = None
            allowed = False
            reason = _failed_feature_reason(encoded)
            availability = "forbidden_or_unknown"
        rows.append(
            {
                "importance_rank": rank,
                "encoded_feature": encoded,
                "source_feature": source,
                "feature_group": (
                    "first_kill_event"
                    if source in FIRST_KILL_EVENT_FEATURES
                    else "purchase_end" if source is not None else "forbidden"
                ),
                "availability": availability,
                "audit_result": "pass" if allowed else "fail",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_source_feature_groups(
    encoded_features: Sequence[str],
    raw_features: Sequence[str],
) -> dict[str, list[str]]:
    groups = {feature: [] for feature in raw_features}
    for encoded in encoded_features:
        source = map_encoded_feature_to_source(encoded, raw_features)
        groups[source].append(encoded)
    missing = [feature for feature, columns in groups.items() if not columns]
    if missing:
        raise ValueError(f"Raw features have no encoded columns: {missing}")
    flattened = [column for columns in groups.values() for column in columns]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(encoded_features):
        raise ValueError("Encoded columns must map to exactly one source feature")
    return groups


def build_macro_feature_groups(
    encoded_features: Sequence[str],
    raw_features: Sequence[str],
) -> dict[str, list[str]]:
    source_groups = build_source_feature_groups(encoded_features, raw_features)
    return {
        "purchase_end": [
            column
            for feature in raw_features
            if feature not in FIRST_KILL_EVENT_FEATURES
            for column in source_groups[feature]
        ],
        "first_kill_event": [
            column
            for feature in raw_features
            if feature in FIRST_KILL_EVENT_FEATURES
            for column in source_groups[feature]
        ],
    }


def grouped_permutation_auc_importance(
    model: Any,
    x: pd.DataFrame,
    y: Sequence[int],
    feature_groups: dict[str, list[str]],
    *,
    n_repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    if x.empty or len(x) != len(y):
        raise ValueError("Grouped permutation inputs must have equal non-zero length")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    flattened = [column for columns in feature_groups.values() for column in columns]
    unknown = sorted(set(flattened) - set(x.columns))
    duplicates = len(flattened) - len(set(flattened))
    if unknown or duplicates:
        raise ValueError(
            f"Invalid grouped permutation columns: unknown={unknown}, duplicates={duplicates}"
        )
    if any(not columns for columns in feature_groups.values()):
        raise ValueError("Every permutation group must contain at least one column")

    labels = np.asarray(y, dtype=int)
    baseline_probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    baseline_auc = float(roc_auc_score(labels, baseline_probability))
    rng = np.random.default_rng(seed)
    rows = []
    for group_name, columns in feature_groups.items():
        decreases = []
        for _ in range(n_repeats):
            order = rng.permutation(len(x))
            permuted = x.copy()
            permuted.loc[:, columns] = x.iloc[order][columns].to_numpy()
            probability = np.asarray(
                model.predict_proba(permuted)[:, 1], dtype=float
            )
            decreases.append(baseline_auc - roc_auc_score(labels, probability))
        values = np.asarray(decreases, dtype=float)
        rows.append(
            {
                "feature_group": group_name,
                "encoded_column_count": len(columns),
                "baseline_auc": baseline_auc,
                "auc_decrease_mean": float(values.mean()),
                "auc_decrease_std": float(values.std(ddof=0)),
                "auc_decrease_min": float(values.min()),
                "auc_decrease_max": float(values.max()),
                "n_repeats": n_repeats,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["auc_decrease_mean", "feature_group"], ascending=[False, True]
    )
    result.insert(0, "grouped_permutation_rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def build_source_importance_summary(
    gain: pd.DataFrame,
    shap: pd.DataFrame,
    grouped_permutation: pd.DataFrame,
    encoded_contract: pd.DataFrame,
) -> pd.DataFrame:
    mapping = encoded_contract.loc[
        encoded_contract["audit_result"].eq("pass"),
        ["encoded_feature", "source_feature", "feature_group"],
    ]
    if mapping["encoded_feature"].duplicated().any():
        raise ValueError("Encoded feature contract must map each column once")
    gain_mapped = gain.merge(
        mapping,
        left_on="feature",
        right_on="encoded_feature",
        how="left",
        validate="one_to_one",
    )
    shap_mapped = shap.merge(
        mapping[["encoded_feature", "source_feature"]],
        left_on="feature",
        right_on="encoded_feature",
        how="left",
        validate="one_to_one",
    )
    if gain_mapped["source_feature"].isna().any() or shap_mapped[
        "source_feature"
    ].isna().any():
        raise ValueError("Every Gain and SHAP column must have a source feature")

    gain_grouped = (
        gain_mapped.groupby(["source_feature", "feature_group"], as_index=False)
        .agg(
            encoded_column_count=("encoded_feature", "size"),
            gain_normalized=("gain_normalized", "sum"),
            split_count=("split_count", "sum"),
        )
    )
    shap_grouped = (
        shap_mapped.groupby("source_feature", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"))
    )
    permutation = grouped_permutation.rename(
        columns={
            "feature_group": "source_feature",
            "auc_decrease_mean": "grouped_auc_decrease_mean",
            "auc_decrease_std": "grouped_auc_decrease_std",
        }
    )[
        [
            "source_feature",
            "grouped_auc_decrease_mean",
            "grouped_auc_decrease_std",
        ]
    ]
    result = gain_grouped.merge(
        shap_grouped, on="source_feature", how="inner", validate="one_to_one"
    ).merge(permutation, on="source_feature", how="inner", validate="one_to_one")
    for value_column, rank_column in (
        ("gain_normalized", "source_gain_rank"),
        ("mean_abs_shap", "source_shap_rank"),
        ("grouped_auc_decrease_mean", "source_permutation_rank"),
    ):
        result[rank_column] = result[value_column].rank(
            method="first", ascending=False
        ).astype(int)
    result["mean_rank"] = result[
        ["source_gain_rank", "source_shap_rank", "source_permutation_rank"]
    ].mean(axis=1)
    return result.sort_values(
        ["mean_rank", "source_feature"], ascending=[True, True]
    ).reset_index(drop=True)


def _target_row(
    target_id: str,
    label: str,
    direction: str,
    current: float,
    target: float,
) -> dict[str, Any]:
    if direction == "higher":
        remaining = max(target - current, 0.0)
        margin = max(current - target, 0.0)
        passed = current >= target
    elif direction == "lower":
        remaining = max(current - target, 0.0)
        margin = max(target - current, 0.0)
        passed = current <= target
    else:
        raise ValueError(f"Unknown target direction: {direction}")
    return {
        "target_id": target_id,
        "label": label,
        "direction": direction,
        "current": float(current),
        "target": float(target),
        "remaining": float(remaining),
        "margin": float(margin),
        "passed": bool(passed),
    }


def build_target_gap_table(m18_summary: dict[str, Any]) -> pd.DataFrame:
    metrics = m18_summary["metrics"]
    global_assessment = m18_summary["global_assessment"]
    source_gap = m18_summary["source_auc_gap"]
    robustness = m18_summary["robustness"]
    current_values = {
        "test_accuracy": metrics["accuracy"],
        "test_auc": metrics["auc"],
        "test_log_loss": metrics["log_loss"],
        "test_brier": metrics["brier_score"],
        "test_ece10": metrics["ece10"],
        "auc_ci_lower": global_assessment["auc_ci_lower_95"],
        "log_loss_ci_upper": global_assessment["log_loss_ci_upper_95"],
        "source_auc_gap": source_gap["absolute_difference"],
        "large_map_min_auc": robustness["large_map_min_auc"],
        "large_map_min_auc_ci_lower": robustness[
            "large_map_min_auc_ci_lower"
        ],
    }
    return pd.DataFrame(
        [
            _target_row(target_id, label, direction, current_values[target_id], target)
            for target_id, label, direction, target in TARGET_DEFINITIONS
        ]
    )


def build_internal_model_gap(comparison: pd.DataFrame) -> pd.DataFrame:
    test = comparison.loc[comparison["split"].eq("test")].set_index("model")
    required_models = {"logistic_regression", "xgboost_tuned"}
    missing = sorted(required_models - set(test.index))
    if missing:
        raise ValueError(f"Internal comparison is missing test models: {missing}")
    rows = []
    for metric in REPORT_METRICS:
        xgboost_value = float(test.loc["xgboost_tuned", metric])
        logistic_value = float(test.loc["logistic_regression", metric])
        raw_difference = xgboost_value - logistic_value
        advantage = (
            raw_difference if metric in HIGHER_IS_BETTER else -raw_difference
        )
        rows.append(
            {
                "metric": metric,
                "xgboost_tuned": xgboost_value,
                "logistic_regression": logistic_value,
                "raw_xgboost_minus_logistic": raw_difference,
                "performance_advantage_xgboost": advantage,
                "formal_target": np.nan,
                "note": "M16 defined no blocking XGBoost-over-logistic target",
            }
        )
    return pd.DataFrame(rows)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sigmoid(log_odds: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(log_odds, dtype=float), -709, 709)
    return 1.0 / (1.0 + np.exp(-clipped))


def verify_m18_prerequisite(
    data_path: str | Path,
    model_path: str | Path,
    m18_summary: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    data_artifact = fingerprint_file(data_path)
    model_artifact = fingerprint_file(model_path)
    expected_data_sha = m18_summary.get("data", {}).get("sha256")
    expected_model_sha = (
        m18_summary.get("prerequisite", {})
        .get("model_artifact", {})
        .get("sha256")
    )
    raw_features = list(bundle.get("raw_features", []))
    encoded_columns = list(bundle.get("columns", []))
    expected_encoded_count = int(
        m18_summary.get("model_replay", {}).get("encoded_feature_count", -1)
    )
    deployed_trees = deployment_tree_count(bundle)
    checks = {
        "m18_accepted": bool(
            m18_summary.get("acceptance", {}).get("ready_for_m19", False)
        ),
        "m18_task": m18_summary.get("task") == "post_first_kill",
        "data_sha256": bool(expected_data_sha)
        and data_artifact["sha256"] == expected_data_sha
        and bundle.get("data_sha256") == expected_data_sha,
        "model_sha256": bool(expected_model_sha)
        and model_artifact["sha256"] == expected_model_sha,
        "bundle_task": bundle.get("task") == "first_kill",
        "raw_feature_contract": raw_features == canonical_feature_names(),
        "encoded_feature_contract": bool(encoded_columns)
        and len(encoded_columns) == expected_encoded_count
        and len(encoded_columns) == len(set(encoded_columns)),
        "deployment_tree_contract": deployed_trees
        == int(bundle.get("best_tree_count", -1)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "data_artifact": data_artifact,
        "model_artifact": model_artifact,
        "raw_feature_count": len(raw_features),
        "encoded_feature_count": len(encoded_columns),
        "deployment_tree_count": deployed_trees,
    }


def prepare_explanation_inputs(
    data: pd.DataFrame,
    bundle: dict[str, Any],
    m18_summary: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    data_audit = audit_training_data(data)
    expected_split_rows = {
        name: int(count)
        for name, count in m18_summary.get("data", {})
        .get("split_rows", {})
        .items()
    }
    if not data_audit["passed"] or data_audit["split_rows"] != expected_split_rows:
        raise RuntimeError("M19 data identity or split rows differ from M18")

    prepared = prepare_profile_splits(data, list(bundle["raw_features"]))
    x_test, y_test, identity = prepared["test"]
    if x_test.columns.tolist() != list(bundle["columns"]):
        raise RuntimeError("M19 encoded test columns differ from the frozen model")
    probability = np.asarray(
        bundle["model"].predict_proba(x_test)[:, 1], dtype=float
    )
    current_metrics = probability_metrics(y_test, probability, n_bins=10)
    metric_difference = max(
        abs(float(current_metrics[name]) - float(m18_summary["metrics"][name]))
        for name in REPORT_METRICS
    )

    metadata_columns = [
        *ID_COLUMNS,
        "map_name",
        "round_num",
        "eq_value_diff_ct",
        "first_kill_advantage_ct",
        "first_kill_time",
        "first_kill_headshot",
        "first_kill_weapon",
    ]
    metadata = data.loc[data["split"].eq("test"), metadata_columns].copy()
    if metadata.duplicated(ID_COLUMNS).any():
        raise RuntimeError("M19 test metadata contains duplicate complete keys")
    predictions = identity.copy()
    predictions["y_true"] = y_test.to_numpy(dtype=int)
    predictions["ct_win_probability"] = probability
    predictions["predicted_label"] = (probability >= 0.5).astype(int)
    predictions["correct"] = predictions["predicted_label"].eq(
        predictions["y_true"]
    )
    predictions = predictions.merge(
        metadata,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    audit = {
        "passed": bool(
            metric_difference <= 1e-12
            and len(predictions) == expected_split_rows["test"]
            and not predictions[ID_COLUMNS].isna().any().any()
        ),
        "test_rows": int(len(predictions)),
        "metric_max_absolute_difference_vs_m18": float(metric_difference),
        "xgboost_fit_calls": 0,
        "split_rows": data_audit["split_rows"],
        "cross_split_series": data_audit["cross_split_series"],
        "duplicate_key_rows": data_audit["duplicate_key_rows"],
    }
    return x_test, y_test, predictions, audit


def run_explanation_core(
    data_path: str | Path,
    model_path: str | Path,
    m18_summary_path: str | Path,
    *,
    permutation_repeats: int = 20,
    seed: int = 42,
    case_features: int = 10,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    m18_summary = _read_json(m18_summary_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("M19 expected the M17 model artifact to contain a bundle")
    prerequisite = verify_m18_prerequisite(
        data_path, model_path, m18_summary, bundle
    )
    if not prerequisite["passed"]:
        raise RuntimeError("M19 input does not match accepted M18 artifacts")

    data = read_table(data_path)
    x_test, y_test, predictions, model_replay = prepare_explanation_inputs(
        data, bundle, m18_summary
    )
    if not model_replay["passed"]:
        raise RuntimeError("M19 could not replay the frozen M18 test metrics")

    gain = gain_importance(bundle)
    encoded_permutation = permutation_auc_importance(
        bundle["model"],
        x_test,
        y_test,
        n_repeats=permutation_repeats,
        seed=seed,
    )
    shap_values, base_values = tree_shap_contributions(bundle, x_test)
    shap = shap_importance(shap_values)
    encoded_comparison = build_importance_comparison(
        gain, encoded_permutation, shap
    )

    raw_features = list(bundle["raw_features"])
    encoded_columns = list(bundle["columns"])
    encoded_contract = audit_post_first_kill_features(
        encoded_columns, raw_features
    )
    leakage_audit = audit_post_first_kill_features(
        shap["feature"].tolist(), raw_features
    ).merge(
        shap[["feature", "shap_rank", "mean_abs_shap"]],
        left_on="encoded_feature",
        right_on="feature",
        how="left",
        validate="one_to_one",
    ).drop(columns="feature")
    top20_audit = leakage_audit.head(20).copy()

    source_groups = build_source_feature_groups(encoded_columns, raw_features)
    grouped_permutation = grouped_permutation_auc_importance(
        bundle["model"],
        x_test,
        y_test,
        source_groups,
        n_repeats=permutation_repeats,
        seed=seed,
    )
    macro_groups = build_macro_feature_groups(encoded_columns, raw_features)
    macro_permutation = grouped_permutation_auc_importance(
        bundle["model"],
        x_test,
        y_test,
        macro_groups,
        n_repeats=permutation_repeats,
        seed=seed + 1,
    )
    source_importance = build_source_importance_summary(
        gain, shap, grouped_permutation, encoded_contract
    )

    probability = predictions["ct_win_probability"].to_numpy(dtype=float)
    reconstructed = _sigmoid(base_values + shap_values.sum(axis=1).to_numpy())
    reconstruction_error = np.abs(reconstructed - probability)
    cases = select_explanation_cases(predictions)
    case_explanations = build_case_explanations(
        cases,
        x_test,
        shap_values,
        base_values,
        top_n=case_features,
    )
    rank_correlations = encoded_comparison[
        ["gain_rank", "permutation_rank", "shap_rank"]
    ].corr(method="spearman")
    summary = {
        "prerequisite": prerequisite,
        "model_replay": model_replay,
        "test_rounds": int(len(x_test)),
        "raw_features": len(raw_features),
        "encoded_features": len(encoded_columns),
        "available_tree_count": len(bundle["model"].get_booster().get_dump()),
        "deployment_tree_count": deployment_tree_count(bundle),
        "permutation_repeats": permutation_repeats,
        "shap_reconstruction_max_abs_error": float(reconstruction_error.max()),
        "shap_reconstruction_mean_abs_error": float(reconstruction_error.mean()),
        "feature_audit": {
            "all_feature_failures": int(
                leakage_audit["audit_result"].eq("fail").sum()
            ),
            "top20_failures": int(
                top20_audit["audit_result"].eq("fail").sum()
            ),
            "mapped_source_features": int(
                encoded_contract["source_feature"].nunique()
            ),
        },
        "top_features": {
            "encoded_gain": gain.head(10)["feature"].tolist(),
            "encoded_permutation": encoded_permutation.head(10)["feature"].tolist(),
            "encoded_shap": shap.head(10)["feature"].tolist(),
            "source_mean_rank": source_importance.head(10)[
                "source_feature"
            ].tolist(),
            "source_grouped_permutation": grouped_permutation.head(10)[
                "feature_group"
            ].tolist(),
        },
        "importance_rank_spearman": rank_correlations.to_dict(),
        "selected_cases": cases.to_dict(orient="records"),
    }
    tables = {
        "gain_importance": gain,
        "permutation_importance_auc": encoded_permutation,
        "shap_importance": shap,
        "importance_comparison": encoded_comparison,
        "encoded_feature_contract": encoded_contract,
        "all_feature_leakage_audit": leakage_audit,
        "top20_feature_audit": top20_audit,
        "grouped_permutation_importance_auc": grouped_permutation,
        "macro_group_permutation_auc": macro_permutation,
        "source_feature_importance": source_importance,
        "selected_cases": cases,
        "case_explanations": case_explanations,
        "x_test": x_test,
        "shap_values": shap_values,
    }
    return summary, tables


def decide_acceptance(checks: dict[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not checks.get(name, False)]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "ready_for_m20": not failures,
    }


def render_m19_external_report(external: pd.DataFrame) -> str:
    report = render_external_report(external)
    return report.replace("# M18 外部模型指标差距", "# M19 外部模型指标差距", 1)


def save_m19_plots(
    tables: dict[str, pd.DataFrame],
    report_dir: str | Path,
    *,
    seed: int,
    shap_plot_rows: int,
) -> None:
    output_dir = Path(report_dir)
    save_explanation_plots(
        tables,
        output_dir,
        seed=seed,
        shap_plot_rows=shap_plot_rows,
    )
    plt.style.use("seaborn-v0_8-whitegrid")

    source = tables["source_feature_importance"].sort_values(
        "grouped_auc_decrease_mean", ascending=False
    ).head(20)
    source = source.sort_values("grouped_auc_decrease_mean", ascending=True)
    colors = np.where(
        source["feature_group"].eq("first_kill_event"), "#C44E52", "#176B87"
    )
    figure, axis = plt.subplots(figsize=(8.6, 6.6))
    axis.barh(
        source["source_feature"],
        source["grouped_auc_decrease_mean"],
        xerr=source["grouped_auc_decrease_std"],
        color=colors,
        capsize=2,
    )
    axis.axvline(0, color="#4B5563", linewidth=1)
    axis.set(
        title="M19 grouped permutation importance by raw feature",
        xlabel="Mean decrease in fixed-test AUC",
        ylabel="Raw feature",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "source_feature_grouped_permutation.png", dpi=170)
    plt.close(figure)

    macro = tables["macro_group_permutation_auc"].sort_values(
        "auc_decrease_mean", ascending=True
    )
    macro_colors = macro["feature_group"].map(
        {"purchase_end": "#176B87", "first_kill_event": "#C44E52"}
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.barh(
        macro["feature_group"],
        macro["auc_decrease_mean"],
        xerr=macro["auc_decrease_std"],
        color=macro_colors,
        capsize=3,
    )
    axis.set(
        title="M19 purchase-end vs first-kill event signal",
        xlabel="Mean decrease in fixed-test AUC",
        ylabel="Feature timing group",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "macro_group_permutation_auc.png", dpi=170)
    plt.close(figure)

    target_gap = tables["target_gap"].copy().sort_values("margin", ascending=True)
    passed = target_gap["passed"].astype(bool).to_numpy()
    values = np.where(passed, target_gap["margin"], -target_gap["remaining"])
    target_colors = np.where(passed, "#2A9D8F", "#C44E52")
    figure, axis = plt.subplots(figsize=(8.4, 5.8))
    axis.barh(target_gap["label"], values, color=target_colors)
    axis.axvline(0, color="#111827", linewidth=1)
    axis.set(
        title="M19 distance to formal stage targets",
        xlabel="Positive = margin passed; negative = remaining improvement",
        ylabel="Target",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "target_gap.png", dpi=170)
    plt.close(figure)


def render_m19_report(
    summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> str:
    target_gap = tables["target_gap"]
    source = tables["source_feature_importance"]
    macro = tables["macro_group_permutation_auc"]
    audit = tables["top20_feature_audit"]
    cases = tables["selected_cases"]
    explanations = tables["case_explanations"]
    internal = tables["internal_model_gap"]
    external = tables["external_benchmark_comparison"]
    acceptance = summary["acceptance"]
    correlations = summary["importance_rank_spearman"]
    lines = [
        "# M19 首杀后模型解释与泄漏审计报告",
        "",
        "## 阶段结论",
        "",
        f"阻断验收状态：**{acceptance['status']}**；可进入 M20："
        f"**{acceptance['ready_for_m20']}**。",
        "本阶段没有训练、调参、删除特征或改变测试概率，只解释 M17/M18 冻结模型。",
        f"模型文件保存 {summary['available_tree_count']} 棵树，部署使用 early stopping "
        f"选中的 {summary['deployment_tree_count']} 棵树。",
        f"TreeSHAP 重建测试概率最大绝对误差为 "
        f"`{summary['shap_reconstruction_max_abs_error']:.10f}`。",
        "",
        "## 离正式目标还有多少",
        "",
        "`remaining` 是仍需改善量；`margin` 是已经超过目标的余量。",
        "",
        "| 目标 | 当前 | 通过线 | 方向 | Remaining | Margin | 通过 |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in target_gap.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {row['current']:.6f} | {row['target']:.6f} | "
            f"{row['direction']} | {row['remaining']:.6f} | {row['margin']:.6f} | "
            f"{row['passed']} |"
        )
    lines.extend(
        [
            "",
            f"十项正式目标通过 {summary['target_gap']['passed_count']}/10；"
            f"仍需改善的正式目标数为 {summary['target_gap']['remaining_count']}。",
            "这表示首杀后 XGBoost 的当前统计验收目标已经达到；它不表示实时胜率、"
            "LightGBM 对照或整个课题已经结束。",
            "",
            "从项目模块看，M19 通过后，首杀后 XGBoost 还剩 M20 单条预测接口和 M21 "
            "最终验收两个模块；之后才进入 LightGBM 同数据对照和实时胜率。",
            "",
            "## 原始特征重要性",
            "",
            "分组 Permutation 会把一个原始特征的全部独热列用同一个排列一起打乱。",
            "Gain 和 SHAP 则把对应编码列的值聚合回原始特征。",
            "",
            "| 原始特征 | 时点组 | Gain 排名 | 分组 Permutation 排名 | SHAP 排名 | "
            "AUC 下降 | Mean abs SHAP |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in source.head(15).to_dict(orient="records"):
        lines.append(
            f"| `{row['source_feature']}` | {row['feature_group']} | "
            f"{int(row['source_gain_rank'])} | {int(row['source_permutation_rank'])} | "
            f"{int(row['source_shap_rank'])} | "
            f"{row['grouped_auc_decrease_mean']:.6f} | "
            f"{row['mean_abs_shap']:.6f} |"
        )
    lines.extend(
        [
            "",
            "编码列三种排名的 Spearman 相关系数：Gain-Permutation "
            f"`{correlations['gain_rank']['permutation_rank']:.3f}`，Gain-SHAP "
            f"`{correlations['gain_rank']['shap_rank']:.3f}`，Permutation-SHAP "
            f"`{correlations['permutation_rank']['shap_rank']:.3f}`。",
            "相关特征会分摊重要性，负 permutation 值也被保留；三种方法都不是因果证明。",
            "",
            "## 购买信息与首杀信息",
            "",
            "| 特征组 | 编码列 | AUC 下降均值 | 标准差 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in macro.to_dict(orient="records"):
        lines.append(
            f"| {row['feature_group']} | {int(row['encoded_column_count'])} | "
            f"{row['auc_decrease_mean']:.6f} | {row['auc_decrease_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "四个首杀事件原始特征：",
            "",
            "| 特征 | Gain 排名 | Permutation 排名 | SHAP 排名 | AUC 下降 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    event_rows = source.loc[source["feature_group"].eq("first_kill_event")]
    for row in event_rows.sort_values("source_permutation_rank").to_dict(
        orient="records"
    ):
        lines.append(
            f"| `{row['source_feature']}` | {int(row['source_gain_rank'])} | "
            f"{int(row['source_permutation_rank'])} | "
            f"{int(row['source_shap_rank'])} | "
            f"{row['grouped_auc_decrease_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 泄漏审计",
            "",
            f"{summary['encoded_features']} 个编码列全部追溯到 "
            f"{summary['raw_features']} 个允许原始特征。全部特征失败数 "
            f"`{summary['feature_audit']['all_feature_failures']}`，TreeSHAP 前 20 "
            f"失败数 `{summary['feature_audit']['top20_failures']}`。",
            "首杀字段只对“首杀刚发生后”合法，不可复制到购买结束模型。ID、标签、"
            "后续击杀、血量/伤害、下包和回合结束字段都没有进入模型。",
            "",
            "| SHAP 排名 | 编码列 | 原始特征 | 时点 | 结果 |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in audit.to_dict(orient="records"):
        lines.append(
            f"| {int(row['shap_rank'])} | `{row['encoded_feature']}` | "
            f"`{row['source_feature']}` | {row['availability']} | "
            f"{row['audit_result']} |"
        )
    lines.extend(
        [
            "",
            "## 三个回合案例",
            "",
            "| 案例 | 主键 | 地图 | 实际 | CT 概率 | 正确 |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for case in cases.to_dict(orient="records"):
        actual = "CT" if int(case["y_true"]) == 1 else "T"
        key = f"{case['series_id']} / {case['game_id']} / {case['round_id']}"
        lines.append(
            f"| {case['case_type']} | `{key}` | {case['map_name']} | {actual} | "
            f"{case['ct_win_probability']:.6f} | {case['correct']} |"
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
                "| 排名 | 编码列 | 值 | SHAP log-odds | 方向 |",
                "|---:|---|---:|---:|---|",
            ]
        )
        part = explanations.loc[
            explanations["case_type"].eq(case_type)
            & explanations["contribution_rank"].le(5)
        ].sort_values("contribution_rank")
        for row in part.to_dict(orient="records"):
            lines.append(
                f"| {int(row['contribution_rank'])} | `{row['feature']}` | "
                f"{row['feature_value']} | {row['shap_value_log_odds']:.6f} | "
                f"{row['direction']} |"
            )
    lines.extend(
        [
            "",
            "## XGBoost 与逻辑回归",
            "",
            "M16 没有为“XGBoost 必须领先逻辑回归”设置正式通过线，以下只报告差值。",
            "`performance_advantage_xgboost` 为正表示 XGBoost 更好。",
            "",
            "| 指标 | XGBoost | 逻辑回归 | 原始差 | 性能优势 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in internal.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['xgboost_tuned']:.6f} | "
            f"{row['logistic_regression']:.6f} | "
            f"{row['raw_xgboost_minus_logistic']:+.6f} | "
            f"{row['performance_advantage_xgboost']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 别人的指标与差值",
            "",
            "差值为“本项目 - 外部报告”。Accuracy/AUC 同时显示百分点；低值更好的"
            "指标不能只看差值正负。",
            "",
            "| 可比性 | 当前模型 | 外部来源 | 指标 | 当前 | 外部 | 差值 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in external.to_dict(orient="records"):
        difference = float(row["raw_difference_ours_minus_reported"])
        difference_text = (
            f"{difference * 100:+.2f} 百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference:+.6f}"
        )
        lines.append(
            f"| {row.get('comparability', '')} | `{row['current_model']}` | "
            f"{row.get('source_title', row['benchmark_id'])} | {row['metric']} | "
            f"{row['current_value']:.6f} | {row['reported_value']:.6f} | "
            f"{difference_text} |"
        )
    lines.extend(
        [
            "",
            "外部数据、预测时点、特征和切分不同，不能把这些差值解释为算法同场排名。",
            "",
            "## 下一阶段",
            "",
            "M20 建立首杀后 JSON/CSV 单条预测接口和输入一致性校验；M21 做首杀后 "
            "XGBoost 最终验收。随后进行 LightGBM 同数据对照，再进入实时胜率。",
            "",
            "运行命令：",
            "",
            "```powershell",
            ".\\scripts\\run_first_kill_explanation.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    data_path: str | Path,
    model_path: str | Path,
    m18_summary_path: str | Path,
    m17_comparison_path: str | Path,
    benchmarks_path: str | Path,
    report_dir: str | Path,
    project_root: str | Path,
    *,
    permutation_repeats: int = 20,
    seed: int = 42,
    case_features: int = 10,
    shap_plot_rows: int = 1500,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    core_summary, tables = run_explanation_core(
        data_path,
        model_path,
        m18_summary_path,
        permutation_repeats=permutation_repeats,
        seed=seed,
        case_features=case_features,
    )

    m18_summary = _read_json(m18_summary_path)
    target_gap = build_target_gap_table(m18_summary)
    m17_comparison = read_table(m17_comparison_path)
    internal_gap = build_internal_model_gap(m17_comparison)
    benchmarks = read_table(benchmarks_path)
    external = compare_external_models(m17_comparison, benchmarks)
    external_report = render_m19_external_report(external)
    tables.update(
        {
            "target_gap": target_gap,
            "internal_model_gap": internal_gap,
            "external_benchmark_comparison": external,
        }
    )

    automated_tests = run_automated_tests(project_root)
    test_count_match = re.search(r"Ran (\d+) tests?", automated_tests["output"])
    automated_test_count = (
        int(test_count_match.group(1)) if test_count_match else None
    )
    expected_types = {
        "ct_high_probability",
        "t_high_probability",
        "high_confidence_error",
    }
    importance_complete = bool(
        len(tables["gain_importance"]) == core_summary["encoded_features"]
        and len(tables["permutation_importance_auc"])
        == core_summary["encoded_features"]
        and len(tables["shap_importance"]) == core_summary["encoded_features"]
        and len(tables["grouped_permutation_importance_auc"])
        == core_summary["raw_features"]
        and len(tables["macro_group_permutation_auc"]) == 2
        and tables["permutation_importance_auc"]["n_repeats"]
        .eq(permutation_repeats)
        .all()
        and tables["grouped_permutation_importance_auc"]["n_repeats"]
        .eq(permutation_repeats)
        .all()
    )
    feature_contract_passed = bool(
        core_summary["feature_audit"]["all_feature_failures"] == 0
        and core_summary["feature_audit"]["top20_failures"] == 0
        and core_summary["feature_audit"]["mapped_source_features"]
        == core_summary["raw_features"]
        and len(tables["encoded_feature_contract"])
        == core_summary["encoded_features"]
    )
    cases_passed = bool(
        set(tables["selected_cases"]["case_type"]) == expected_types
        and len(tables["selected_cases"]) == 3
        and len(tables["case_explanations"]) == 3 * case_features
        and tables["case_explanations"]
        .groupby("case_type")["contribution_rank"]
        .max()
        .eq(case_features)
        .all()
    )
    target_and_internal_passed = bool(
        len(target_gap) == len(TARGET_DEFINITIONS)
        and target_gap[
            ["current", "target", "remaining", "margin"]
        ].notna().all().all()
        and len(internal_gap) == len(REPORT_METRICS)
        and internal_gap["formal_target"].isna().all()
    )
    checks = {
        "m18_prerequisite": core_summary["prerequisite"]["passed"],
        "model_replay": core_summary["model_replay"]["passed"],
        "importance_methods": importance_complete,
        "feature_mapping_and_leakage": feature_contract_passed,
        "shap_reconstruction": core_summary[
            "shap_reconstruction_max_abs_error"
        ]
        <= 1e-5,
        "case_explanations": cases_passed,
        "target_and_internal_gaps": target_and_internal_passed,
        "external_report": bool(len(external) == len(benchmarks) and external_report),
        "automated_tests": automated_tests["passed"],
    }
    acceptance = decide_acceptance(checks)
    target_passed_count = int(target_gap["passed"].sum())
    target_remaining_count = int(target_gap["remaining"].gt(0).sum())
    source_importance = tables["source_feature_importance"]
    event_importance = source_importance.loc[
        source_importance["feature_group"].eq("first_kill_event")
    ].sort_values("source_permutation_rank")
    summary = {
        "stage": "M19",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "definition": "immediately after the first valid enemy kill",
        "model_policy": "M17/M18 model frozen; no fit, tuning, or feature selection",
        "acceptance": acceptance,
        "checks": checks,
        "prerequisite": core_summary["prerequisite"],
        "model_replay": core_summary["model_replay"],
        "test_rounds": core_summary["test_rounds"],
        "raw_features": core_summary["raw_features"],
        "encoded_features": core_summary["encoded_features"],
        "available_tree_count": core_summary["available_tree_count"],
        "deployment_tree_count": core_summary["deployment_tree_count"],
        "permutation_repeats": permutation_repeats,
        "seed": seed,
        "shap_method": "xgboost_native_tree_shap",
        "shap_units": "log_odds",
        "shap_reconstruction_max_abs_error": core_summary[
            "shap_reconstruction_max_abs_error"
        ],
        "shap_reconstruction_mean_abs_error": core_summary[
            "shap_reconstruction_mean_abs_error"
        ],
        "feature_audit": core_summary["feature_audit"],
        "top_features": core_summary["top_features"],
        "importance_rank_spearman": core_summary["importance_rank_spearman"],
        "macro_group_permutation": json.loads(
            tables["macro_group_permutation_auc"].to_json(orient="records")
        ),
        "first_kill_event_importance": json.loads(
            event_importance.to_json(orient="records")
        ),
        "selected_cases": core_summary["selected_cases"],
        "target_gap": {
            "passed_count": target_passed_count,
            "target_count": int(len(target_gap)),
            "remaining_count": target_remaining_count,
            "all_formal_targets_passed": target_passed_count == len(target_gap),
            "rows": json.loads(target_gap.to_json(orient="records")),
        },
        "internal_model_gap": json.loads(
            internal_gap.to_json(orient="records")
        ),
        "external_comparison_rows": int(len(external)),
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_test_count,
        },
        "roadmap": {
            "pre_round_xgboost": "complete_through_M14",
            "first_kill_xgboost_current": "M19_explanation_complete",
            "first_kill_xgboost_modules_remaining_after_m19": 2,
            "remaining_modules": [
                "M20 first-kill JSON/CSV prediction interface",
                "M21 first-kill final acceptance",
            ],
            "later_tracks": [
                "LightGBM controlled comparison",
                "real-time win probability data and model",
            ],
        },
        "next_stage": "M20 first-kill prediction interface",
    }

    saved_tables = {
        name: table
        for name, table in tables.items()
        if name not in {"x_test", "shap_values", "external_benchmark_comparison"}
    }
    for name, table in saved_tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)
    external.to_csv(
        report_dir / "external_benchmark_comparison.csv", index=False
    )
    (report_dir / "external_benchmark_comparison.md").write_text(
        external_report, encoding="utf-8"
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests["output"], encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"check": name, "passed": passed, "blocking": True}
            for name, passed in checks.items()
        ]
    ).to_csv(report_dir / "m19_checks.csv", index=False)
    write_json(summary, report_dir / "m19_summary.json")
    (report_dir / "m19_first_kill_explanation_report.md").write_text(
        render_m19_report(summary, tables), encoding="utf-8"
    )
    save_m19_plots(
        tables,
        report_dir,
        seed=seed,
        shap_plot_rows=shap_plot_rows,
    )
    return summary, tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M19 first-kill Gain, permutation, TreeSHAP, and leakage audit."
    )
    parser.add_argument(
        "--data", default="data/processed/esta_full/first_kill.parquet"
    )
    parser.add_argument(
        "--model",
        default="models/esta_full_m17/first_kill_xgboost_tuned.joblib",
    )
    parser.add_argument(
        "--m18-summary", default="reports/esta_full_m18/m18_summary.json"
    )
    parser.add_argument(
        "--m17-comparison",
        default="reports/esta_full_m17/model_comparison.csv",
    )
    parser.add_argument(
        "--benchmarks",
        default="benchmarks/external_first_kill_tuned_metrics.csv",
    )
    parser.add_argument("--report-dir", default="reports/esta_full_m19")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-features", type=int, default=10)
    parser.add_argument("--shap-plot-rows", type=int, default=1500)
    args = parser.parse_args()

    summary, tables = run(
        data_path=args.data,
        model_path=args.model,
        m18_summary_path=args.m18_summary,
        m17_comparison_path=args.m17_comparison,
        benchmarks_path=args.benchmarks,
        report_dir=args.report_dir,
        project_root=args.project_root,
        permutation_repeats=args.permutation_repeats,
        seed=args.seed,
        case_features=args.case_features,
        shap_plot_rows=args.shap_plot_rows,
    )
    print(
        tables["source_feature_importance"][
            [
                "source_feature",
                "feature_group",
                "source_gain_rank",
                "source_permutation_rank",
                "source_shap_rank",
                "grouped_auc_decrease_mean",
            ]
        ]
        .head(12)
        .round(6)
        .to_string(index=False)
    )
    print(
        f"M19 {summary['acceptance']['status']}; "
        f"ready_for_m20={summary['acceptance']['ready_for_m20']}"
    )
    if not summary["acceptance"]["ready_for_m20"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

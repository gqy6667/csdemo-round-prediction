from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .config import RANDOM_STATE, REPORT_DIR
from .io import read_table
from .m9_evaluation import METRIC_ORDER, bootstrap_metric_intervals
from .metrics import probability_metrics
from .schema import PRE_ROUND_FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROUND_STAGE_LABELS = ("early_01_10", "middle_11_20", "late_21_plus")
EQUIPMENT_BAND_LABELS = (
    "t_major",
    "t_moderate",
    "balanced",
    "ct_moderate",
    "ct_major",
)


def assign_round_stage(round_number: pd.Series) -> pd.Series:
    values = pd.to_numeric(round_number, errors="raise")
    return pd.Series(
        pd.cut(
            values,
            bins=[0, 11, 21, np.inf],
            labels=ROUND_STAGE_LABELS,
            right=False,
        ).astype("string"),
        index=round_number.index,
    )


def assign_equipment_band(equipment_difference: pd.Series) -> pd.Series:
    values = pd.to_numeric(equipment_difference, errors="raise")
    labels = np.select(
        [
            values.lt(-5000),
            values.lt(-1500),
            values.le(1500),
            values.le(5000),
        ],
        EQUIPMENT_BAND_LABELS[:-1],
        default=EQUIPMENT_BAND_LABELS[-1],
    )
    return pd.Series(labels, index=equipment_difference.index, dtype="string")


def select_high_confidence_errors(
    predictions: pd.DataFrame,
    *,
    minimum_confidence: float = 0.8,
    max_cases: int | None = 30,
) -> pd.DataFrame:
    probability = predictions["ct_win_probability"].to_numpy(dtype=float)
    predicted = predictions["predicted_label"].to_numpy(dtype=int)
    assigned_probability = np.where(predicted == 1, probability, 1 - probability)
    wrong = predictions["y_true"].to_numpy(dtype=int) != predicted
    selected = predictions.loc[wrong & (assigned_probability >= minimum_confidence)].copy()
    selected["assigned_side_probability"] = assigned_probability[
        wrong & (assigned_probability >= minimum_confidence)
    ]
    selected = selected.sort_values(
        ["assigned_side_probability", "round_id"], ascending=[False, True]
    )
    if max_cases is not None:
        selected = selected.head(max_cases)
    return selected.reset_index(drop=True)


def _favored_value(row: pd.Series, ct_value: str) -> float:
    value = float(row[ct_value])
    return value if int(row["predicted_label"]) == 1 else -value


def pre_round_error_pattern(row: pd.Series) -> str:
    favored_equipment = _favored_value(row, "eq_value_diff_ct")
    favored_rifles = _favored_value(row, "rifle_diff_ct")
    favored_awp = _favored_value(row, "awp_diff_ct")
    total_equipment = float(row["ct_eq_value"]) + float(row["t_eq_value"])

    if total_equipment < 15000:
        return "low_equipment_volatility"
    if favored_equipment >= 5000:
        return "favored_side_major_equipment_upset"
    if favored_equipment >= 1500:
        return "favored_side_moderate_equipment_upset"
    if favored_equipment <= -1500:
        return "model_confidence_against_equipment"
    if abs(favored_rifles) <= 1 and abs(favored_awp) <= 1:
        return "balanced_buy_prior_error"
    return "weapon_signal_without_equipment_edge"


def outcome_error_pattern(row: pd.Series) -> str:
    first_kill_side = row.get("first_kill_side")
    if pd.isna(first_kill_side):
        return "no_valid_first_kill_event"
    predicted_side = "CT" if int(row["predicted_label"]) == 1 else "T"
    if str(first_kill_side).upper() == predicted_side:
        return "predicted_favorite_lost_after_first_kill"
    return "predicted_favorite_lost_first_kill"


def group_metrics_with_intervals(
    predictions: pd.DataFrame,
    group_column: str,
    *,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    rows = []
    for group_value, group in predictions.groupby(group_column, dropna=False):
        intervals = bootstrap_metric_intervals(
            group,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ).set_index("metric")
        row = {
            group_column: group_value,
            "rounds": int(len(group)),
            "series": int(group["series_id"].nunique()),
            "ct_win_rate": float(group["y_true"].mean()),
        }
        for metric in METRIC_ORDER:
            metric_row = intervals.loc[metric]
            row[metric] = float(metric_row["point_estimate"])
            row[f"{metric}_ci_lower_95"] = float(metric_row["ci_lower_95"])
            row[f"{metric}_ci_upper_95"] = float(metric_row["ci_upper_95"])
            row[f"{metric}_successful_bootstraps"] = int(
                metric_row["successful_bootstraps"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rounds", ascending=False)


def _bootstrap_auc_values(
    group: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> np.ndarray:
    series = group["series_id"].astype(str).to_numpy()
    y = group["y_true"].to_numpy(dtype=int)
    probability = group["ct_win_probability"].to_numpy(dtype=float)
    unique_series = np.unique(series)
    positions = [np.flatnonzero(series == value) for value in unique_series]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(positions), size=len(positions))
        sampled = np.concatenate([positions[index] for index in selected])
        auc = probability_metrics(y[sampled], probability[sampled])["auc"]
        if np.isfinite(auc):
            values.append(auc)
    return np.asarray(values, dtype=float)


def source_auc_gap(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict:
    lan = predictions[predictions["source_subset"].eq("lan")]
    online = predictions[predictions["source_subset"].eq("online")]
    lan_auc = probability_metrics(lan["y_true"], lan["ct_win_probability"])["auc"]
    online_auc = probability_metrics(
        online["y_true"], online["ct_win_probability"]
    )["auc"]
    lan_samples = _bootstrap_auc_values(lan, n_bootstrap=n_bootstrap, seed=seed)
    online_samples = _bootstrap_auc_values(
        online, n_bootstrap=n_bootstrap, seed=seed + 1
    )
    paired_count = min(len(lan_samples), len(online_samples))
    differences = lan_samples[:paired_count] - online_samples[:paired_count]
    return {
        "comparison": "lan_minus_online_auc",
        "lan_auc": lan_auc,
        "online_auc": online_auc,
        "signed_difference": lan_auc - online_auc,
        "absolute_difference": abs(lan_auc - online_auc),
        "ci_lower_95": float(np.quantile(differences, 0.025)),
        "ci_upper_95": float(np.quantile(differences, 0.975)),
        "successful_bootstraps": int(paired_count),
        "bootstrap_unit": "series_id_within_source",
    }


def prepare_analysis_table(
    predictions: pd.DataFrame, features: pd.DataFrame, kills: pd.DataFrame
) -> pd.DataFrame:
    test_features = features[features["split"].eq("test")].copy()
    add_features = [
        column
        for column in PRE_ROUND_FEATURES
        if column not in predictions.columns and column in test_features.columns
    ]
    analysis = predictions.merge(
        test_features[["round_id", *add_features]],
        on="round_id",
        how="left",
        validate="one_to_one",
    )
    if analysis[add_features].isna().any().any():
        raise ValueError("M11 feature join produced missing values")

    first_kills = kills[kills["is_first_kill"].eq(1)][
        ["round_id", "killer_side", "victim_side", "weapon", "headshot", "time"]
    ].rename(
        columns={
            "killer_side": "first_kill_side",
            "victim_side": "first_death_side",
            "weapon": "first_kill_weapon",
            "headshot": "first_kill_headshot",
            "time": "first_kill_time",
        }
    )
    if first_kills["round_id"].duplicated().any():
        raise ValueError("M11 found multiple first kills for one round")
    analysis = analysis.merge(first_kills, on="round_id", how="left", validate="one_to_one")
    analysis["round_stage"] = assign_round_stage(analysis["round_num"])
    analysis["equipment_band"] = assign_equipment_band(
        analysis["eq_value_diff_ct"]
    )
    return analysis


def enrich_error_cases(errors: pd.DataFrame) -> pd.DataFrame:
    errors = errors.copy()
    errors["predicted_side"] = np.where(errors["predicted_label"].eq(1), "CT", "T")
    errors["actual_winner"] = np.where(errors["y_true"].eq(1), "CT", "T")
    errors["pre_round_pattern"] = errors.apply(pre_round_error_pattern, axis=1)
    errors["outcome_pattern"] = errors.apply(outcome_error_pattern, axis=1)
    errors["predicted_side_won_first_kill"] = (
        errors["predicted_side"].eq(errors["first_kill_side"])
    ).where(errors["first_kill_side"].notna())
    return errors


def write_case_review(cases: pd.DataFrame, path: Path) -> None:
    lines = [
        "# M11 High-Confidence Error Review",
        "",
        "These are post-hoc diagnostic patterns, not proven causal explanations.",
        "First-kill fields are outcomes used only for error analysis and never model inputs.",
        "",
    ]
    for index, row in cases.reset_index(drop=True).iterrows():
        first_kill = row["first_kill_side"] if pd.notna(row["first_kill_side"]) else "none"
        lines.append(
            f"{index + 1}. `{row['round_id']}`: predicted {row['predicted_side']} "
            f"at {row['assigned_side_probability']:.3f}, actual {row['actual_winner']}; "
            f"equipment diff CT {row['eq_value_diff_ct']:+.0f}, first kill {first_kill}; "
            f"{row['pre_round_pattern']} / {row['outcome_pattern']}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(
    map_metrics: pd.DataFrame, error_summary: pd.DataFrame, report_dir: Path
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    maps = map_metrics.sort_values("auc").copy()
    maps["display_name"] = maps.apply(
        lambda row: f"{row['map_name']} (n={row['rounds']}, series={row['series']})",
        axis=1,
    )
    lower = maps["auc"] - maps["auc_ci_lower_95"]
    upper = maps["auc_ci_upper_95"] - maps["auc"]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.errorbar(
        maps["auc"],
        maps["display_name"],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#176B87",
        ecolor="#6B7280",
        capsize=3,
    )
    ax.axvline(0.67, color="#C44E52", linestyle="--", linewidth=1, label="Minimum 0.67")
    ax.axvline(0.69, color="#2A9D8F", linestyle=":", linewidth=1.5, label="Target 0.69")
    ax.set(xlabel="Test AUC with series-level 95% CI", ylabel="Map", title="M11 map robustness")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / "map_auc_with_ci.png", dpi=160)
    plt.close(fig)

    error_plot = error_summary.sort_values("cases").copy()
    error_plot["display_pattern"] = error_plot["pattern"].str.replace("_", " ")
    colors = error_plot["pattern_type"].map(
        {"pre_round": "#C44E52", "outcome": "#176B87"}
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.barh(error_plot["display_pattern"], error_plot["cases"], color=colors)
    ax.set(
        xlabel="High-confidence error cases",
        ylabel="Post-hoc pattern",
        title="M11 reviewed two-layer error patterns",
    )
    fig.tight_layout()
    fig.savefig(report_dir / "error_pattern_counts.png", dpi=160)
    plt.close(fig)


def write_report(summary: dict, path: Path) -> None:
    source_gap = summary["source_auc_gap"]
    lines = [
        "# M11 Robustness and Error Analysis Report",
        "",
        f"LAN-online absolute AUC gap: {source_gap['absolute_difference']:.6f} "
        f"(signed 95% CI [{source_gap['ci_lower_95']:.6f}, "
        f"{source_gap['ci_upper_95']:.6f}]).",
        f"Maps with at least 300 rounds: {summary['large_map_count']}; "
        f"minimum AUC: {summary['large_map_min_auc']:.6f}.",
        f"Lowest large-map AUC CI lower bound: "
        f"{summary['large_map_min_auc_ci_lower']:.6f}.",
        f"High-confidence wrong rounds available: {summary['high_confidence_errors_available']}; "
        f"reviewed: {summary['reviewed_error_cases']}.",
        "",
        "## Acceptance",
        "",
        f"LAN-online AUC gap <= 0.04: {summary['source_gap_passed']}.",
        f"Large-map minimum AUC >= 0.67: {summary['large_map_minimum_passed']}.",
        f"Large-map minimum AUC >= 0.69: {summary['large_map_target_passed']}.",
        f"Every large-map AUC CI lower bound >= 0.67: "
        f"{summary['all_large_map_ci_lower_minimum_passed']}.",
        f"At least 30 high-confidence errors reviewed: {summary['error_review_passed']}.",
        "All group tables include rounds, series counts, and series-level 95% intervals.",
        "First-kill data is used only as a post-hoc diagnostic outcome.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    predictions_path: str | Path,
    data_path: str | Path,
    kills_path: str | Path,
    report_dir: str | Path,
    *,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
    review_cases: int = 30,
):
    predictions = read_table(predictions_path)
    features = read_table(data_path)
    kills = read_table(kills_path)
    analysis = prepare_analysis_table(predictions, features, kills)

    grouped = {
        "map": group_metrics_with_intervals(
            analysis, "map_name", n_bootstrap=n_bootstrap, seed=seed
        ),
        "source": group_metrics_with_intervals(
            analysis, "source_subset", n_bootstrap=n_bootstrap, seed=seed
        ),
        "round_stage": group_metrics_with_intervals(
            analysis, "round_stage", n_bootstrap=n_bootstrap, seed=seed
        ),
        "equipment_band": group_metrics_with_intervals(
            analysis, "equipment_band", n_bootstrap=n_bootstrap, seed=seed
        ),
    }
    gap = source_auc_gap(analysis, n_bootstrap=n_bootstrap, seed=seed)

    all_errors = enrich_error_cases(
        select_high_confidence_errors(
            analysis, minimum_confidence=0.8, max_cases=None
        )
    )
    reviewed = all_errors.head(review_cases).copy()
    pre_round_counts = reviewed["pre_round_pattern"].value_counts().rename_axis("pattern")
    outcome_counts = reviewed["outcome_pattern"].value_counts().rename_axis("pattern")
    error_summary = pd.concat(
        [
            pre_round_counts.rename("cases").reset_index().assign(pattern_type="pre_round"),
            outcome_counts.rename("cases").reset_index().assign(pattern_type="outcome"),
        ],
        ignore_index=True,
    )

    large_maps = grouped["map"][grouped["map"]["rounds"].ge(300)]
    large_map_min_auc = float(large_maps["auc"].min())
    large_map_min_auc_ci_lower = float(large_maps["auc_ci_lower_95"].min())
    reviewed_pre_round_counts = {
        str(key): int(value)
        for key, value in reviewed["pre_round_pattern"].value_counts().items()
    }
    reviewed_outcome_counts = {
        str(key): int(value)
        for key, value in reviewed["outcome_pattern"].value_counts().items()
    }
    all_outcome_counts = {
        str(key): int(value)
        for key, value in all_errors["outcome_pattern"].value_counts().items()
    }
    summary = {
        "task": "pre_round",
        "bootstrap_unit": "series_id within each group",
        "bootstrap_samples": n_bootstrap,
        "seed": seed,
        "source_auc_gap": gap,
        "source_gap_passed": bool(gap["absolute_difference"] <= 0.04),
        "large_map_count": int(len(large_maps)),
        "large_map_min_auc": large_map_min_auc,
        "large_map_min_auc_ci_lower": large_map_min_auc_ci_lower,
        "large_map_minimum_passed": bool(large_map_min_auc >= 0.67),
        "large_map_target_passed": bool(large_map_min_auc >= 0.69),
        "all_large_map_ci_lower_minimum_passed": bool(
            large_map_min_auc_ci_lower >= 0.67
        ),
        "high_confidence_definition": "assigned predicted-side probability >= 0.80 and wrong",
        "high_confidence_errors_available": int(len(all_errors)),
        "reviewed_error_cases": int(len(reviewed)),
        "error_review_passed": bool(len(reviewed) >= 30),
        "reviewed_pre_round_pattern_counts": reviewed_pre_round_counts,
        "reviewed_outcome_pattern_counts": reviewed_outcome_counts,
        "all_high_confidence_outcome_pattern_counts": all_outcome_counts,
        "first_kill_usage": "post-hoc diagnosis only; never a pre-round model feature",
    }

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, table in grouped.items():
        table.to_csv(report_dir / f"metrics_by_{name}_with_ci.csv", index=False)
    pd.DataFrame([gap]).to_csv(report_dir / "source_auc_gap.csv", index=False)
    all_errors.to_csv(report_dir / "all_high_confidence_errors.csv", index=False)
    reviewed.to_csv(report_dir / "reviewed_top30_errors.csv", index=False)
    error_summary.to_csv(report_dir / "error_pattern_summary.csv", index=False)
    with (report_dir / "m11_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_case_review(reviewed, report_dir / "top30_error_review.md")
    write_report(summary, report_dir / "m11_robustness_report.md")
    save_plots(grouped["map"], error_summary, report_dir)
    return grouped, reviewed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M11 grouped robustness analysis.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--kills", required=True)
    parser.add_argument("--report-dir", default=str(REPORT_DIR / "m11"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--review-cases", type=int, default=30)
    args = parser.parse_args()

    grouped, _, summary = run(
        args.predictions,
        args.data,
        args.kills,
        args.report_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        review_cases=args.review_cases,
    )
    print(grouped["source"].round(6).to_string(index=False))
    print(grouped["map"][["map_name", "rounds", "series", "auc", "auc_ci_lower_95", "auc_ci_upper_95"]].round(6).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

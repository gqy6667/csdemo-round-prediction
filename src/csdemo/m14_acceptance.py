from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


METRIC_TARGETS = {
    "auc": {"minimum": 0.70, "stage": 0.73, "higher_is_better": True},
    "log_loss": {"minimum": 0.61, "stage": 0.58, "higher_is_better": False},
    "accuracy": {"minimum": 0.64, "stage": 0.66, "higher_is_better": True},
    "brier_score": {"minimum": 0.21, "stage": 0.195, "higher_is_better": False},
}

BLOCKING_CHECKS = (
    "required_artifacts",
    "data_identity",
    "quality_gate",
    "split_contract",
    "minimum_metrics",
    "generalization_gap",
    "calibration",
    "robustness",
    "explanation",
    "prediction_interface",
    "automated_tests",
    "reproduction_entrypoint",
)


def assess_metric_targets(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Separate phase-completion minimums from aspirational stage targets."""

    results: dict[str, dict[str, Any]] = {}
    for name, target in METRIC_TARGETS.items():
        if name not in metrics:
            raise KeyError(f"Missing required metric: {name}")
        value = float(metrics[name])
        higher_is_better = bool(target["higher_is_better"])
        minimum = float(target["minimum"])
        stage = float(target["stage"])
        minimum_passed = value >= minimum if higher_is_better else value <= minimum
        stage_passed = value >= stage if higher_is_better else value <= stage
        stage_gap = max(0.0, stage - value) if higher_is_better else max(0.0, value - stage)
        results[name] = {
            "value": value,
            "minimum": minimum,
            "stage_target": stage,
            "higher_is_better": higher_is_better,
            "minimum_passed": minimum_passed,
            "stage_passed": stage_passed,
            "stage_gap": stage_gap,
        }

    minimum_count = sum(item["minimum_passed"] for item in results.values())
    stage_count = sum(item["stage_passed"] for item in results.values())
    return {
        "metrics": results,
        "minimum_passed_count": minimum_count,
        "stage_passed_count": stage_count,
        "all_minimum_passed": minimum_count == len(results),
        "all_stage_passed": stage_count == len(results),
    }


def audit_split_contract(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit unique IDs and series-level train/validation/test isolation."""

    required = {"series_id", "game_id", "round_id", "split"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return {
            "passed": False,
            "errors": ["missing columns: " + ", ".join(missing_columns)],
            "missing_columns": missing_columns,
        }

    errors: list[str] = []
    duplicate_round_ids = int(frame["round_id"].value_counts().gt(1).sum())
    cross_split_series = int(frame.groupby("series_id")["split"].nunique().gt(1).sum())
    cross_split_games = int(frame.groupby("game_id")["split"].nunique().gt(1).sum())
    cross_split_rounds = int(frame.groupby("round_id")["split"].nunique().gt(1).sum())
    missing_id_rows = int(frame[["series_id", "game_id", "round_id"]].isna().any(axis=1).sum())
    expected_splits = {"train", "val", "test"}
    observed_splits = set(frame["split"].dropna().astype(str))

    if duplicate_round_ids:
        errors.append("round_id is not unique")
    if cross_split_series:
        errors.append("series_id appears in multiple splits")
    if cross_split_games:
        errors.append("game_id appears in multiple splits")
    if cross_split_rounds:
        errors.append("round_id appears in multiple splits")
    if missing_id_rows:
        errors.append("identifier columns contain missing values")
    if observed_splits != expected_splits:
        errors.append(
            f"split values must be {sorted(expected_splits)}; got {sorted(observed_splits)}"
        )

    series_counts = {
        split: int(frame.loc[frame["split"].eq(split), "series_id"].nunique())
        for split in ("train", "val", "test")
    }
    row_counts = {
        split: int(frame["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    return {
        "passed": not errors,
        "errors": errors,
        "rows": int(len(frame)),
        "series": int(frame["series_id"].nunique()),
        "series_counts": series_counts,
        "row_counts": row_counts,
        "duplicate_round_ids": duplicate_round_ids,
        "cross_split_series": cross_split_series,
        "cross_split_games": cross_split_games,
        "cross_split_rounds": cross_split_rounds,
        "missing_id_rows": missing_id_rows,
    }


def audit_quality_summary(summary: pd.DataFrame) -> dict[str, Any]:
    """Treat informational findings as non-blocking and warnings/errors as blockers."""

    required = {"severity", "count"}
    missing = sorted(required - set(summary.columns))
    if missing:
        return {
            "passed": False,
            "errors": ["missing columns: " + ", ".join(missing)],
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
        }

    severities = summary["severity"].astype(str).str.lower()
    counts = pd.to_numeric(summary["count"], errors="coerce").fillna(0).astype(int)
    totals = {
        severity: int(counts[severities.eq(severity)].sum())
        for severity in ("error", "warning", "info")
    }
    return {
        "passed": totals["error"] == 0 and totals["warning"] == 0,
        "errors": [],
        "error_count": totals["error"],
        "warning_count": totals["warning"],
        "info_count": totals["info"],
    }


def decide_phase_readiness(checks: Mapping[str, bool]) -> dict[str, Any]:
    """Decide whether phase 1 can close and phase 2 may start."""

    missing = [name for name in BLOCKING_CHECKS if name not in checks]
    if missing:
        raise KeyError("Missing blocking checks: " + ", ".join(missing))
    failures = [name for name in BLOCKING_CHECKS if not bool(checks[name])]
    return {
        "status": "passed" if not failures else "failed",
        "phase_1_pre_round_xgboost_complete": not failures,
        "ready_for_first_kill_xgboost": not failures,
        "blocking_failures": failures,
    }

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import ID_COLUMNS


ROUND_REQUIRED_COLUMNS = ID_COLUMNS + [
    "source_subset",
    "map_name",
    "round_num",
    "ct_score",
    "t_score",
    "ct_win",
    "ct_alive",
    "t_alive",
]
KILL_REQUIRED_COLUMNS = ID_COLUMNS + ["time", "killer_side", "victim_side"]
TEAM_COUNT_COLUMNS = [
    "ct_armor",
    "t_armor",
    "ct_helmets",
    "t_helmets",
    "ct_defuse_kits",
    "t_defuse_kits",
]
WEAPON_COUNT_COLUMNS = [
    "ct_ak47",
    "t_ak47",
    "ct_m4a4",
    "t_m4a4",
    "ct_m4a1_s",
    "t_m4a1_s",
    "ct_awp",
    "t_awp",
    "ct_rifles",
    "t_rifles",
    "ct_smgs",
    "t_smgs",
]


def _context_columns(df: pd.DataFrame) -> list[str]:
    preferred = ID_COLUMNS + [
        "source_subset",
        "map_name",
        "round_num",
        "ct_score",
        "t_score",
        "ct_alive",
        "t_alive",
        "ct_rifles",
        "t_rifles",
        "ct_grenades",
        "t_grenades",
        "ct_win",
        "time",
        "killer_side",
        "victim_side",
    ]
    return [column for column in preferred if column in df.columns]


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return sorted(set(required) - set(df.columns))


def evaluate_quality(rounds: pd.DataFrame, kills: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return non-zero quality checks and example rows without modifying source data."""
    summary_rows: list[dict[str, object]] = []
    example_frames: list[pd.DataFrame] = []

    def add_issue(check: str, severity: str, mask: pd.Series, frame: pd.DataFrame) -> None:
        count = int(mask.sum())
        if count == 0:
            return
        summary_rows.append({"severity": severity, "check": check, "count": count})
        examples = frame.loc[mask, _context_columns(frame)].copy()
        examples.insert(0, "check", check)
        examples.insert(0, "severity", severity)
        example_frames.append(examples)

    missing_round_columns = _missing_columns(rounds, ROUND_REQUIRED_COLUMNS)
    if missing_round_columns:
        summary_rows.append(
            {
                "severity": "error",
                "check": "missing_round_columns",
                "count": len(missing_round_columns),
                "details": ",".join(missing_round_columns),
            }
        )
    else:
        add_issue("missing_round_core_value", "error", rounds[ROUND_REQUIRED_COLUMNS].isna().any(axis=1), rounds)
        add_issue("duplicate_round_identity", "error", rounds.duplicated(ID_COLUMNS, keep=False), rounds)
        add_issue("invalid_ct_win", "error", ~rounds["ct_win"].isin([0, 1]), rounds)
        add_issue("invalid_round_num", "error", rounds["round_num"] <= 0, rounds)

        numeric_columns = rounds.select_dtypes(include="number").columns.tolist()
        add_issue(
            "negative_round_numeric_value",
            "error",
            (rounds[numeric_columns] < 0).any(axis=1),
            rounds,
        )
        alive_out_of_range = (
            (rounds["ct_alive"] < 0)
            | (rounds["ct_alive"] > 5)
            | (rounds["t_alive"] < 0)
            | (rounds["t_alive"] > 5)
        )
        add_issue("alive_count_out_of_range", "error", alive_out_of_range, rounds)
        add_issue(
            "pre_combat_alive_not_five",
            "warning",
            (rounds["ct_alive"] != 5) | (rounds["t_alive"] != 5),
            rounds,
        )
        add_issue(
            "score_round_relation_mismatch",
            "warning",
            (rounds["ct_score"] + rounds["t_score"]) != (rounds["round_num"] - 1),
            rounds,
        )

        for column in [column for column in TEAM_COUNT_COLUMNS if column in rounds.columns]:
            add_issue(f"{column}_out_of_range", "error", ~rounds[column].between(0, 5), rounds)
        for column in [column for column in WEAPON_COUNT_COLUMNS if column in rounds.columns]:
            add_issue(f"{column}_negative", "error", rounds[column] < 0, rounds)
        weapon_columns = [column for column in WEAPON_COUNT_COLUMNS if column in rounds.columns]
        add_issue(
            "weapon_count_exceeds_five",
            "warning",
            (rounds[weapon_columns] > 5).any(axis=1),
            rounds,
        )
        grenade_columns = [column for column in ["ct_grenades", "t_grenades"] if column in rounds.columns]
        add_issue(
            "grenade_count_exceeds_twenty",
            "warning",
            (rounds[grenade_columns] > 20).any(axis=1),
            rounds,
        )

    missing_kill_columns = _missing_columns(kills, KILL_REQUIRED_COLUMNS)
    if missing_kill_columns:
        summary_rows.append(
            {
                "severity": "error",
                "check": "missing_kill_columns",
                "count": len(missing_kill_columns),
                "details": ",".join(missing_kill_columns),
            }
        )
    else:
        add_issue("missing_kill_core_value", "error", kills[KILL_REQUIRED_COLUMNS].isna().any(axis=1), kills)
        add_issue("negative_kill_time", "error", kills["time"] < 0, kills)
        add_issue(
            "invalid_kill_side",
            "error",
            ~kills["killer_side"].str.upper().isin(["CT", "T"])
            | ~kills["victim_side"].str.upper().isin(["CT", "T"]),
            kills,
        )

    if not missing_round_columns and not missing_kill_columns:
        round_keys = rounds[ID_COLUMNS].drop_duplicates()
        kill_keys = kills[ID_COLUMNS].drop_duplicates()
        orphan_keys = kill_keys.merge(round_keys, on=ID_COLUMNS, how="left", indicator=True)
        orphan_keys = orphan_keys[orphan_keys["_merge"].eq("left_only")]
        if not orphan_keys.empty:
            orphan_index = pd.MultiIndex.from_frame(orphan_keys[ID_COLUMNS])
            kill_index = pd.MultiIndex.from_frame(kills[ID_COLUMNS])
            add_issue(
                "orphan_kill_round",
                "error",
                pd.Series(kill_index.isin(orphan_index), index=kills.index),
                kills,
            )

        rounds_without_kills = round_keys.merge(kill_keys, on=ID_COLUMNS, how="left", indicator=True)
        missing_kill_keys = rounds_without_kills[rounds_without_kills["_merge"].eq("left_only")]
        if not missing_kill_keys.empty:
            missing_index = pd.MultiIndex.from_frame(missing_kill_keys[ID_COLUMNS])
            round_index = pd.MultiIndex.from_frame(rounds[ID_COLUMNS])
            add_issue(
                "round_without_valid_kill",
                "info",
                pd.Series(round_index.isin(missing_index), index=rounds.index),
                rounds,
            )

    summary = pd.DataFrame(summary_rows, columns=["severity", "check", "count", "details"])
    examples = (
        pd.concat(example_frames, ignore_index=True, sort=False)
        if example_frames
        else pd.DataFrame(columns=["severity", "check"] + _context_columns(rounds))
    )
    return summary, examples


def raise_for_errors(summary: pd.DataFrame) -> None:
    errors = summary[summary["severity"].eq("error")]
    if errors.empty:
        return
    details = ", ".join(f"{row.check}={row.count}" for row in errors.itertuples())
    raise ValueError(f"Data quality errors: {details}")


def write_quality_report(summary: pd.DataFrame, examples: pd.DataFrame, report_dir: str | Path) -> None:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "quality_summary.csv", index=False)
    examples.to_csv(report_dir / "quality_examples.csv", index=False)

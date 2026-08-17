from __future__ import annotations

import pandas as pd

from .config import LABEL_COL
from .schema import FIRST_KILL_FEATURES, ID_COLUMNS, PRE_ROUND_FEATURES


def _safe_diff(df: pd.DataFrame, left: str, right: str, out: str) -> None:
    if left in df.columns and right in df.columns and out not in df.columns:
        df[out] = df[left].fillna(0) - df[right].fillna(0)


def add_common_diffs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("ct_score", "t_score", "score_diff_ct"),
        ("ct_alive", "t_alive", "alive_diff_ct"),
        ("ct_eq_value", "t_eq_value", "eq_value_diff_ct"),
        ("ct_cash", "t_cash", "cash_diff_ct"),
        ("ct_armor", "t_armor", "armor_diff_ct"),
        ("ct_helmets", "t_helmets", "helmet_diff_ct"),
        ("ct_awp", "t_awp", "awp_diff_ct"),
        ("ct_rifles", "t_rifles", "rifle_diff_ct"),
        ("ct_smgs", "t_smgs", "smg_diff_ct"),
        ("ct_grenades", "t_grenades", "grenade_diff_ct"),
    ]
    for left, right, name in pairs:
        _safe_diff(out, left, right, name)
    return out


def make_pre_round_samples(rounds: pd.DataFrame) -> pd.DataFrame:
    required = set(ID_COLUMNS + [LABEL_COL])
    missing = sorted(required - set(rounds.columns))
    if missing:
        raise KeyError(f"Missing required round columns: {missing}")

    out = add_common_diffs(rounds)
    if out.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate round identity: series_id, game_id, and round_id must be unique")
    cols = ID_COLUMNS + [LABEL_COL] + [c for c in PRE_ROUND_FEATURES if c in out.columns]
    return out[cols].copy()


def make_first_kill_samples(rounds: pd.DataFrame, kills: pd.DataFrame) -> pd.DataFrame:
    required_kill_cols = set(ID_COLUMNS + ["time", "killer_side", "victim_side"])
    missing = sorted(required_kill_cols - set(kills.columns))
    if missing:
        raise KeyError(f"Missing required kill columns: {missing}")

    base = make_pre_round_samples(rounds)
    # All identity columns are required so kills cannot cross between maps in one series.
    first_kills = (
        kills.sort_values(ID_COLUMNS + ["time"])
        .groupby(ID_COLUMNS, as_index=False)
        .first()
    )

    fk = first_kills.rename(
        columns={
            "time": "first_kill_time",
            "weapon": "first_kill_weapon",
            "headshot": "first_kill_headshot",
        }
    )
    fk["first_kill_is_ct"] = fk["killer_side"].str.upper().eq("CT").astype(int)
    fk["first_death_is_ct"] = fk["victim_side"].str.upper().eq("CT").astype(int)
    fk["first_kill_advantage_ct"] = fk["first_kill_is_ct"] - fk["first_death_is_ct"]

    samples = base.merge(
        fk,
        on=ID_COLUMNS,
        how="inner",
        suffixes=("", "_kill"),
        validate="one_to_one",
    )

    if "ct_alive" in samples.columns and "t_alive" in samples.columns:
        samples["ct_alive_after_fk"] = samples["ct_alive"] - samples["first_death_is_ct"]
        samples["t_alive_after_fk"] = samples["t_alive"] - (1 - samples["first_death_is_ct"])
        samples["alive_diff_ct_after_fk"] = (
            samples["ct_alive_after_fk"] - samples["t_alive_after_fk"]
        )

    keep = ID_COLUMNS + [LABEL_COL] + [c for c in FIRST_KILL_FEATURES if c in samples.columns]
    return samples[keep].copy()

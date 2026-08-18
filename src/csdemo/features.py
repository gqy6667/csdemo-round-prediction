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


def select_first_valid_kills(kills: pd.DataFrame) -> pd.DataFrame:
    """Select the earliest enemy kill by demo tick within each repaired round key."""

    required = set(ID_COLUMNS + ["tick", "time", "killer_side", "victim_side"])
    missing = sorted(required - set(kills.columns))
    if missing:
        raise KeyError(f"Missing required kill columns: {missing}")

    identity_and_order = ID_COLUMNS + ["tick"]
    if kills[identity_and_order].isna().any().any():
        raise ValueError("Kill identity columns and tick must not be null")

    valid = kills.copy()
    killer_side = valid["killer_side"].astype("string").str.upper()
    victim_side = valid["victim_side"].astype("string").str.upper()
    enemy_kill = (
        killer_side.isin(["CT", "T"])
        & victim_side.isin(["CT", "T"])
        & killer_side.ne(victim_side)
    )
    valid = valid.loc[enemy_kill].copy()
    valid["_source_order"] = range(len(valid))

    ordered = valid.sort_values(
        ID_COLUMNS + ["tick", "_source_order"], kind="mergesort"
    )
    selected = ordered.groupby(ID_COLUMNS, sort=False, as_index=False).head(1)
    return selected.drop(columns="_source_order").reset_index(drop=True)


def make_first_kill_samples(rounds: pd.DataFrame, kills: pd.DataFrame) -> pd.DataFrame:
    required_round_cols = set(ID_COLUMNS + ["ct_alive", "t_alive"])
    missing_round_cols = sorted(required_round_cols - set(rounds.columns))
    if missing_round_cols:
        raise KeyError(f"Missing required first-kill round columns: {missing_round_cols}")
    if (~rounds["ct_alive"].eq(5) | ~rounds["t_alive"].eq(5)).any():
        raise ValueError("First-kill samples require a 5v5 pre-combat snapshot")

    base = make_pre_round_samples(rounds)
    alive_before = rounds[ID_COLUMNS + ["ct_alive", "t_alive"]].copy()
    base = base.merge(
        alive_before,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    # Full identity and tick ordering prevent events crossing maps or time resets.
    first_kills = select_first_valid_kills(kills)

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

    samples["ct_alive_after_fk"] = samples["ct_alive"] - samples["first_death_is_ct"]
    samples["t_alive_after_fk"] = samples["t_alive"] - (1 - samples["first_death_is_ct"])
    samples["alive_diff_ct_after_fk"] = (
        samples["ct_alive_after_fk"] - samples["t_alive_after_fk"]
    )

    keep = ID_COLUMNS + [LABEL_COL] + [c for c in FIRST_KILL_FEATURES if c in samples.columns]
    return samples[keep].copy()

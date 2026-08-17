from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GROUP_COL, RANDOM_STATE, SPLIT_RATIOS


def add_group_split(
    df: pd.DataFrame,
    group_col: str = GROUP_COL,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Assign train/val/test by series so maps from one series do not leak."""
    if group_col not in df.columns:
        raise KeyError(f"Missing split group column: {group_col}")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    out = df.copy()
    groups = pd.Series(out[group_col].dropna().unique())
    groups = groups.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_groups = len(groups)
    train_end = int(n_groups * ratios[0])
    val_end = train_end + int(n_groups * ratios[1])

    split_map = {}
    split_map.update({g: "train" for g in groups.iloc[:train_end]})
    split_map.update({g: "val" for g in groups.iloc[train_end:val_end]})
    split_map.update({g: "test" for g in groups.iloc[val_end:]})

    out["split"] = out[group_col].map(split_map)
    return out

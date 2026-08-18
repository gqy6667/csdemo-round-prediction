from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import LABEL_COL
from .features import make_first_kill_samples, select_first_valid_kills
from .io import read_table, write_table
from .schema import FIRST_KILL_EXTRA_FEATURES, ID_COLUMNS


SPLIT_ORDER = ["train", "val", "test"]
EVENT_COLUMNS = [
    "first_kill_time",
    "first_kill_is_ct",
    "first_death_is_ct",
    "first_kill_headshot",
    "first_kill_weapon",
    "first_kill_advantage_ct",
]


def apply_split_manifest(samples: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach the exact M14 series split instead of drawing a new split."""

    required = {"series_id", "split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Missing split manifest columns: {missing}")
    if manifest["series_id"].isna().any():
        raise ValueError("M14 split manifest contains a null series_id")
    if manifest["series_id"].duplicated().any():
        raise ValueError("M14 split manifest must contain one row per series_id")
    invalid_splits = sorted(set(manifest["split"].dropna()) - set(SPLIT_ORDER))
    if invalid_splits:
        raise ValueError(f"M14 split manifest contains invalid splits: {invalid_splits}")

    out = samples.drop(columns="split", errors="ignore").copy()
    out = out.merge(
        manifest[["series_id", "split"]],
        on="series_id",
        how="left",
        validate="many_to_one",
    )
    missing_series = sorted(out.loc[out["split"].isna(), "series_id"].unique())
    if missing_series:
        preview = ", ".join(str(value) for value in missing_series[:5])
        raise ValueError(f"Series missing from the M14 split manifest: {preview}")
    return out


def _null_safe_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.eq(right) | (left.isna() & right.isna())).fillna(False)


def _expected_events(selected: pd.DataFrame) -> pd.DataFrame:
    expected = selected.rename(
        columns={
            "time": "first_kill_time",
            "headshot": "first_kill_headshot",
            "weapon": "first_kill_weapon",
        }
    ).copy()
    expected["first_kill_is_ct"] = (
        expected["killer_side"].astype("string").str.upper().eq("CT").astype(int)
    )
    expected["first_death_is_ct"] = (
        expected["victim_side"].astype("string").str.upper().eq("CT").astype(int)
    )
    expected["first_kill_advantage_ct"] = (
        expected["first_kill_is_ct"] - expected["first_death_is_ct"]
    )
    return expected[ID_COLUMNS + EVENT_COLUMNS]


def _excluded_rounds(rounds: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    context = ID_COLUMNS + [
        column
        for column in ("source_subset", "map_name", "round_num")
        if column in rounds.columns
    ]
    round_keys = rounds[context].drop_duplicates(ID_COLUMNS)
    selected_keys = selected[ID_COLUMNS].drop_duplicates()
    excluded = round_keys.merge(selected_keys, on=ID_COLUMNS, how="left", indicator=True)
    excluded = excluded.loc[excluded["_merge"].eq("left_only")].drop(columns="_merge")
    excluded["reason"] = "no_valid_enemy_kill"
    return excluded.reset_index(drop=True)


def audit_first_kill_data(
    rounds: pd.DataFrame,
    kills: pd.DataFrame,
    samples: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the blocking M15 data, identity, event, and split contracts."""

    checks: dict[str, dict[str, Any]] = {}

    def add_check(name: str, passed: bool, **details: Any) -> None:
        checks[name] = {"passed": bool(passed), "blocking": True, **details}

    required_round = set(ID_COLUMNS + [LABEL_COL, "ct_alive", "t_alive"])
    required_kill = set(
        ID_COLUMNS
        + ["tick", "time", "killer_side", "victim_side", "weapon", "headshot"]
    )
    required_sample = set(
        ID_COLUMNS + [LABEL_COL, "split"] + FIRST_KILL_EXTRA_FEATURES
    )
    missing_round = sorted(required_round - set(rounds.columns))
    missing_kill = sorted(required_kill - set(kills.columns))
    missing_sample = sorted(required_sample - set(samples.columns))
    missing_manifest = sorted({"series_id", "split"} - set(manifest.columns))
    schema_ok = not (missing_round or missing_kill or missing_sample or missing_manifest)
    add_check(
        "required_columns",
        schema_ok,
        missing_round_columns=missing_round,
        missing_kill_columns=missing_kill,
        missing_sample_columns=missing_sample,
        missing_manifest_columns=missing_manifest,
    )
    if not schema_ok:
        counts = {
            "round_rows": int(len(rounds)),
            "kill_rows": int(len(kills)),
            "sample_rows": int(len(samples)),
            "excluded_rounds": 0,
        }
        summary = {
            "stage": "M15",
            "passed": False,
            "checks": checks,
            "counts": counts,
            "diagnostics": {},
        }
        return summary, pd.DataFrame(columns=ID_COLUMNS + ["reason"])

    selected = select_first_valid_kills(kills)
    expected = _expected_events(selected)
    excluded = _excluded_rounds(rounds, selected)

    round_key_duplicates = int(rounds.duplicated(ID_COLUMNS).sum())
    sample_key_duplicates = int(samples.duplicated(ID_COLUMNS).sum())
    selected_key_duplicates = int(selected.duplicated(ID_COLUMNS).sum())
    add_check(
        "unique_repaired_key",
        round_key_duplicates == sample_key_duplicates == selected_key_duplicates == 0,
        round_duplicate_rows=round_key_duplicates,
        sample_duplicate_rows=sample_key_duplicates,
        selected_event_duplicate_rows=selected_key_duplicates,
    )

    sample_keys = samples[ID_COLUMNS].drop_duplicates()
    selected_keys = selected[ID_COLUMNS].drop_duplicates()
    round_keys = rounds[ID_COLUMNS].drop_duplicates()
    coverage = sample_keys.merge(selected_keys, on=ID_COLUMNS, how="outer", indicator=True)
    coverage_mismatches = int(coverage["_merge"].ne("both").sum())
    orphan_samples = sample_keys.merge(
        round_keys, on=ID_COLUMNS, how="left", indicator=True
    )
    orphan_sample_rows = int(orphan_samples["_merge"].eq("left_only").sum())
    add_check(
        "sample_coverage",
        coverage_mismatches == 0 and orphan_sample_rows == 0,
        selected_event_rows=int(len(selected_keys)),
        sample_key_rows=int(len(sample_keys)),
        coverage_mismatch_rows=coverage_mismatches,
        orphan_sample_rows=orphan_sample_rows,
    )

    actual_events = samples[ID_COLUMNS + EVENT_COLUMNS].drop_duplicates(ID_COLUMNS)
    linked = actual_events.merge(
        expected,
        on=ID_COLUMNS,
        how="outer",
        suffixes=("_actual", "_expected"),
        indicator=True,
    )
    event_mismatch = linked["_merge"].ne("both")
    mismatch_by_field: dict[str, int] = {}
    for column in EVENT_COLUMNS:
        equal = _null_safe_equal(
            linked[f"{column}_actual"], linked[f"{column}_expected"]
        )
        mismatch_by_field[column] = int((~equal & linked["_merge"].eq("both")).sum())
        event_mismatch |= ~equal
    event_mismatch_rows = int(event_mismatch.sum())
    add_check(
        "event_linkage",
        event_mismatch_rows == 0,
        mismatch_rows=event_mismatch_rows,
        mismatch_by_field=mismatch_by_field,
    )

    core_columns = ID_COLUMNS + [LABEL_COL, "split"] + FIRST_KILL_EXTRA_FEATURES
    core_null_cells = int(samples[core_columns].isna().sum().sum())
    add_check("core_values_present", core_null_cells == 0, null_cells=core_null_cells)

    initial_5v5_rows = int((rounds["ct_alive"].eq(5) & rounds["t_alive"].eq(5)).sum())
    add_check(
        "initial_5v5_state",
        initial_5v5_rows == len(rounds),
        valid_rows=initial_5v5_rows,
        total_rows=int(len(rounds)),
    )

    alive_pair = list(zip(samples["ct_alive_after_fk"], samples["t_alive_after_fk"]))
    invalid_alive_rows = sum(pair not in {(5, 4), (4, 5)} for pair in alive_pair)
    side_complement = samples["first_kill_is_ct"] + samples["first_death_is_ct"]
    advantage_expected = samples["first_kill_is_ct"] - samples["first_death_is_ct"]
    alive_diff_expected = samples["ct_alive_after_fk"] - samples["t_alive_after_fk"]
    derived_mismatch = (
        side_complement.ne(1)
        | samples["first_kill_advantage_ct"].ne(advantage_expected)
        | samples["alive_diff_ct_after_fk"].ne(alive_diff_expected)
        | samples["alive_diff_ct_after_fk"].ne(samples["first_kill_advantage_ct"])
    )
    derived_mismatch_rows = int(derived_mismatch.sum())
    add_check(
        "post_kill_state",
        invalid_alive_rows == 0 and derived_mismatch_rows == 0,
        invalid_alive_rows=int(invalid_alive_rows),
        derived_mismatch_rows=derived_mismatch_rows,
    )

    labels = samples[ID_COLUMNS + [LABEL_COL]].merge(
        rounds[ID_COLUMNS + [LABEL_COL]],
        on=ID_COLUMNS,
        how="left",
        suffixes=("_sample", "_round"),
        validate="one_to_one",
    )
    label_mismatch_rows = int(
        (~_null_safe_equal(labels[f"{LABEL_COL}_sample"], labels[f"{LABEL_COL}_round"])).sum()
    )
    add_check("label_linkage", label_mismatch_rows == 0, mismatch_rows=label_mismatch_rows)

    manifest_duplicates = int(manifest["series_id"].duplicated().sum())
    split_join = samples[["series_id", "split"]].merge(
        manifest[["series_id", "split"]],
        on="series_id",
        how="left",
        suffixes=("_sample", "_manifest"),
    )
    split_mismatch = (
        split_join["split_manifest"].isna()
        | split_join["split_sample"].ne(split_join["split_manifest"])
    )
    split_mismatch_rows = int(split_mismatch.sum())
    sample_series = set(samples["series_id"].dropna().unique())
    manifest_series = set(manifest["series_id"].dropna().unique())
    missing_manifest_series = len(sample_series - manifest_series)
    unused_manifest_series = len(manifest_series - sample_series)
    add_check(
        "split_manifest",
        manifest_duplicates == 0
        and split_mismatch_rows == 0
        and missing_manifest_series == 0
        and unused_manifest_series == 0,
        manifest_duplicate_rows=manifest_duplicates,
        mismatch_rows=split_mismatch_rows,
        sample_series_missing_from_manifest=missing_manifest_series,
        manifest_series_without_sample=unused_manifest_series,
    )

    cross_split_series = int((samples.groupby("series_id")["split"].nunique() > 1).sum())
    cross_split_games = int((samples.groupby("game_id")["split"].nunique() > 1).sum())
    cross_split_rounds = int((samples.groupby("round_id")["split"].nunique() > 1).sum())
    add_check(
        "split_isolation",
        cross_split_series == cross_split_games == cross_split_rounds == 0,
        cross_split_series=cross_split_series,
        cross_split_games=cross_split_games,
        cross_split_rounds=cross_split_rounds,
    )

    valid_sides = (
        kills["killer_side"].astype("string").str.upper().isin(["CT", "T"])
        & kills["victim_side"].astype("string").str.upper().isin(["CT", "T"])
        & kills["killer_side"]
        .astype("string")
        .str.upper()
        .ne(kills["victim_side"].astype("string").str.upper())
    )
    invalid_normalized_kills = int((~valid_sides).sum())
    add_check(
        "normalized_kill_sides",
        invalid_normalized_kills == 0,
        invalid_rows=invalid_normalized_kills,
    )

    valid_kills = kills.loc[valid_sides].copy()
    valid_kills["_source_order"] = range(len(valid_kills))
    by_time = (
        valid_kills.sort_values(
            ID_COLUMNS + ["time", "tick", "_source_order"], kind="mergesort"
        )
        .groupby(ID_COLUMNS, sort=False, as_index=False)
        .head(1)
    )
    time_vs_tick = selected[ID_COLUMNS + ["tick"]].merge(
        by_time[ID_COLUMNS + ["tick"]],
        on=ID_COLUMNS,
        suffixes=("_tick_order", "_time_order"),
        validate="one_to_one",
    )
    time_order_disagreements = int(
        time_vs_tick["tick_tick_order"].ne(time_vs_tick["tick_time_order"]).sum()
    )
    nonpositive_first_kill_time = int(samples["first_kill_time"].le(0).sum())

    selected_without_source_flag = None
    flagged_not_earliest_tick = None
    if "is_first_kill" in kills.columns:
        selected_without_source_flag = int(selected["is_first_kill"].ne(1).sum())
        flagged = valid_kills.loc[valid_kills["is_first_kill"].eq(1)]
        flagged_comparison = flagged[ID_COLUMNS + ["tick"]].merge(
            selected[ID_COLUMNS + ["tick"]],
            on=ID_COLUMNS,
            how="left",
            suffixes=("_flag", "_selected"),
        )
        missing_selected = flagged_comparison["tick_selected"].isna()
        wrong_selected = (
            ~missing_selected
            & flagged_comparison["tick_flag"].ne(flagged_comparison["tick_selected"])
        )
        flagged_not_earliest_tick = int((missing_selected | wrong_selected).sum())

    counts = {
        "round_rows": int(len(rounds)),
        "kill_rows": int(len(kills)),
        "sample_rows": int(len(samples)),
        "excluded_rounds": int(len(excluded)),
        "series": int(samples["series_id"].nunique()),
        "games": int(samples["game_id"].nunique()),
    }
    diagnostics = {
        "time_order_disagreements": time_order_disagreements,
        "nonpositive_first_kill_time": nonpositive_first_kill_time,
        "selected_without_source_first_kill_flag": selected_without_source_flag,
        "source_flags_not_at_earliest_tick": flagged_not_earliest_tick,
    }
    passed = all(check["passed"] for check in checks.values() if check["blocking"])
    summary = {
        "stage": "M15",
        "passed": passed,
        "checks": checks,
        "counts": counts,
        "diagnostics": diagnostics,
    }
    return summary, excluded


def compare_previous_samples(
    previous: pd.DataFrame | None, current: pd.DataFrame
) -> dict[str, Any]:
    if previous is None:
        return {"available": False, "reason": "no_previous_first_kill_artifact"}

    result: dict[str, Any] = {
        "available": True,
        "previous_rows": int(len(previous)),
        "current_rows": int(len(current)),
        "previous_duplicate_keys": int(previous.duplicated(ID_COLUMNS).sum()),
        "current_duplicate_keys": int(current.duplicated(ID_COLUMNS).sum()),
        "new_columns": sorted(set(current.columns) - set(previous.columns)),
        "removed_columns": sorted(set(previous.columns) - set(current.columns)),
    }
    previous_keys = previous[ID_COLUMNS].drop_duplicates()
    current_keys = current[ID_COLUMNS].drop_duplicates()
    key_comparison = previous_keys.merge(
        current_keys, on=ID_COLUMNS, how="outer", indicator=True
    )
    result["added_keys"] = int(key_comparison["_merge"].eq("right_only").sum())
    result["removed_keys"] = int(key_comparison["_merge"].eq("left_only").sum())

    comparable = [column for column in EVENT_COLUMNS if column in previous.columns]
    joined = previous[ID_COLUMNS + comparable].merge(
        current[ID_COLUMNS + comparable],
        on=ID_COLUMNS,
        how="inner",
        suffixes=("_previous", "_current"),
        validate="one_to_one",
    )
    changed_any = pd.Series(False, index=joined.index)
    changed_by_field: dict[str, int] = {}
    for column in comparable:
        equal = _null_safe_equal(
            joined[f"{column}_previous"], joined[f"{column}_current"]
        )
        changed = ~equal
        changed_by_field[column] = int(changed.sum())
        changed_any |= changed
    result["matched_keys"] = int(len(joined))
    result["event_changed_rows"] = int(changed_any.sum())
    result["event_changed_fraction"] = (
        float(changed_any.mean()) if len(changed_any) else 0.0
    )
    result["changed_by_field"] = changed_by_field
    return result


def build_split_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        frame = samples.loc[samples["split"].eq(split)]
        rows.append(
            {
                "split": split,
                "series": int(frame["series_id"].nunique()),
                "games": int(frame["game_id"].nunique()),
                "samples": int(len(frame)),
                "series_fraction": float(
                    frame["series_id"].nunique() / samples["series_id"].nunique()
                ),
                "ct_win_rate": float(frame[LABEL_COL].mean()),
            }
        )
    return pd.DataFrame(rows)


def checks_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, check in summary["checks"].items():
        details = {
            key: value
            for key, value in check.items()
            if key not in {"passed", "blocking"}
        }
        rows.append(
            {
                "check": name,
                "passed": check["passed"],
                "blocking": check["blocking"],
                "details": json.dumps(details, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def run_automated_tests(project_root: str | Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    return {
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "command": command,
        "output": output,
    }


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": artifact.as_posix(),
        "bytes": artifact.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def resolve_previous_comparison(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    output_path: str | Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    calculated = compare_previous_samples(previous, current)
    output = Path(output_path)
    existing_summary = Path(report_dir) / "m15_summary.json"
    if not output.is_file() or not existing_summary.is_file():
        return calculated
    if calculated.get("event_changed_rows", 0) or any(
        calculated.get(key)
        for key in ("added_keys", "removed_keys", "new_columns", "removed_columns")
    ):
        return calculated

    with existing_summary.open(encoding="utf-8") as handle:
        saved = json.load(handle)
    saved_hash = saved.get("data_artifact", {}).get("sha256")
    if saved_hash != fingerprint_file(output)["sha256"]:
        return calculated

    preserved = saved.get("previous_dataset_comparison")
    if not isinstance(preserved, dict):
        return calculated
    result = dict(preserved)
    result["preserved_from_first_m15_run"] = True
    return result


def build_external_comparison_status(
    m14_comparison_path: str | Path,
) -> pd.DataFrame:
    columns = [
        "stage",
        "scope",
        "source_title",
        "source_url",
        "metric",
        "ours",
        "external",
        "difference_ours_minus_external",
        "unit",
        "status",
        "comparability",
        "notes",
    ]
    rows: list[dict[str, Any]] = [
        {
            "stage": "M15",
            "scope": "post_first_kill",
            "source_title": "",
            "source_url": "",
            "metric": "",
            "ours": None,
            "external": None,
            "difference_ours_minus_external": None,
            "unit": "",
            "status": "not_applicable_no_model",
            "comparability": "not_available",
            "notes": "M15 repairs and audits data; the first valid model comparison starts in M16.",
        }
    ]
    path = Path(m14_comparison_path)
    if path.is_file():
        previous = pd.read_csv(path)
        closest = previous.loc[previous["comparability"].eq("closest_task")]
        for row in closest.to_dict(orient="records"):
            metric = str(row["metric"])
            unit = "percentage_points" if metric == "accuracy" else "raw_metric"
            difference = row.get("difference_percentage_points")
            if metric != "accuracy" or pd.isna(difference):
                difference = row.get("raw_difference_ours_minus_reported")
            rows.append(
                {
                    "stage": "M14",
                    "scope": "pre_round_carry_forward_only",
                    "source_title": row.get("source_title", ""),
                    "source_url": row.get("source_url", ""),
                    "metric": metric,
                    "ours": row.get("current_value"),
                    "external": row.get("reported_value"),
                    "difference_ours_minus_external": difference,
                    "unit": unit,
                    "status": "carry_forward_not_m15_result",
                    "comparability": row.get("comparability", ""),
                    "notes": "Latest validated pre-round comparison; it is not a post-first-kill benchmark.",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def render_external_comparison(comparison: pd.DataFrame) -> str:
    lines = [
        "# M15 外部模型比较状态",
        "",
        "M15 只修复和验收首杀后数据，没有训练新模型。因此本阶段没有可诚实计算的首杀后",
        "Accuracy、AUC、Log Loss 或 Brier 外部差值；状态为 `not_applicable_no_model`。",
        "",
        "作为连续记录，下面只带入最近一次已验证的 M14 开局前比较：",
        "",
        "| 阶段 | 指标 | 我们 | 外部 | 差值 | 说明 |",
        "|---|---|---:|---:|---:|---|",
    ]
    carry = comparison.loc[comparison["stage"].eq("M14")]
    if carry.empty:
        lines.append("| M14 | - | - | - | - | 未找到 M14 比较产物 |")
    else:
        for row in carry.to_dict(orient="records"):
            metric = row["metric"]
            difference = row["difference_ours_minus_external"]
            if row["unit"] == "percentage_points":
                difference_text = f"{difference:+.2f} 个百分点"
            else:
                difference_text = f"{difference:+.6f}"
            lines.append(
                f"| M14 | {metric} | {row['ours']:.6f} | {row['external']:.6f} | "
                f"{difference_text} | 仅为开局前参考 |"
            )
    lines.extend(
        [
            "",
            "首杀后任务比购买结束任务多看到了击杀信息，不能把上表当成 M15 或 M16 的公平",
            "对比。M16 训练出固定 split 的三个基线后，再新增首杀后正式数值和差距说明。",
            "",
        ]
    )
    return "\n".join(lines)


def render_m15_report(
    summary: dict[str, Any],
    previous: dict[str, Any],
    split_summary: pd.DataFrame,
    artifact: dict[str, Any],
) -> str:
    status = "passed" if summary["passed"] else "failed"
    counts = summary["counts"]
    diagnostics = summary["diagnostics"]
    lines = [
        "# M15 首杀后样本修复与验收报告",
        "",
        "## 阶段决定",
        "",
        f"验收状态：**{status}**。",
        "M15 不训练模型；通过后只表示修复主键和首杀事件的数据可以进入 M16 基线实验。",
        "",
        "## 本次修复",
        "",
        "- 首杀从“`time` 最小”改为同一完整主键内“`tick` 最小”的有效敌方击杀。",
        "- 使用 `series_id + game_id + round_id` 关联，阻止同系列赛不同地图串行。",
        "- 新增首杀后的 CT/T 存活人数和人数差，状态必须为 5v4 或 4v5。",
        "- 直接复用 M14 的 782 个系列赛 split，不重新抽签。",
        "",
        "## 数据结果",
        "",
        f"- 输入回合：{counts['round_rows']:,}；击杀事件：{counts['kill_rows']:,}。",
        f"- 首杀后样本：{counts['sample_rows']:,}；排除无有效敌方击杀回合：{counts['excluded_rounds']:,}。",
        f"- 覆盖系列赛：{counts['series']:,}；地图 demo：{counts['games']:,}。",
        f"- 按秒数与按 tick 选择不一致：{diagnostics['time_order_disagreements']:,} 回合。",
        f"- 最小 tick 事件没有 ESTA 首杀标记：{diagnostics['selected_without_source_first_kill_flag']:,} 回合；使用明确定义的 tick 兜底。",
        f"- 原始首杀标记不在最小 tick：{diagnostics['source_flags_not_at_earliest_tick']:,} 回合。",
        f"- 修复后首杀时间小于等于 0：{diagnostics['nonpositive_first_kill_time']:,} 回合。",
        "",
        "## 修复前后",
        "",
    ]
    if previous.get("available"):
        changed = previous["event_changed_rows"]
        fraction = previous["event_changed_fraction"]
        lines.extend(
            [
                f"旧产物与新产物主键都为 {previous['current_rows']:,} 条；新增主键 "
                f"{previous['added_keys']:,}，移除主键 {previous['removed_keys']:,}。",
                f"首杀事件字段发生变化：**{changed:,} 条（{fraction:.2%}）**。",
                f"新增字段：`{', '.join(previous['new_columns'])}`。",
                "",
                "| 字段 | 变化行数 |",
                "|---|---:|",
            ]
        )
        for column, changed_rows in previous["changed_by_field"].items():
            lines.append(f"| `{column}` | {changed_rows:,} |")
    else:
        lines.append("没有找到修复前产物，因此无法生成逐行前后差异。")

    lines.extend(
        [
            "",
            "## 阻塞检查",
            "",
            "| 检查 | 结果 |",
            "|---|---|",
        ]
    )
    for name, check in summary["checks"].items():
        result = "PASS" if check["passed"] else "FAIL"
        lines.append(f"| `{name}` | {result} |")

    lines.extend(
        [
            "",
            "## 固定切分",
            "",
            "| split | 系列赛 | 地图 | 样本 | 系列赛占比 | CT 胜率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in split_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['split']} | {row['series']:,} | {row['games']:,} | "
            f"{row['samples']:,} | {row['series_fraction']:.2%} | {row['ct_win_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 特征说明",
            "",
            "`first_kill_is_ct`、`first_death_is_ct`、两侧存活人数和优势字段彼此可以确定，",
            "所以它们不是五份独立信息。M16 会固定其他条件，分别比较“仅首杀阵营”、",
            "“阵营加时间/爆头/武器”等特征组，避免把确定性冗余当成性能提升。",
            "",
            "## 与外部模型相差多少",
            "",
            "M15 没有新模型，因此首杀后差值为“不适用”。最近有效的 M14 开局前结果仍是：",
            "Accuracy 比最接近的公开 DNN 低 3.18 个百分点，Log Loss 高 0.023873。",
            "两者数据和切分不同；而且 M15 的预测时点更晚，不能直接代替首杀后比较。",
            "历史首杀 XGBoost 测试 AUC 0.774750 使用旧主键和旧事件选择，继续标记为无效历史值。",
            "",
            "## 可复现产物",
            "",
            f"- 数据 SHA-256：`{artifact['sha256']}`。",
            f"- 数据字节数：{artifact['bytes']:,}。",
            "- 完整检查明细：`m15_checks.csv`。",
            "- 47 个排除回合：`excluded_rounds.csv`。",
            "- 外部比较状态：`external_benchmark_comparison.csv/.md`。",
            "",
            "运行命令：",
            "",
            "```powershell",
            ".\\scripts\\run_first_kill_data_stage.ps1",
            "```",
            "",
            "## 下一阶段",
            "",
            "M16 在这份固定样本上先训练 Dummy 和逻辑回归，再训练未经调参的 XGBoost。",
            "三者使用完全相同的 train/validation/test 行和特征组；测试集在方案冻结前不参与选择。",
            "",
        ]
    )
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run(
    project_root: str | Path,
    rounds_path: str | Path,
    kills_path: str | Path,
    output_path: str | Path,
    split_manifest_path: str | Path,
    report_dir: str | Path,
    m14_comparison_path: str | Path,
) -> dict[str, Any]:
    rounds = read_table(rounds_path)
    kills = read_table(kills_path)
    manifest = read_table(split_manifest_path)
    output = Path(output_path)
    reports = Path(report_dir)
    reports.mkdir(parents=True, exist_ok=True)

    previous_samples = read_table(output) if output.is_file() else None
    samples = apply_split_manifest(make_first_kill_samples(rounds, kills), manifest)
    summary, excluded = audit_first_kill_data(rounds, kills, samples, manifest)
    previous = resolve_previous_comparison(
        previous_samples, samples, output_path=output, report_dir=reports
    )
    split_summary = build_split_summary(samples)

    automated_tests = run_automated_tests(project_root)
    summary["checks"]["automated_tests"] = {
        "passed": automated_tests["passed"],
        "blocking": True,
        "return_code": automated_tests["return_code"],
        "elapsed_seconds": automated_tests["elapsed_seconds"],
    }
    summary["passed"] = all(
        check["passed"]
        for check in summary["checks"].values()
        if check["blocking"]
    )

    checks_frame(summary).to_csv(reports / "m15_checks.csv", index=False)
    (reports / "automated_test_output.txt").write_text(
        automated_tests["output"], encoding="utf-8"
    )
    excluded.to_csv(reports / "excluded_rounds.csv", index=False)
    split_summary.to_csv(reports / "split_summary.csv", index=False)

    comparison = build_external_comparison_status(m14_comparison_path)
    comparison.to_csv(reports / "external_benchmark_comparison.csv", index=False)
    (reports / "external_benchmark_comparison.md").write_text(
        render_external_comparison(comparison), encoding="utf-8"
    )

    if not summary["passed"]:
        failure_summary = {
            **summary,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_dataset_comparison": previous,
            "data_artifact_written": False,
        }
        with (reports / "m15_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                failure_summary, handle, indent=2, ensure_ascii=False, default=_json_default
            )
        raise RuntimeError("M15 data acceptance failed; existing output was not replaced")

    write_table(samples, output)
    artifact = fingerprint_file(output)
    summary = {
        **summary,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_definition": "purchase complete, immediately after earliest valid enemy kill by tick",
        "previous_dataset_comparison": previous,
        "split_summary": split_summary.to_dict(orient="records"),
        "data_artifact": artifact,
        "external_comparison_status": "not_applicable_no_model",
        "next_stage": "M16 fixed-split Dummy, logistic regression, and untuned XGBoost baselines",
    }
    with (reports / "m15_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=_json_default)
    (reports / "m15_first_kill_data_report.md").write_text(
        render_m15_report(summary, previous, split_summary, artifact), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild and audit repaired-key post-first-kill samples for M15."
    )
    parser.add_argument("--rounds", default="data/interim/esta_full/rounds.parquet")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--kills", default="data/interim/esta_full/kills.parquet")
    parser.add_argument(
        "--output", default="data/processed/esta_full/first_kill.parquet"
    )
    parser.add_argument(
        "--split-manifest",
        default="reports/esta_full_m14/split_assignments.csv",
    )
    parser.add_argument("--report-dir", default="reports/esta_full_m15")
    parser.add_argument(
        "--m14-comparison",
        default="reports/esta_full_m14/external_benchmark_comparison.csv",
    )
    args = parser.parse_args()

    summary = run(
        project_root=args.project_root,
        rounds_path=args.rounds,
        kills_path=args.kills,
        output_path=args.output,
        split_manifest_path=args.split_manifest,
        report_dir=args.report_dir,
        m14_comparison_path=args.m14_comparison,
    )
    print(
        f"M15 passed: {summary['counts']['sample_rows']:,} samples; "
        f"{summary['counts']['excluded_rounds']:,} rounds excluded"
    )


if __name__ == "__main__":
    main()

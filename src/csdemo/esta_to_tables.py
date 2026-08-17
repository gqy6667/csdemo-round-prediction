from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path
from typing import Iterable

import pandas as pd

from .io import write_table


WEAPON_ALIASES = {
    "ak-47": "ak47",
    "m4a4": "m4a4",
    # ESTA/AWPY uses "M4A1" for the silenced M4A1-S inventory item.
    "m4a1": "m4a1_s",
    "m4a1-s": "m4a1_s",
    "awp": "awp",
}

RIFLE_NAMES = {
    "ak-47",
    "m4a4",
    "m4a1",
    "m4a1-s",
    "galil ar",
    "famas",
    "sg 553",
    "aug",
}

SMG_CLASSES = {"SMG"}
GRENADE_CLASSES = {"Grenade"}


def iter_demo_files(input_dir: Path, subsets: list[str]) -> Iterable[Path]:
    for subset in subsets:
        yield from sorted((input_dir / subset).glob("*.json.xz"))


def load_demo(path: Path) -> dict:
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def nearest_frame(round_data: dict, target_tick: int | None) -> dict | None:
    frames = round_data.get("frames") or []
    if not frames:
        return None
    if target_tick is None:
        return frames[0]
    return min(frames, key=lambda frame: abs((frame.get("tick") or 0) - target_tick))


def demo_identity(demo: dict, path: Path) -> tuple[str, str]:
    """Return the series-level split ID and map-level demo ID."""
    # ESTA matchId groups maps in one series; each source file represents one map/demo.
    file_id = path.name.removesuffix(".json.xz")
    game_id = f"{path.parent.name}:{file_id}"
    series_id = str(demo.get("matchId") or demo.get("demoId") or game_id)
    return series_id, game_id


def make_round_id(game_id: str, round_num: int) -> str:
    """Build an ID that stays unique when one series contains several maps."""
    if not game_id:
        raise ValueError("game_id must not be empty")
    return f"{game_id}_{round_num}"


def weapon_bucket(inventory: list[dict]) -> dict[str, int]:
    """Return normalized features for one player's inventory."""
    counts = {
        "ak47": 0,
        "m4a4": 0,
        "m4a1_s": 0,
        "awp": 0,
        "rifles": 0,
        "smgs": 0,
        "grenades": 0,
    }
    for item in inventory or []:
        name = str(item.get("weaponName") or "").lower()
        cls = str(item.get("weaponClass") or "")
        alias = WEAPON_ALIASES.get(name)
        if alias:
            counts[alias] = 1
        if name in RIFLE_NAMES:
            counts["rifles"] = 1
        if cls in SMG_CLASSES:
            counts["smgs"] = 1
        if cls in GRENADE_CLASSES:
            counts["grenades"] = min(counts["grenades"] + 1, 4)
    return counts


def side_features(frame: dict, side: str) -> dict[str, int | float]:
    side_data = frame.get(side, {}) or {}
    players = side_data.get("players") or []
    prefix = "ct" if side == "ct" else "t"

    features: dict[str, int | float] = {
        f"{prefix}_alive": int(side_data.get("alivePlayers") or 0),
        f"{prefix}_eq_value": int(side_data.get("teamEqVal") or 0),
        f"{prefix}_cash": 0,
        f"{prefix}_armor": 0,
        f"{prefix}_helmets": 0,
        f"{prefix}_defuse_kits": 0,
        f"{prefix}_grenades": 0,
        f"{prefix}_ak47": 0,
        f"{prefix}_m4a4": 0,
        f"{prefix}_m4a1_s": 0,
        f"{prefix}_awp": 0,
        f"{prefix}_rifles": 0,
        f"{prefix}_smgs": 0,
    }

    for player in players:
        if not player.get("isAlive", True):
            continue
        features[f"{prefix}_cash"] += int(player.get("cash") or 0)
        features[f"{prefix}_armor"] += int((player.get("armor") or 0) > 0)
        features[f"{prefix}_helmets"] += int(bool(player.get("hasHelmet")))
        features[f"{prefix}_defuse_kits"] += int(bool(player.get("hasDefuse")))
        buckets = weapon_bucket(player.get("inventory") or [])
        for key, value in buckets.items():
            features[f"{prefix}_{key}"] += value
    return features


def round_row(demo: dict, path: Path, round_data: dict) -> dict | None:
    if round_data.get("isWarmup"):
        return None
    # This is the purchase-complete, pre-combat snapshot used by the first task.
    frame = nearest_frame(round_data, round_data.get("freezeTimeEndTick"))
    if frame is None:
        return None

    ct_features = side_features(frame, "ct")
    t_features = side_features(frame, "t")
    if ct_features["ct_alive"] != 5 or t_features["t_alive"] != 5:
        return None

    series_id, game_id = demo_identity(demo, path)
    round_num = int(round_data.get("roundNum") or 0)
    row = {
        "series_id": series_id,
        "game_id": game_id,
        "round_id": make_round_id(game_id, round_num),
        "source_subset": path.parent.name,
        "map_name": demo.get("mapName"),
        "round_num": round_num,
        "ct_score": int(round_data.get("ctScore") or 0),
        "t_score": int(round_data.get("tScore") or 0),
        "ct_win": int(str(round_data.get("winningSide")).upper() == "CT"),
    }
    row.update(ct_features)
    row.update(t_features)
    return row


def kill_rows(demo: dict, path: Path, round_data: dict) -> list[dict]:
    if round_data.get("isWarmup"):
        return []
    series_id, game_id = demo_identity(demo, path)
    round_num = int(round_data.get("roundNum") or 0)
    round_id = make_round_id(game_id, round_num)
    rows = []
    for kill in round_data.get("kills") or []:
        if kill.get("isSuicide") or kill.get("isTeamkill"):
            continue
        rows.append(
            {
                "series_id": series_id,
                "game_id": game_id,
                "round_id": round_id,
                "time": float(kill.get("seconds") or 0),
                "tick": int(kill.get("tick") or 0),
                "killer_side": kill.get("attackerSide"),
                "victim_side": kill.get("victimSide"),
                "weapon": kill.get("weapon"),
                "weapon_class": kill.get("weaponClass"),
                "headshot": int(bool(kill.get("isHeadshot"))),
                "is_trade": int(bool(kill.get("isTrade"))),
                "is_first_kill": int(bool(kill.get("isFirstKill"))),
            }
        )
    return rows


def demo_rows(demo: dict, path: Path) -> tuple[list[dict], list[dict]]:
    """Build tables only for rounds with a valid 5v5 pre-combat snapshot."""
    rounds: list[dict] = []
    kills: list[dict] = []
    for round_data in demo.get("gameRounds") or []:
        row = round_row(demo, path, round_data)
        if row is None:
            continue
        rounds.append(row)
        kills.extend(kill_rows(demo, path, round_data))
    return rounds, kills


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=r"C:\project1\data\esta")
    parser.add_argument("--output", default="data/interim/esta")
    parser.add_argument("--subsets", nargs="+", default=["lan", "online"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    files = list(iter_demo_files(input_dir, args.subsets))
    if args.max_files:
        files = files[: args.max_files]

    rounds: list[dict] = []
    kills: list[dict] = []
    for idx, path in enumerate(files, start=1):
        demo = load_demo(path)
        demo_round_rows, demo_kill_rows = demo_rows(demo, path)
        rounds.extend(demo_round_rows)
        kills.extend(demo_kill_rows)
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(files)} demos")

    write_table(pd.DataFrame(rounds), output_dir / f"rounds.{args.format}")
    write_table(pd.DataFrame(kills), output_dir / f"kills.{args.format}")
    print(f"Wrote {len(rounds):,} rounds and {len(kills):,} kills from {len(files):,} demos")


if __name__ == "__main__":
    main()

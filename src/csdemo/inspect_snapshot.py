from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .esta_to_tables import GRENADE_CLASSES, RIFLE_NAMES, load_demo, nearest_frame


def _inventory_items(player: dict) -> list[dict[str, str]]:
    return [
        {
            "weapon_name": str(item.get("weaponName") or ""),
            "weapon_class": str(item.get("weaponClass") or ""),
        }
        for item in (player.get("inventory") or [])
    ]


def _player_report(player: dict) -> dict:
    inventory = _inventory_items(player)
    rifle_count = sum(
        item["weapon_name"].lower() in RIFLE_NAMES for item in inventory
    )
    utility_count = sum(
        item["weapon_class"] in GRENADE_CLASSES for item in inventory
    )
    item_counts = Counter(
        (item["weapon_name"], item["weapon_class"]) for item in inventory
    )
    duplicates = [
        {"weapon_name": name, "weapon_class": weapon_class, "count": count}
        for (name, weapon_class), count in sorted(item_counts.items())
        if count > 1
    ]
    return {
        "steam_id": player.get("steamID"),
        "name": player.get("name"),
        "is_alive": bool(player.get("isAlive", True)),
        "is_alive_field_present": "isAlive" in player,
        "active_weapon": player.get("activeWeapon"),
        "reported_total_utility": player.get("totalUtility"),
        "rifle_count": rifle_count,
        "utility_count": utility_count,
        "duplicate_inventory_items": duplicates,
        "inventory": inventory,
    }


def _side_report(frame: dict, side: str) -> dict:
    side_data = frame.get(side, {}) or {}
    players = [_player_report(player) for player in (side_data.get("players") or [])]
    alive_players = [player for player in players if player["is_alive"]]
    steam_id_counts = Counter(
        player["steam_id"] for player in players if player["steam_id"] is not None
    )
    return {
        "reported_alive_players": side_data.get("alivePlayers"),
        "listed_player_count": len(players),
        "listed_alive_player_count": len(alive_players),
        "reported_total_utility": side_data.get("totalUtility"),
        "derived_rifle_count": sum(player["rifle_count"] for player in alive_players),
        "derived_utility_count": sum(
            player["utility_count"] for player in alive_players
        ),
        "duplicate_steam_ids": sorted(
            steam_id for steam_id, count in steam_id_counts.items() if count > 1
        ),
        "players": players,
    }


def _find_round(demo: dict, round_num: int) -> dict:
    for round_data in demo.get("gameRounds") or []:
        if int(round_data.get("roundNum") or 0) == round_num:
            return round_data
    raise ValueError(f"Round {round_num} was not found")


def inspect_round_data(data: dict, round_num: int | None = None) -> dict:
    round_data = _find_round(data, round_num) if round_num is not None else data
    freeze_tick = round_data.get("freezeTimeEndTick")
    frame = nearest_frame(round_data, freeze_tick)
    if frame is None:
        raise ValueError(f"Round {round_data.get('roundNum')} has no frames")

    frame_tick = frame.get("tick")
    tick_offset = None
    if frame_tick is not None and freeze_tick is not None:
        tick_offset = int(frame_tick) - int(freeze_tick)

    return {
        "round_num": round_data.get("roundNum"),
        "freeze_time_end_tick": freeze_tick,
        "frame_tick": frame_tick,
        "frame_tick_offset": tick_offset,
        "ct": _side_report(frame, "ct"),
        "t": _side_report(frame, "t"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one ESTA freeze-time-end snapshot and its inventories."
    )
    parser.add_argument("--file", required=True, help="Path to an ESTA .json.xz file")
    parser.add_argument("--round", required=True, type=int, dest="round_num")
    args = parser.parse_args()

    path = Path(args.file)
    demo = load_demo(path)
    report = inspect_round_data(demo, round_num=args.round_num)
    report = {
        "source_file": str(path),
        "match_id": demo.get("matchId"),
        "map_name": demo.get("mapName"),
        **report,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

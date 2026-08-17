from pathlib import Path
import unittest

import pandas as pd

from src.csdemo.esta_to_tables import kill_rows, round_row
from src.csdemo.features import make_first_kill_samples, make_pre_round_samples
from src.csdemo.split import add_group_split


def demo_round(round_num: int, weapon: str) -> dict:
    frame = {
        "tick": 100,
        "ct": {"alivePlayers": 5, "teamEqVal": 10_000, "players": []},
        "t": {"alivePlayers": 5, "teamEqVal": 9_000, "players": []},
    }
    return {
        "roundNum": round_num,
        "freezeTimeEndTick": 100,
        "winningSide": "CT",
        "frames": [frame],
        "kills": [
            {
                "seconds": 20.0,
                "tick": 200,
                "attackerSide": "CT",
                "victimSide": "T",
                "weapon": weapon,
                "weaponClass": "Rifle",
            }
        ],
    }


class RoundIdentityTests(unittest.TestCase):
    def test_sample_tables_follow_the_new_identity_contract(self) -> None:
        rounds = pd.read_csv("data/sample/rounds.csv")
        kills = pd.read_csv("data/sample/kills.csv")

        for table in (rounds, kills):
            self.assertFalse(table[["series_id", "game_id", "round_id"]].isna().any().any())
            self.assertTrue(
                table.apply(lambda row: row["round_id"].startswith(row["game_id"] + "_"), axis=1).all()
            )

    def test_two_maps_in_one_series_get_distinct_game_and_round_ids(self) -> None:
        demo = {"matchId": "series-1", "mapName": "de_mirage"}
        round_data = demo_round(round_num=1, weapon="ak47")

        mirage = round_row(demo, Path("lan/game-a.json.xz"), round_data)
        inferno = round_row(demo, Path("lan/game-b.json.xz"), round_data)

        self.assertEqual(mirage["series_id"], inferno["series_id"])
        self.assertNotEqual(mirage["game_id"], inferno["game_id"])
        self.assertNotEqual(mirage["round_id"], inferno["round_id"])

    def test_first_kill_stays_attached_to_its_map(self) -> None:
        demo = {"matchId": "series-1", "mapName": "de_mirage"}
        mirage_round = demo_round(round_num=1, weapon="ak47")
        inferno_round = demo_round(round_num=1, weapon="awp")

        rounds = pd.DataFrame(
            [
                round_row(demo, Path("lan/game-a.json.xz"), mirage_round),
                round_row(demo, Path("lan/game-b.json.xz"), inferno_round),
            ]
        )
        kills = pd.DataFrame(
            kill_rows(demo, Path("lan/game-a.json.xz"), mirage_round)
            + kill_rows(demo, Path("lan/game-b.json.xz"), inferno_round)
        )

        samples = make_first_kill_samples(rounds, kills)
        weapons_by_game = samples.set_index("game_id")["first_kill_weapon"].to_dict()

        self.assertEqual(weapons_by_game["lan:game-a"], "ak47")
        self.assertEqual(weapons_by_game["lan:game-b"], "awp")

    def test_all_games_in_one_series_receive_the_same_split(self) -> None:
        rows = pd.DataFrame(
            {
                "series_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
                "game_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            }
        )

        result = add_group_split(rows)

        self.assertTrue((result.groupby("series_id")["split"].nunique() == 1).all())

    def test_duplicate_round_identity_is_rejected(self) -> None:
        rounds = pd.DataFrame(
            {
                "series_id": ["s1", "s1"],
                "game_id": ["g1", "g1"],
                "round_id": ["g1_1", "g1_1"],
                "ct_win": [1, 0],
            }
        )

        with self.assertRaisesRegex(ValueError, "Duplicate round identity"):
            make_pre_round_samples(rounds)


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from src.csdemo.quality import evaluate_quality, raise_for_errors
from src.csdemo.make_dataset import find_table


def valid_rounds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["series-1"],
            "game_id": ["lan:game-1"],
            "round_id": ["lan:game-1_1"],
            "source_subset": ["lan"],
            "map_name": ["de_mirage"],
            "round_num": [1],
            "ct_score": [0],
            "t_score": [0],
            "ct_win": [1],
            "ct_alive": [5],
            "t_alive": [5],
            "ct_eq_value": [10_000],
            "t_eq_value": [9_000],
            "ct_cash": [4_000],
            "t_cash": [3_000],
            "ct_armor": [5],
            "t_armor": [5],
            "ct_helmets": [5],
            "t_helmets": [5],
            "ct_defuse_kits": [2],
            "t_defuse_kits": [0],
            "ct_grenades": [12],
            "t_grenades": [12],
            "ct_ak47": [0],
            "t_ak47": [3],
            "ct_m4a4": [3],
            "t_m4a4": [0],
            "ct_m4a1_s": [0],
            "t_m4a1_s": [0],
            "ct_awp": [1],
            "t_awp": [1],
            "ct_rifles": [3],
            "t_rifles": [3],
            "ct_smgs": [0],
            "t_smgs": [0],
        }
    )


def valid_kills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["series-1"],
            "game_id": ["lan:game-1"],
            "round_id": ["lan:game-1_1"],
            "time": [20.0],
            "killer_side": ["CT"],
            "victim_side": ["T"],
        }
    )


class QualityTests(unittest.TestCase):
    def test_find_table_accepts_a_string_directory(self) -> None:
        table = find_table("data/sample", "rounds")

        self.assertEqual(table.name, "rounds.csv")

    def test_valid_tables_have_no_errors_or_warnings(self) -> None:
        summary, examples = evaluate_quality(valid_rounds(), valid_kills())

        self.assertFalse((summary["severity"] == "error").any())
        self.assertFalse((summary["severity"] == "warning").any())
        self.assertTrue(examples.empty)

    def test_hard_errors_are_reported_and_block_the_pipeline(self) -> None:
        rounds = pd.concat([valid_rounds(), valid_rounds()], ignore_index=True)
        rounds.loc[0, "ct_win"] = 2
        kills = valid_kills().copy()
        kills.loc[0, "round_id"] = "lan:missing_1"

        summary, _ = evaluate_quality(rounds, kills)
        checks = set(summary.loc[summary["severity"] == "error", "check"])

        self.assertIn("duplicate_round_identity", checks)
        self.assertIn("invalid_ct_win", checks)
        self.assertIn("orphan_kill_round", checks)
        with self.assertRaisesRegex(ValueError, "Data quality errors"):
            raise_for_errors(summary)

    def test_suspicious_pre_combat_values_are_warnings(self) -> None:
        rounds = valid_rounds()
        rounds.loc[0, "ct_alive"] = 4
        rounds.loc[0, "round_num"] = 5
        rounds.loc[0, "ct_rifles"] = 6
        rounds.loc[0, "t_grenades"] = 21

        summary, examples = evaluate_quality(rounds, valid_kills())
        checks = set(summary.loc[summary["severity"] == "warning", "check"])

        self.assertIn("pre_combat_alive_not_five", checks)
        self.assertIn("score_round_relation_mismatch", checks)
        self.assertIn("weapon_count_exceeds_five", checks)
        self.assertIn("grenade_count_exceeds_twenty", checks)
        self.assertFalse(examples.empty)

    def test_orphan_kill_example_identifies_the_orphan_row(self) -> None:
        kills = pd.concat([valid_kills(), valid_kills()], ignore_index=True)
        kills.loc[1, "round_id"] = "lan:missing_1"

        _, examples = evaluate_quality(valid_rounds(), kills)
        orphan_examples = examples[examples["check"].eq("orphan_kill_round")]

        self.assertEqual(orphan_examples["round_id"].tolist(), ["lan:missing_1"])


if __name__ == "__main__":
    unittest.main()

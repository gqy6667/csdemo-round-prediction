import unittest

import numpy as np
import pandas as pd

from src.csdemo.m18_first_kill_evaluation import (
    KEY_COLUMNS,
    assign_first_kill_time_band,
    assign_weapon_family,
    audit_prediction_replay,
    bootstrap_source_auc_gap,
    parse_source_subset,
    post_first_kill_error_pattern,
    prepare_analysis_table,
)


class M18FirstKillEvaluationTests(unittest.TestCase):
    def test_fixed_time_bands_use_documented_boundaries(self) -> None:
        result = assign_first_kill_time_band(
            pd.Series([0.0, 14.999, 15.0, 29.999, 30.0, 59.999, 60.0, 100.0])
        )

        self.assertEqual(
            result.tolist(),
            [
                "fast_00_15",
                "fast_00_15",
                "normal_15_30",
                "normal_15_30",
                "late_30_60",
                "late_30_60",
                "very_late_60_plus",
                "very_late_60_plus",
            ],
        )

    def test_weapon_families_cover_each_fixed_category(self) -> None:
        weapons = pd.Series(
            ["AK-47", "AWP", "Desert Eagle", "MP9", "Molotov", "unknown"]
        )

        result = assign_weapon_family(weapons)

        self.assertEqual(
            result.tolist(),
            [
                "rifle",
                "sniper",
                "pistol",
                "smg_shotgun",
                "utility_other",
                "utility_other",
            ],
        )

    def test_source_parser_is_strict_about_game_id_prefix(self) -> None:
        result = parse_source_subset(
            pd.Series(["lan:game-1", "online:game-2"], name="game_id")
        )

        self.assertEqual(result.tolist(), ["lan", "online"])
        with self.assertRaisesRegex(ValueError, "LAN/online"):
            parse_source_subset(pd.Series(["game-without-prefix"], name="game_id"))

    def test_analysis_join_uses_the_complete_three_column_key(self) -> None:
        predictions = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "game_id": ["lan:g1", "online:g2"],
                "round_id": ["same-round", "same-round"],
                "y_true": [1, 0],
                "ct_win_probability": [0.8, 0.3],
                "predicted_label": [1, 0],
            }
        )
        features = pd.DataFrame(
            {
                "series_id": ["s2", "s1"],
                "game_id": ["online:g2", "lan:g1"],
                "round_id": ["same-round", "same-round"],
                "split": ["test", "test"],
                "ct_win": [0, 1],
                "map_name": ["de_nuke", "de_inferno"],
                "round_num": [12, 3],
                "eq_value_diff_ct": [-2000, 3000],
                "first_kill_time": [35.0, 10.0],
                "first_kill_advantage_ct": [-1, 1],
                "first_kill_weapon": ["AWP", "AK-47"],
                "first_kill_headshot": [0, 1],
            }
        )

        analysis = prepare_analysis_table(predictions, features)

        self.assertEqual(list(KEY_COLUMNS), ["series_id", "game_id", "round_id"])
        by_game = analysis.set_index("game_id")
        self.assertEqual(by_game.loc["lan:g1", "map_name"], "de_inferno")
        self.assertEqual(by_game.loc["online:g2", "map_name"], "de_nuke")
        self.assertEqual(by_game.loc["lan:g1", "source_subset"], "lan")
        self.assertEqual(by_game.loc["online:g2", "first_kill_side"], "T")

    def test_analysis_join_rejects_an_incomplete_key_set(self) -> None:
        predictions = pd.DataFrame(
            {
                "series_id": ["s1"],
                "game_id": ["lan:g1"],
                "round_id": ["r1"],
                "y_true": [1],
                "ct_win_probability": [0.8],
                "predicted_label": [1],
            }
        )
        features = pd.DataFrame(
            {
                "series_id": ["s1"],
                "game_id": ["lan:g1"],
                "round_id": ["different"],
                "split": ["test"],
                "ct_win": [1],
            }
        )

        with self.assertRaisesRegex(ValueError, "complete key set"):
            prepare_analysis_table(predictions, features)

    def test_probability_replay_requires_keys_and_values_to_match(self) -> None:
        saved = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "game_id": ["lan:g1", "online:g2"],
                "round_id": ["r1", "r2"],
                "xgboost_tuned_probability": [0.2, 0.8],
            }
        )
        replayed = saved[list(KEY_COLUMNS)].iloc[::-1].reset_index(drop=True)
        replayed["ct_win_probability"] = [0.8, 0.2]

        result = audit_prediction_replay(saved, replayed, tolerance=1e-12)

        self.assertTrue(result["passed"])
        self.assertEqual(result["key_mismatch_count"], 0)
        self.assertLessEqual(result["max_absolute_probability_difference"], 1e-12)
        broken = replayed.copy()
        broken.loc[0, "ct_win_probability"] += 1e-5
        self.assertFalse(
            audit_prediction_replay(saved, broken, tolerance=1e-12)["passed"]
        )

    def test_error_pattern_separates_first_kill_and_equipment_support(self) -> None:
        base = {
            "predicted_label": 1,
            "first_kill_advantage_ct": 1,
            "eq_value_diff_ct": 2000,
        }

        self.assertEqual(
            post_first_kill_error_pattern(pd.Series(base)),
            "first_kill_and_equipment_agree",
        )
        self.assertEqual(
            post_first_kill_error_pattern(
                pd.Series({**base, "eq_value_diff_ct": 0})
            ),
            "first_kill_only",
        )
        self.assertEqual(
            post_first_kill_error_pattern(
                pd.Series({**base, "first_kill_advantage_ct": -1})
            ),
            "equipment_only",
        )
        self.assertEqual(
            post_first_kill_error_pattern(
                pd.Series(
                    {**base, "first_kill_advantage_ct": -1, "eq_value_diff_ct": 0}
                )
            ),
            "neither",
        )

    def test_source_auc_gap_bootstraps_complete_series(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": np.repeat([f"s{i}" for i in range(8)], 4),
                "source_subset": np.repeat(["lan"] * 4 + ["online"] * 4, 4),
                "y_true": np.tile([0, 0, 1, 1], 8),
                "ct_win_probability": np.concatenate(
                    [
                        np.tile([0.1, 0.2, 0.8, 0.9], 4),
                        np.tile([0.6, 0.4, 0.5, 0.7], 4),
                    ]
                ),
            }
        )

        result = bootstrap_source_auc_gap(frame, n_bootstrap=40, seed=7)

        self.assertGreater(result["signed_difference"], 0)
        self.assertEqual(result["successful_bootstraps"], 40)
        self.assertEqual(result["bootstrap_unit"], "series_id_global")


if __name__ == "__main__":
    unittest.main()

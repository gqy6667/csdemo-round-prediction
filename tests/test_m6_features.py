import unittest

import pandas as pd

from src.csdemo.features import add_common_diffs
from src.csdemo.m6_analysis import (
    ABLATION_VARIANTS,
    drop_feature_group,
    make_ablation_model,
)
from src.csdemo.schema import PRE_ROUND_FEATURES
from src.csdemo.train_xgb import align_columns, prepare_xy


class M6FeatureTests(unittest.TestCase):
    def test_all_common_differences_are_ct_minus_t(self) -> None:
        source = pd.DataFrame(
            {
                "ct_score": [8],
                "t_score": [5],
                "ct_alive": [5],
                "t_alive": [4],
                "ct_eq_value": [24_000],
                "t_eq_value": [20_000],
                "ct_cash": [3_000],
                "t_cash": [4_000],
                "ct_armor": [5],
                "t_armor": [4],
                "ct_helmets": [3],
                "t_helmets": [5],
                "ct_awp": [1],
                "t_awp": [2],
                "ct_rifles": [4],
                "t_rifles": [3],
                "ct_smgs": [0],
                "t_smgs": [1],
                "ct_grenades": [12],
                "t_grenades": [9],
            }
        )

        result = add_common_diffs(source).iloc[0]

        expected = {
            "score_diff_ct": 3,
            "alive_diff_ct": 1,
            "eq_value_diff_ct": 4_000,
            "cash_diff_ct": -1_000,
            "armor_diff_ct": 1,
            "helmet_diff_ct": -2,
            "awp_diff_ct": -1,
            "rifle_diff_ct": 1,
            "smg_diff_ct": -1,
            "grenade_diff_ct": 3,
        }
        self.assertEqual(result[list(expected)].to_dict(), expected)

    def test_constant_precombat_alive_columns_are_not_model_features(self) -> None:
        self.assertNotIn("ct_alive", PRE_ROUND_FEATURES)
        self.assertNotIn("t_alive", PRE_ROUND_FEATURES)
        self.assertNotIn("alive_diff_ct", PRE_ROUND_FEATURES)

    def test_validation_categories_are_aligned_to_training_vocabulary(self) -> None:
        train = pd.DataFrame(
            {
                "series_id": ["s1"],
                "game_id": ["g1"],
                "round_id": ["r1"],
                "split": ["train"],
                "ct_win": [1],
                "map_name": ["de_mirage"],
                "ct_cash": [1000],
            }
        )
        validation = train.assign(
            series_id="s2",
            game_id="g2",
            round_id="r2",
            split="val",
            map_name="de_unseen",
        )

        x_train, _ = prepare_xy(train)
        x_validation, _ = prepare_xy(validation)
        aligned = align_columns(x_train, x_validation)

        self.assertListEqual(list(aligned.columns), list(x_train.columns))
        self.assertNotIn("map_name_de_unseen", aligned.columns)
        self.assertNotIn("map_name_nan", aligned.columns)
        self.assertEqual(int(aligned.loc[0, "map_name_de_mirage"]), 0)

    def test_ablation_removes_only_the_requested_group(self) -> None:
        frame = pd.DataFrame(
            {
                "ct_score": [1],
                "t_score": [2],
                "score_diff_ct": [-1],
                "ct_cash": [500],
                "ct_win": [0],
            }
        )

        result = drop_feature_group(frame, "score")

        self.assertNotIn("ct_score", result.columns)
        self.assertNotIn("t_score", result.columns)
        self.assertNotIn("score_diff_ct", result.columns)
        self.assertIn("ct_cash", result.columns)
        self.assertIn("ct_win", result.columns)

    def test_targeted_economy_ablations_separate_cash_and_equipment_value(self) -> None:
        self.assertSetEqual(
            set(ABLATION_VARIANTS["without_cash"]),
            {"ct_cash", "t_cash", "cash_diff_ct"},
        )
        self.assertSetEqual(
            set(ABLATION_VARIANTS["without_equipment_value"]),
            {"ct_eq_value", "t_eq_value", "eq_value_diff_ct"},
        )

    def test_ablation_model_does_not_randomly_drop_rows_or_features(self) -> None:
        params = make_ablation_model().get_params()

        self.assertEqual(params["subsample"], 1.0)
        self.assertEqual(params["colsample_bytree"], 1.0)


if __name__ == "__main__":
    unittest.main()

import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.csdemo.m31_post_first_kill_lightgbm_explanation import (
    BLOCKING_CHECKS,
    FIRST_KILL_EVENT_FEATURES,
    audit_post_first_kill_features,
    audit_reproduction_entrypoint,
    build_macro_feature_groups,
    build_model_importance_comparison,
    build_source_feature_groups,
    decide_acceptance,
    lightgbm_gain_importance,
    map_encoded_feature_to_source,
    verify_m30_prerequisite,
)
from src.csdemo.m16_first_kill_baselines import canonical_feature_names


class FakeBooster:
    def __init__(self, trees: int):
        self.trees = trees

    def num_trees(self):
        return self.trees


class FakeModel:
    def __init__(self, trees: int):
        self.booster_ = FakeBooster(trees)

    def predict_proba(self, values):
        return values


class FakeImportanceBooster(FakeBooster):
    def feature_name(self):
        return ["round_num", "first_kill_weapon_Desert_Eagle"]

    def feature_importance(self, importance_type, iteration):
        if importance_type == "gain":
            return [3.0, 1.0]
        return [2, 1]


class FakeImportanceModel:
    def __init__(self):
        self.booster_ = FakeImportanceBooster(211)


class M31PostFirstKillLightGBMExplanationTests(unittest.TestCase):
    def test_mapping_handles_numeric_map_and_first_kill_weapon_columns(self) -> None:
        raw = ["eq_value_diff_ct", "map_name", "first_kill_weapon"]

        self.assertEqual(map_encoded_feature_to_source("eq_value_diff_ct", raw), "eq_value_diff_ct")
        self.assertEqual(map_encoded_feature_to_source("map_name_de_nuke", raw), "map_name")
        self.assertEqual(
            map_encoded_feature_to_source("first_kill_weapon_AWP", raw),
            "first_kill_weapon",
        )

    def test_leakage_audit_allows_first_kill_contract_and_rejects_future(self) -> None:
        raw = ["map_name", "first_kill_advantage_ct", "first_kill_weapon"]
        encoded = [
            "map_name_de_nuke",
            "first_kill_advantage_ct",
            "first_kill_weapon_AWP",
            "series_id",
            "ct_alive_after_second_kill",
            "ct_win",
        ]

        audit = audit_post_first_kill_features(encoded, raw).set_index("encoded_feature")

        self.assertEqual(audit.loc["first_kill_weapon_AWP", "audit_result"], "pass")
        self.assertEqual(audit.loc["series_id", "audit_result"], "fail")
        self.assertEqual(audit.loc["ct_alive_after_second_kill", "audit_result"], "fail")
        self.assertEqual(audit.loc["ct_win", "audit_result"], "fail")

    def test_source_and_macro_groups_cover_columns_once(self) -> None:
        raw = ["round_num", "map_name", *FIRST_KILL_EVENT_FEATURES]
        encoded = [
            "round_num",
            "map_name_de_nuke",
            "first_kill_advantage_ct",
            "first_kill_time",
            "first_kill_headshot",
            "first_kill_weapon_AWP",
        ]

        source = build_source_feature_groups(encoded, raw)
        macro = build_macro_feature_groups(encoded, raw)

        self.assertCountEqual(
            [column for columns in source.values() for column in columns], encoded
        )
        self.assertEqual(set(macro), {"purchase_end", "first_kill_event"})
        self.assertCountEqual(
            [column for columns in macro.values() for column in columns], encoded
        )

    def test_model_importance_comparison_reports_rank_agreement(self) -> None:
        lightgbm = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "gain_rank": [1, 2, 3],
                "permutation_rank": [1, 3, 2],
                "shap_rank": [1, 2, 3],
                "mean_rank": [1.0, 2.3, 2.7],
            }
        )
        xgboost = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "gain_rank": [1, 3, 2],
                "permutation_rank": [1, 2, 3],
                "shap_rank": [1, 2, 3],
                "mean_rank": [1.0, 2.3, 2.7],
            }
        )

        detail, summary = build_model_importance_comparison(lightgbm, xgboost, top_n=2)

        self.assertEqual(len(detail), 3)
        self.assertEqual(set(summary["method"]), {"gain", "permutation_auc", "tree_shap", "mean_rank"})
        self.assertTrue(summary["top_overlap_count"].between(1, 2).all())

    def test_gain_importance_preserves_bundle_names_when_booster_sanitizes_spaces(self) -> None:
        bundle = {
            "model": FakeImportanceModel(),
            "columns": ["round_num", "first_kill_weapon_Desert Eagle"],
            "best_iteration": 211,
        }

        importance = lightgbm_gain_importance(bundle)

        self.assertEqual(
            set(importance["feature"]),
            {"round_num", "first_kill_weapon_Desert Eagle"},
        )
        self.assertAlmostEqual(float(importance["gain_normalized"].sum()), 1.0)

    def test_prerequisite_rejects_task_or_tree_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "first_kill.parquet"
            model_path = root / "model.joblib"
            data_path.write_bytes(b"data")
            model_path.write_bytes(b"model")
            data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            raw = canonical_feature_names()
            bundle = {
                "model": FakeModel(211),
                "task": "post_first_kill",
                "model_name": "lightgbm_tuned",
                "raw_features": raw,
                "columns": [f"f{i}" for i in range(82)],
                "best_iteration": 211,
                "data_sha256": data_sha,
            }
            summary = {
                "acceptance": {"status": "passed", "ready_for_m31": True},
                "task": "post_first_kill",
                "data": {"sha256": data_sha},
                "model_replay": {"raw_feature_count": 40, "encoded_feature_count": 82},
                "prerequisite": {"model_artifact": {"sha256": model_sha}},
            }

            accepted = verify_m30_prerequisite(data_path, model_path, summary, bundle)
            rejected = verify_m30_prerequisite(
                data_path, model_path, summary, {**bundle, "best_iteration": 210}
            )

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])

    def test_acceptance_requires_evidence_but_not_rank_agreement_target(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        accepted = decide_acceptance(checks)
        checks["shap_reconstruction"] = False
        rejected = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["ready_for_m32"])
        self.assertNotIn("high_xgboost_rank_correlation", BLOCKING_CHECKS)
        self.assertEqual(rejected["blocking_failures"], ["shap_reconstruction"])

    def test_reproduction_entrypoint_freezes_model_and_twenty_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "run.ps1"
            script.write_text(
                " ".join(
                    [
                        "src.csdemo.m31_post_first_kill_lightgbm_explanation",
                        "post_first_kill_lightgbm_tuned.joblib",
                        "m30_summary.json",
                        "[int]$PermutationRepeats = 20",
                    ]
                ),
                encoding="utf-8",
            )

            accepted = audit_reproduction_entrypoint(script)
            script.write_text("wrong", encoding="utf-8")
            rejected = audit_reproduction_entrypoint(script)

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])


if __name__ == "__main__":
    unittest.main()

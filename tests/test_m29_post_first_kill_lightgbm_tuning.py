import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.csdemo.m29_post_first_kill_lightgbm_tuning import (
    BASE_TUNING_PARAMS,
    BLOCKING_CHECKS,
    MINIMUM_PHASE_IMPROVEMENT,
    PHASE_DEFINITIONS,
    STABILITY_LIMITS,
    STABILITY_SEEDS,
    assess_stage_goals,
    assess_seed_stability,
    audit_candidate_grid,
    audit_frozen_params,
    audit_reproduction_entrypoint,
    build_final_prediction_table,
    decide_acceptance,
    run_sequential_search,
    select_phase_winner,
    validate_validation_only_table,
)


class FakeModel:
    def __init__(self, params):
        self.params = params

    def get_params(self):
        return self.params


class M29PostFirstKillLightGBMTuningTests(unittest.TestCase):
    def test_frozen_grid_has_nine_phases_and_exactly_36_candidates(self) -> None:
        audit = audit_candidate_grid(PHASE_DEFINITIONS, BASE_TUNING_PARAMS)

        self.assertTrue(audit["passed"], audit["violations"])
        self.assertEqual(audit["phase_count"], 9)
        self.assertEqual(audit["candidate_count"], 36)

    def test_every_phase_changes_one_parameter_and_contains_incumbent(self) -> None:
        for phase in PHASE_DEFINITIONS:
            self.assertEqual(len(phase["allowed_parameters"]), 1, phase["name"])
            parameter = phase["allowed_parameters"][0]
            values = [
                candidate["overrides"][parameter]
                for candidate in phase["candidates"]
            ]
            self.assertIn(BASE_TUNING_PARAMS[parameter], values, phase["name"])

    def test_selection_uses_only_real_validation_improvement(self) -> None:
        below = pd.DataFrame(
            [
                {
                    "candidate_id": "incumbent",
                    "candidate_order": 0,
                    "is_incumbent": True,
                    "val_log_loss": 0.528700,
                },
                {
                    "candidate_id": "tiny_gain",
                    "candidate_order": 1,
                    "is_incumbent": False,
                    "val_log_loss": 0.528650,
                },
            ]
        )
        threshold = below.copy()
        threshold.loc[1, "val_log_loss"] = 0.528600

        kept = select_phase_winner(below, MINIMUM_PHASE_IMPROVEMENT)
        changed = select_phase_winner(threshold, MINIMUM_PHASE_IMPROVEMENT)

        self.assertEqual(kept["candidate_id"], "incumbent")
        self.assertFalse(kept["changed"])
        self.assertEqual(changed["candidate_id"], "tiny_gain")
        self.assertTrue(changed["changed"])

    def test_search_entrypoint_cannot_receive_test_data(self) -> None:
        parameters = inspect.signature(run_sequential_search).parameters

        self.assertIn("train_prepared", parameters)
        self.assertIn("val_prepared", parameters)
        self.assertNotIn("test", parameters)
        self.assertNotIn("test_prepared", parameters)

    def test_validation_tables_reject_test_metric_columns(self) -> None:
        valid = pd.DataFrame([{"candidate_id": "a", "val_log_loss": 0.52}])
        leaked = valid.assign(test_auc=0.81)

        self.assertTrue(validate_validation_only_table(valid)["passed"])
        rejected = validate_validation_only_table(leaked)
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["forbidden_columns"], ["test_auc"])

    def test_frozen_parameter_audit_detects_drift(self) -> None:
        expected = {"num_leaves": 15, "min_child_samples": 40, "random_state": 42}

        accepted = audit_frozen_params(FakeModel(expected), expected)
        rejected = audit_frozen_params(
            FakeModel({**expected, "num_leaves": 31}), expected
        )

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])
        self.assertIn("num_leaves", rejected["mismatches"])

    def test_seed_stability_uses_five_frozen_seeds_without_test_columns(self) -> None:
        results = pd.DataFrame(
            {
                "seed": STABILITY_SEEDS,
                "val_log_loss": [0.5280, 0.5281, 0.5282, 0.5283, 0.5284],
                "val_auc": [0.8030, 0.8031, 0.8032, 0.8033, 0.8034],
            }
        )

        accepted = assess_seed_stability(results)
        rejected = assess_seed_stability(results.assign(test_auc=0.81))

        self.assertTrue(accepted["passed"])
        self.assertLessEqual(
            accepted["val_log_loss_range"], STABILITY_LIMITS["val_log_loss"]
        )
        self.assertFalse(rejected["passed"])

    def test_final_predictions_require_exact_m28_keys(self) -> None:
        rows = pd.DataFrame(
            [
                {"series_id": "s1", "game_id": "g1", "round_id": "r1", "ct_win": 1},
                {"series_id": "s2", "game_id": "g2", "round_id": "r2", "ct_win": 0},
            ]
        )
        m28 = rows.assign(
            xgboost_frozen_probability=[0.7, 0.3],
            lightgbm_baseline_probability=[0.8, 0.2],
        )

        result = build_final_prediction_table(rows, np.array([0.85, 0.15]), m28)

        self.assertEqual(result["lightgbm_tuned_probability"].tolist(), [0.85, 0.15])
        with self.assertRaisesRegex(ValueError, "M28 test keys"):
            build_final_prediction_table(
                rows.iloc[::-1].reset_index(drop=True), np.array([0.15, 0.85]), m28
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_final_prediction_table(rows, np.array([0.85, 1.2]), m28)

    def test_acceptance_requires_all_blockers_but_not_a_model_win(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        accepted = decide_acceptance(checks)
        checks["seed_stability"] = False
        rejected = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["ready_for_m30"])
        self.assertNotIn("beats_xgboost_on_test", BLOCKING_CHECKS)
        self.assertNotIn("beats_m28_on_test", BLOCKING_CHECKS)
        self.assertEqual(rejected["blocking_failures"], ["seed_stability"])

    def test_stage_goals_compare_tuned_validation_to_m28_baseline(self) -> None:
        baseline = pd.DataFrame(
            [
                {"split": "train", "auc": 0.8235, "log_loss": 0.5102},
                {"split": "val", "auc": 0.8029, "log_loss": 0.5287},
            ]
        )
        tuned = pd.DataFrame(
            [
                {"split": "train", "auc": 0.8240, "log_loss": 0.5099},
                {"split": "val", "auc": 0.8030, "log_loss": 0.5281},
            ]
        )

        goals = assess_stage_goals(baseline, tuned)

        self.assertTrue(goals["all_passed"])
        self.assertAlmostEqual(
            goals["goals"]["validation_log_loss_improvement"]["value"], 0.0006
        )
        self.assertAlmostEqual(
            goals["goals"]["validation_auc"]["target"], 0.8009
        )

    def test_reproduction_entrypoint_requires_m29_first_kill_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "run.ps1"
            script.write_text(
                " ".join(
                    [
                        "src.csdemo.m29_post_first_kill_lightgbm_tuning",
                        r"data\processed\esta_full\first_kill.parquet",
                        r"models\esta_full_m29",
                        r"reports\esta_full_m29",
                    ]
                ),
                encoding="utf-8",
            )

            accepted = audit_reproduction_entrypoint(script)
            script.write_text("wrong module", encoding="utf-8")
            rejected = audit_reproduction_entrypoint(script)

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])
        self.assertIn(
            "src.csdemo.m29_post_first_kill_lightgbm_tuning",
            rejected["missing_tokens"],
        )


if __name__ == "__main__":
    unittest.main()

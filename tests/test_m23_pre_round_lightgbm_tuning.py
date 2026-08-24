import inspect
import unittest

import numpy as np
import pandas as pd

from src.csdemo.m23_pre_round_lightgbm_tuning import (
    BASE_TUNING_PARAMS,
    BLOCKING_CHECKS,
    MINIMUM_PHASE_IMPROVEMENT,
    PHASE_DEFINITIONS,
    STABILITY_LIMITS,
    STABILITY_SEEDS,
    assess_seed_stability,
    audit_candidate_grid,
    audit_frozen_params,
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


class M23PreRoundLightGBMTuningTests(unittest.TestCase):
    def test_frozen_grid_has_nine_phases_and_exactly_36_candidates(self) -> None:
        audit = audit_candidate_grid(PHASE_DEFINITIONS, BASE_TUNING_PARAMS)

        self.assertTrue(audit["passed"], audit["violations"])
        self.assertEqual(audit["phase_count"], 9)
        self.assertEqual(audit["candidate_count"], 36)

    def test_every_phase_changes_one_parameter_and_contains_the_incumbent(self) -> None:
        for phase in PHASE_DEFINITIONS:
            self.assertEqual(len(phase["allowed_parameters"]), 1, phase["name"])
            parameter = phase["allowed_parameters"][0]
            values = [
                candidate["overrides"][parameter]
                for candidate in phase["candidates"]
            ]
            self.assertIn(BASE_TUNING_PARAMS[parameter], values, phase["name"])

    def test_selection_keeps_incumbent_when_gain_is_below_threshold(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "candidate_id": "incumbent",
                    "candidate_order": 0,
                    "is_incumbent": True,
                    "val_log_loss": 0.595600,
                },
                {
                    "candidate_id": "tiny_gain",
                    "candidate_order": 1,
                    "is_incumbent": False,
                    "val_log_loss": 0.595550,
                },
            ]
        )

        selected = select_phase_winner(results, MINIMUM_PHASE_IMPROVEMENT)

        self.assertEqual(selected["candidate_id"], "incumbent")
        self.assertFalse(selected["changed"])
        self.assertAlmostEqual(selected["best_observed_improvement"], 0.00005)

    def test_selection_accepts_validation_log_loss_gain_at_threshold(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "candidate_id": "incumbent",
                    "candidate_order": 0,
                    "is_incumbent": True,
                    "val_log_loss": 0.595600,
                },
                {
                    "candidate_id": "real_gain",
                    "candidate_order": 1,
                    "is_incumbent": False,
                    "val_log_loss": 0.595500,
                },
            ]
        )

        selected = select_phase_winner(results, MINIMUM_PHASE_IMPROVEMENT)

        self.assertEqual(selected["candidate_id"], "real_gain")
        self.assertTrue(selected["changed"])
        self.assertAlmostEqual(selected["accepted_improvement"], 0.0001)

    def test_search_entrypoint_cannot_receive_test_data(self) -> None:
        parameters = inspect.signature(run_sequential_search).parameters

        self.assertIn("train_prepared", parameters)
        self.assertIn("val_prepared", parameters)
        self.assertNotIn("test", parameters)
        self.assertNotIn("test_prepared", parameters)

    def test_validation_tables_reject_test_metric_columns(self) -> None:
        valid = pd.DataFrame([{"candidate_id": "a", "val_log_loss": 0.59}])
        leaked = valid.assign(test_auc=0.73)

        accepted = validate_validation_only_table(valid)
        rejected = validate_validation_only_table(leaked)

        self.assertTrue(accepted["passed"])
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

    def test_seed_stability_uses_fixed_seeds_limits_and_no_test_columns(self) -> None:
        results = pd.DataFrame(
            {
                "seed": STABILITY_SEEDS,
                "val_log_loss": [0.5940, 0.5941, 0.5942, 0.5943, 0.5944],
                "val_auc": [0.7190, 0.7191, 0.7192, 0.7193, 0.7194],
            }
        )

        accepted = assess_seed_stability(results)
        rejected = assess_seed_stability(results.assign(test_auc=0.73))

        self.assertTrue(accepted["passed"])
        self.assertLessEqual(
            accepted["val_log_loss_range"], STABILITY_LIMITS["val_log_loss"]
        )
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["forbidden_columns"], ["test_auc"])

    def test_final_predictions_require_exact_m22_keys_and_valid_probability(self) -> None:
        rows = pd.DataFrame(
            [
                {"series_id": "s1", "game_id": "g1", "round_id": "r1", "ct_win": 1},
                {"series_id": "s2", "game_id": "g2", "round_id": "r2", "ct_win": 0},
            ]
        )
        m22 = rows.assign(
            xgboost_frozen_probability=[0.7, 0.3],
            lightgbm_baseline_probability=[0.8, 0.2],
        )

        result = build_final_prediction_table(rows, np.array([0.85, 0.15]), m22)

        self.assertEqual(result["lightgbm_tuned_probability"].tolist(), [0.85, 0.15])
        changed = rows.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "M22 test keys"):
            build_final_prediction_table(changed, np.array([0.15, 0.85]), m22)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_final_prediction_table(rows, np.array([0.85, 1.2]), m22)

    def test_acceptance_requires_all_blockers_but_not_test_improvement(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        accepted = decide_acceptance(checks)
        checks["seed_stability"] = False
        rejected = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["ready_for_m24"])
        self.assertNotIn("beats_m22_on_test", BLOCKING_CHECKS)
        self.assertEqual(rejected["blocking_failures"], ["seed_stability"])


if __name__ == "__main__":
    unittest.main()

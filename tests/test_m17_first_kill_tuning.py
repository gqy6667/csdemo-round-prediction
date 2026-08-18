import inspect
import unittest

import numpy as np
import pandas as pd

from src.csdemo.m17_first_kill_tuning import (
    BASE_TUNING_PARAMS,
    BLOCKING_CHECKS,
    MINIMUM_PHASE_IMPROVEMENT,
    PHASE_DEFINITIONS,
    assess_seed_stability,
    audit_candidate_grid,
    audit_frozen_params,
    build_final_prediction_table,
    decide_acceptance,
    run_sequential_search,
    select_phase_winner,
    validate_tuning_table,
)


class FakeModel:
    def __init__(self, params: dict) -> None:
        self.params = params

    def get_params(self) -> dict:
        return dict(self.params)


class M17FirstKillTuningTests(unittest.TestCase):
    def test_frozen_grid_has_eight_phases_and_exactly_39_candidates(self) -> None:
        result = audit_candidate_grid(PHASE_DEFINITIONS, BASE_TUNING_PARAMS)

        self.assertTrue(result["passed"])
        self.assertEqual(result["phase_count"], 8)
        self.assertEqual(result["candidate_count"], 39)
        self.assertEqual(result["violations"], [])

    def test_tree_policy_is_the_only_phase_allowed_to_change_two_parameters(self) -> None:
        result = audit_candidate_grid(PHASE_DEFINITIONS, BASE_TUNING_PARAMS)
        changes = result["changed_parameters_by_phase"]

        self.assertEqual(
            changes["tree_policy"], ["early_stopping_rounds", "n_estimators"]
        )
        for phase_name, changed in changes.items():
            if phase_name != "tree_policy":
                self.assertEqual(len(changed), 1)

    def test_phase_selection_keeps_incumbent_below_minimum_improvement(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "candidate_id": "incumbent",
                    "candidate_order": 0,
                    "is_incumbent": True,
                    "val_log_loss": 0.52980,
                },
                {
                    "candidate_id": "tiny_gain",
                    "candidate_order": 1,
                    "is_incumbent": False,
                    "val_log_loss": 0.52975,
                },
            ]
        )

        selected = select_phase_winner(results, MINIMUM_PHASE_IMPROVEMENT)

        self.assertEqual(selected["candidate_id"], "incumbent")
        self.assertFalse(selected["changed"])
        self.assertAlmostEqual(selected["best_observed_improvement"], 0.00005)

    def test_phase_selection_accepts_a_real_validation_log_loss_gain(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "candidate_id": "incumbent",
                    "candidate_order": 0,
                    "is_incumbent": True,
                    "val_log_loss": 0.52980,
                },
                {
                    "candidate_id": "winner",
                    "candidate_order": 1,
                    "is_incumbent": False,
                    "val_log_loss": 0.52960,
                },
            ]
        )

        selected = select_phase_winner(results, MINIMUM_PHASE_IMPROVEMENT)

        self.assertEqual(selected["candidate_id"], "winner")
        self.assertTrue(selected["changed"])
        self.assertAlmostEqual(selected["accepted_improvement"], 0.00020)

    def test_tuning_entrypoint_cannot_receive_a_test_split(self) -> None:
        parameter_names = set(inspect.signature(run_sequential_search).parameters)

        self.assertNotIn("test", parameter_names)
        self.assertNotIn("test_prepared", parameter_names)
        self.assertEqual(
            parameter_names,
            {"train_prepared", "val_prepared", "phases", "minimum_improvement"},
        )

    def test_tuning_table_rejects_any_test_metric_column(self) -> None:
        valid = pd.DataFrame(
            [{"phase": "depth", "candidate_id": "depth_2", "val_log_loss": 0.52}]
        )
        leaked = valid.assign(test_auc=0.81)

        self.assertTrue(validate_tuning_table(valid)["passed"])
        result = validate_tuning_table(leaked)
        self.assertFalse(result["passed"])
        self.assertEqual(result["forbidden_columns"], ["test_auc"])

    def test_frozen_parameter_audit_detects_a_single_mismatch(self) -> None:
        expected = {**BASE_TUNING_PARAMS, "max_depth": 2}
        model = FakeModel({**expected, "max_depth": 3})

        result = audit_frozen_params(model, expected)

        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatches"]["max_depth"]["expected"], 2)
        self.assertEqual(result["mismatches"]["max_depth"]["actual"], 3)

    def test_seed_stability_uses_validation_ranges_and_fixed_thresholds(self) -> None:
        stable = pd.DataFrame(
            {
                "seed": [42, 43, 44, 45, 46],
                "val_log_loss": [0.5200, 0.5204, 0.5198, 0.5201, 0.5202],
                "val_auc": [0.8100, 0.8108, 0.8097, 0.8102, 0.8103],
            }
        )

        result = assess_seed_stability(stable)

        self.assertTrue(result["passed"])
        self.assertLessEqual(result["val_log_loss_range"], 0.002)
        self.assertLessEqual(result["val_auc_range"], 0.003)

    def test_final_prediction_table_requires_exact_m16_test_keys(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "series_id": "s1",
                    "game_id": "g1",
                    "round_id": "r1",
                    "ct_win": 1,
                }
            ]
        )
        m16 = rows[["series_id", "game_id", "round_id", "ct_win"]].copy()
        m16["xgboost_untuned_probability"] = 0.7

        result = build_final_prediction_table(rows, np.array([0.8]), m16)

        self.assertEqual(result.loc[0, "xgboost_tuned_probability"], 0.8)
        self.assertEqual(result.loc[0, "xgboost_tuned_prediction"], 1)
        broken = m16.assign(round_id="other")
        with self.assertRaisesRegex(ValueError, "M16 test keys"):
            build_final_prediction_table(rows, np.array([0.8]), broken)

    def test_acceptance_lists_only_failed_declared_blockers(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}
        checks["seed_stability"] = False

        result = decide_acceptance(checks)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["blocking_failures"], ["seed_stability"])
        self.assertFalse(result["ready_for_m18"])


if __name__ == "__main__":
    unittest.main()

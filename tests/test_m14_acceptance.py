import unittest

import pandas as pd

from src.csdemo.m14_acceptance import (
    assess_metric_targets,
    audit_quality_summary,
    audit_split_contract,
    decide_phase_readiness,
)


class M14AcceptanceTests(unittest.TestCase):
    def test_metric_assessment_separates_minimum_gate_from_stage_targets(self) -> None:
        metrics = {
            "accuracy": 0.647411,
            "auc": 0.727122,
            "log_loss": 0.591733,
            "brier_score": 0.205294,
        }

        result = assess_metric_targets(metrics)

        self.assertTrue(result["all_minimum_passed"])
        self.assertFalse(result["all_stage_passed"])
        self.assertEqual(result["minimum_passed_count"], 4)
        self.assertEqual(result["stage_passed_count"], 0)
        self.assertAlmostEqual(result["metrics"]["auc"]["stage_gap"], 0.002878)
        self.assertAlmostEqual(
            result["metrics"]["log_loss"]["stage_gap"], 0.011733
        )

    def test_split_audit_passes_grouped_unique_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": ["s1", "s1", "s2", "s3"],
                "game_id": ["g1", "g1", "g2", "g3"],
                "round_id": ["g1_1", "g1_2", "g2_1", "g3_1"],
                "split": ["train", "train", "val", "test"],
            }
        )

        result = audit_split_contract(frame)

        self.assertTrue(result["passed"])
        self.assertEqual(result["duplicate_round_ids"], 0)
        self.assertEqual(result["cross_split_series"], 0)
        self.assertEqual(result["series_counts"], {"train": 1, "val": 1, "test": 1})

    def test_split_audit_rejects_cross_split_series_and_duplicate_rounds(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": ["s1", "s1", "s2"],
                "game_id": ["g1", "g2", "g3"],
                "round_id": ["r1", "r1", "r3"],
                "split": ["train", "test", "val"],
            }
        )

        result = audit_split_contract(frame)

        self.assertFalse(result["passed"])
        self.assertEqual(result["cross_split_series"], 1)
        self.assertEqual(result["duplicate_round_ids"], 1)
        self.assertIn("series_id appears in multiple splits", result["errors"])
        self.assertIn("round_id is not unique", result["errors"])

    def test_quality_audit_allows_info_but_blocks_warning_or_error(self) -> None:
        info_only = pd.DataFrame(
            [{"severity": "info", "check": "round_without_valid_kill", "count": 47}]
        )
        with_warning = pd.concat(
            [
                info_only,
                pd.DataFrame(
                    [{"severity": "warning", "check": "bad_cash", "count": 2}]
                ),
            ],
            ignore_index=True,
        )

        self.assertTrue(audit_quality_summary(info_only)["passed"])
        self.assertFalse(audit_quality_summary(with_warning)["passed"])
        self.assertEqual(audit_quality_summary(with_warning)["warning_count"], 2)

    def test_readiness_uses_only_declared_blockers(self) -> None:
        checks = {
            "required_artifacts": True,
            "data_identity": True,
            "quality_gate": True,
            "split_contract": True,
            "minimum_metrics": True,
            "generalization_gap": True,
            "calibration": True,
            "robustness": True,
            "explanation": True,
            "prediction_interface": True,
            "automated_tests": True,
            "reproduction_entrypoint": True,
        }

        ready = decide_phase_readiness(checks)
        checks["split_contract"] = False
        blocked = decide_phase_readiness(checks)

        self.assertEqual(ready["status"], "passed")
        self.assertTrue(ready["ready_for_first_kill_xgboost"])
        self.assertEqual(ready["blocking_failures"], [])
        self.assertEqual(blocked["status"], "failed")
        self.assertFalse(blocked["ready_for_first_kill_xgboost"])
        self.assertEqual(blocked["blocking_failures"], ["split_contract"])


if __name__ == "__main__":
    unittest.main()

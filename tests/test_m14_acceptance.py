import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import src.csdemo.m14_acceptance as m14_module

from src.csdemo.m14_acceptance import (
    assess_metric_targets,
    audit_data_identity,
    audit_required_artifacts,
    audit_quality_summary,
    audit_split_contract,
    build_split_assignments,
    decide_phase_readiness,
    fingerprint_file,
)


class M14AcceptanceTests(unittest.TestCase):
    def test_split_assignments_save_one_row_per_series(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": ["s1", "s1", "s2"],
                "game_id": ["g1", "g2", "g3"],
                "round_id": ["g1_1", "g2_1", "g3_1"],
                "split": ["train", "train", "test"],
                "ct_win": [1, 0, 1],
            }
        )

        result = build_split_assignments(frame)

        self.assertEqual(result["series_id"].tolist(), ["s1", "s2"])
        self.assertEqual(result.loc[0, "split"], "train")
        self.assertEqual(result.loc[0, "game_count"], 2)
        self.assertEqual(result.loc[0, "round_count"], 2)
        self.assertAlmostEqual(result.loc[0, "ct_win_rate"], 0.5)

    def test_module_entrypoint_runs_after_all_function_definitions(self) -> None:
        source = Path(m14_module.__file__).read_text(encoding="utf-8")

        self.assertGreater(
            source.rfind('if __name__ == "__main__":'),
            source.rfind("def decide_phase_readiness"),
        )

    def test_data_identity_connects_rounds_kills_and_model_rows(self) -> None:
        rounds = pd.DataFrame(
            {"round_id": ["g1_1", "g1_2"], "ct_win": [1, 0]}
        )
        kills = pd.DataFrame({"round_id": ["g1_1", "g1_1", "g1_2"]})
        pre_round = pd.DataFrame(
            {"round_id": ["g1_1", "g1_2"], "ct_win": [1, 0]}
        )

        result = audit_data_identity(rounds, kills, pre_round)

        self.assertTrue(result["passed"])
        self.assertEqual(result["round_rows"], 2)
        self.assertEqual(result["orphan_kills"], 0)

    def test_data_identity_rejects_orphan_kills_and_row_loss(self) -> None:
        rounds = pd.DataFrame({"round_id": ["g1_1"], "ct_win": [1]})
        kills = pd.DataFrame({"round_id": ["missing_1"]})
        pre_round = pd.DataFrame(columns=["round_id", "ct_win"])

        result = audit_data_identity(rounds, kills, pre_round)

        self.assertFalse(result["passed"])
        self.assertEqual(result["orphan_kills"], 1)
        self.assertIn("round and pre-round row counts differ", result["errors"])
        self.assertIn("kills contain orphan round_id values", result["errors"])

    def test_file_fingerprint_records_size_and_sha256(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"
            path.write_bytes(b"abc")

            result = fingerprint_file(path)

        self.assertEqual(result["bytes"], 3)
        self.assertEqual(
            result["sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_required_artifact_audit_lists_missing_relative_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present.txt").write_text("ok", encoding="utf-8")

            result = audit_required_artifacts(
                root, ["present.txt", "missing/report.json"]
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["present_count"], 1)
        self.assertEqual(result["required_count"], 2)
        self.assertEqual(result["missing"], ["missing/report.json"])

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
            "raw_source": True,
            "data_identity": True,
            "quality_gate": True,
            "split_contract": True,
            "baseline_models": True,
            "minimum_metrics": True,
            "generalization_gap": True,
            "calibration": True,
            "robustness": True,
            "explanation": True,
            "prediction_interface": True,
            "automated_tests": True,
            "environment_lock": True,
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

        checks["split_contract"] = True
        checks["environment_lock"] = False
        environment_blocked = decide_phase_readiness(checks)
        self.assertEqual(
            environment_blocked["blocking_failures"], ["environment_lock"]
        )


if __name__ == "__main__":
    unittest.main()

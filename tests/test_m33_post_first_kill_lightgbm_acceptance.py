import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.csdemo.m33_post_first_kill_lightgbm_acceptance import (
    BLOCKING_CHECKS,
    audit_frozen_metrics,
    audit_paired_uncertainty,
    audit_prediction_replay,
    audit_reproduction_entrypoint,
    audit_stage_chain,
    decide_acceptance,
    parse_unittest_count,
    run_acceptance,
)


def accepted_stage_summaries() -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for stage, next_stage in (
        ("M28", "M29"),
        ("M29", "M30"),
        ("M30", "M31"),
        ("M31", "M32"),
        ("M32", "M33"),
    ):
        summaries[stage] = {
            "stage": stage,
            "acceptance": {
                "status": "passed",
                f"ready_for_{next_stage.lower()}": True,
            },
        }
    return summaries


class M33PostFirstKillLightGBMAcceptanceTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_unittest_count_is_parsed_for_formal_report(self) -> None:
        self.assertEqual(parse_unittest_count("Ran 264 tests in 18.3s\n\nOK\n"), 264)
        self.assertIsNone(parse_unittest_count("no unittest summary"))

    def test_stage_chain_accepts_m28_through_m32(self) -> None:
        result = audit_stage_chain(accepted_stage_summaries())

        self.assertTrue(result["passed"])
        self.assertEqual(result["accepted_stages"], 5)
        self.assertEqual(result["failed_stages"], [])

    def test_stage_chain_rejects_missing_handoff(self) -> None:
        summaries = accepted_stage_summaries()
        summaries["M30"]["acceptance"]["ready_for_m31"] = False

        result = audit_stage_chain(summaries)

        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_stages"], ["M30"])

    def test_prediction_replay_joins_complete_keys_not_row_order(self) -> None:
        saved = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "game_id": ["g1", "g2"],
                "round_id": ["g1_1", "g2_1"],
                "y_true": [1, 0],
                "ct_win_probability": [0.8, 0.2],
            }
        )
        replayed = saved.iloc[::-1].reset_index(drop=True)

        result = audit_prediction_replay(saved, replayed)

        self.assertTrue(result["passed"])
        self.assertEqual(result["key_mismatch_count"], 0)
        self.assertEqual(result["label_mismatch_count"], 0)
        self.assertEqual(result["max_absolute_probability_difference"], 0.0)

    def test_prediction_replay_rejects_probability_drift(self) -> None:
        saved = pd.DataFrame(
            {
                "series_id": ["s1"],
                "game_id": ["g1"],
                "round_id": ["g1_1"],
                "y_true": [1],
                "ct_win_probability": [0.8],
            }
        )
        replayed = saved.copy()
        replayed["ct_win_probability"] = 0.800001

        result = audit_prediction_replay(saved, replayed, tolerance=1e-12)

        self.assertFalse(result["passed"])
        self.assertGreater(result["max_absolute_probability_difference"], 1e-12)

    def test_frozen_metrics_require_all_five_and_strict_tolerance(self) -> None:
        expected = {
            "accuracy": 0.74,
            "auc": 0.81,
            "log_loss": 0.52,
            "brier_score": 0.18,
            "ece10": 0.01,
        }

        self.assertTrue(audit_frozen_metrics(expected.copy(), expected)["passed"])
        self.assertFalse(
            audit_frozen_metrics({**expected, "auc": 0.810001}, expected)["passed"]
        )

    def test_paired_uncertainty_requires_honest_no_winner_claim(self) -> None:
        table = pd.DataFrame(
            {
                "metric": ["accuracy", "auc", "log_loss", "brier_score", "ece10"],
                "performance_advantage_ci_lower_95": [-0.01] * 5,
                "performance_advantage_ci_upper_95": [0.01] * 5,
                "ci_includes_zero": [True] * 5,
                "lightgbm_significantly_better": [False] * 5,
                "successful_bootstraps": [2000] * 5,
                "bootstrap_unit": ["series_id_paired"] * 5,
            }
        )
        summary = {
            "bootstrap": {"samples": 2000},
            "paired_comparison": {"significant_better_count": 0},
        }

        self.assertTrue(audit_paired_uncertainty(table, summary)["passed"])
        dishonest = table.copy()
        dishonest.loc[dishonest["metric"].eq("auc"), "ci_includes_zero"] = False
        self.assertFalse(audit_paired_uncertainty(dishonest, summary)["passed"])

    def test_reproduction_entrypoint_requires_three_modes_and_m28_to_m33(self) -> None:
        script = "\n".join(
            [
                "run_first_kill_pipeline.ps1",
                "run_post_first_kill_lightgbm_baseline.ps1",
                "run_post_first_kill_lightgbm_tuning.ps1",
                "run_post_first_kill_lightgbm_evaluation.ps1",
                "run_post_first_kill_lightgbm_explanation.ps1",
                "run_post_first_kill_lightgbm_interface.ps1",
                "src.csdemo.m33_post_first_kill_lightgbm_acceptance",
                "RebuildLightGBM",
                "FullRebuild",
            ]
        )

        self.assertTrue(audit_reproduction_entrypoint(script)["passed"])
        self.assertFalse(
            audit_reproduction_entrypoint(
                script.replace("run_post_first_kill_lightgbm_evaluation.ps1", "")
            )["passed"]
        )

    def test_final_decision_requires_every_declared_blocker(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}
        accepted = decide_acceptance(checks)
        checks["prediction_replay"] = False
        rejected = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["ready_for_teacher_report"])
        self.assertEqual(rejected["blocking_failures"], ["prediction_replay"])

    def test_real_artifact_runner_writes_complete_m33_evidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "m33"
            summary = run_acceptance(
                project_root=self.root,
                report_dir=report_dir,
                run_verification=False,
            )

            self.assertEqual(summary["acceptance"]["status"], "passed")
            self.assertTrue(summary["acceptance"]["ready_for_teacher_report"])
            self.assertEqual(summary["prediction_replay"]["replayed_rows"], 4170)
            self.assertLessEqual(
                summary["prediction_replay"]["max_absolute_probability_difference"],
                1e-12,
            )
            self.assertEqual(summary["lightgbm_fit_calls"], 0)
            for name in (
                "m33_summary.json",
                "m33_checks.csv",
                "m33_experiment_manifest.json",
                "m33_post_first_kill_lightgbm_final_acceptance_report.md",
                "runtime_environment.json",
                "split_assignments.csv",
                "replayed_test_predictions.csv",
                "fixed_test_metrics.csv",
                "paired_lightgbm_vs_xgboost_bootstrap.csv",
            ):
                self.assertTrue((report_dir / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()

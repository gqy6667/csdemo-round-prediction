import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.csdemo.m27_pre_round_lightgbm_acceptance import (
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
        ("M22", "M23"),
        ("M23", "M24"),
        ("M24", "M25"),
        ("M25", "M26"),
        ("M26", "M27"),
    ):
        summaries[stage] = {
            "stage": stage,
            "acceptance": {
                "status": "passed",
                f"ready_for_{next_stage.lower()}": True,
            },
        }
    return summaries


class M27PreRoundLightGBMAcceptanceTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_unittest_count_is_parsed_for_the_formal_report(self) -> None:
        output = "Ran 210 tests in 13.139s\n\nOK\n"

        self.assertEqual(parse_unittest_count(output), 210)
        self.assertIsNone(parse_unittest_count("no unittest summary"))

    def test_stage_chain_accepts_m22_through_m26(self) -> None:
        result = audit_stage_chain(accepted_stage_summaries())

        self.assertTrue(result["passed"])
        self.assertEqual(result["accepted_stages"], 5)
        self.assertEqual(result["failed_stages"], [])

    def test_stage_chain_rejects_missing_handoff(self) -> None:
        summaries = accepted_stage_summaries()
        summaries["M24"]["acceptance"]["ready_for_m25"] = False

        result = audit_stage_chain(summaries)

        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_stages"], ["M24"])

    def test_prediction_replay_joins_complete_keys_not_row_order(self) -> None:
        saved = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "game_id": ["g1", "g2"],
                "round_id": ["g1_1", "g2_1"],
                "ct_win": [1, 0],
                "lightgbm_tuned_probability": [0.8, 0.2],
            }
        )
        replayed = pd.DataFrame(
            {
                "series_id": ["s2", "s1"],
                "game_id": ["g2", "g1"],
                "round_id": ["g2_1", "g1_1"],
                "y_true": [0, 1],
                "ct_win_probability": [0.2, 0.8],
            }
        )

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
                "ct_win": [1],
                "lightgbm_tuned_probability": [0.8],
            }
        )
        replayed = saved.rename(
            columns={
                "ct_win": "y_true",
                "lightgbm_tuned_probability": "ct_win_probability",
            }
        )
        replayed["ct_win_probability"] = 0.800001

        result = audit_prediction_replay(saved, replayed, tolerance=1e-12)

        self.assertFalse(result["passed"])
        self.assertGreater(result["max_absolute_probability_difference"], 1e-12)

    def test_frozen_metrics_require_all_five_and_strict_tolerance(self) -> None:
        expected = {
            "accuracy": 0.65,
            "auc": 0.73,
            "log_loss": 0.59,
            "brier_score": 0.20,
            "ece10": 0.02,
        }

        stable = audit_frozen_metrics(expected.copy(), expected)
        drifted = audit_frozen_metrics(
            {**expected, "auc": expected["auc"] + 1e-6}, expected
        )
        missing = audit_frozen_metrics(
            {key: value for key, value in expected.items() if key != "ece10"},
            expected,
        )

        self.assertTrue(stable["passed"])
        self.assertEqual(stable["max_absolute_difference"], 0.0)
        self.assertFalse(drifted["passed"])
        self.assertFalse(missing["passed"])

    def test_paired_uncertainty_requires_five_series_bootstraps_and_honest_claim(self) -> None:
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
        m24 = {
            "bootstrap": {"samples": 2000},
            "paired_comparison": {"significant_better_count": 0},
        }

        accepted = audit_paired_uncertainty(table, m24)
        dishonest = table.copy()
        dishonest.loc[dishonest["metric"].eq("auc"), "lightgbm_significantly_better"] = True

        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["metric_count"], 5)
        self.assertEqual(accepted["significant_better_count"], 0)
        self.assertFalse(audit_paired_uncertainty(dishonest, m24)["passed"])

    def test_reproduction_entrypoint_requires_three_modes_and_m22_to_m27(self) -> None:
        script = "\n".join(
            [
                "run_pre_round_pipeline.ps1",
                "run_pre_round_lightgbm_baseline.ps1",
                "run_pre_round_lightgbm_tuning.ps1",
                "run_pre_round_lightgbm_evaluation.ps1",
                "run_pre_round_lightgbm_explanation.ps1",
                "run_pre_round_lightgbm_interface.ps1",
                "src.csdemo.m27_pre_round_lightgbm_acceptance",
                "RebuildLightGBM",
                "FullRebuild",
            ]
        )

        complete = audit_reproduction_entrypoint(script)
        incomplete = audit_reproduction_entrypoint(
            script.replace("run_pre_round_lightgbm_evaluation.ps1", "")
        )

        self.assertTrue(complete["passed"])
        self.assertEqual(complete["missing_tokens"], [])
        self.assertFalse(incomplete["passed"])

    def test_final_decision_requires_every_declared_blocker(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}
        accepted = decide_acceptance(checks)
        checks["prediction_replay"] = False
        rejected = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["ready_for_m28"])
        self.assertEqual(rejected["blocking_failures"], ["prediction_replay"])

    def test_real_artifact_runner_writes_complete_m27_evidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir) / "m27"
            summary = run_acceptance(
                project_root=self.root,
                report_dir=report_dir,
                run_verification=False,
            )

            self.assertEqual(summary["acceptance"]["status"], "passed")
            self.assertTrue(summary["acceptance"]["ready_for_m28"])
            self.assertEqual(summary["prediction_replay"]["replayed_rows"], 4172)
            self.assertLessEqual(
                summary["prediction_replay"]["max_absolute_probability_difference"],
                1e-12,
            )
            self.assertEqual(summary["lightgbm_fit_calls"], 0)
            for name in (
                "m27_summary.json",
                "m27_checks.csv",
                "m27_experiment_manifest.json",
                "m27_pre_round_lightgbm_final_acceptance_report.md",
                "runtime_environment.json",
                "split_assignments.csv",
                "replayed_test_predictions.csv",
                "fixed_test_metrics.csv",
                "paired_lightgbm_vs_xgboost_bootstrap.csv",
            ):
                self.assertTrue((report_dir / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()

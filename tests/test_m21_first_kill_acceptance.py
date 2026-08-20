import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import src.csdemo.m21_first_kill_acceptance as m21_module

from src.csdemo.m21_first_kill_acceptance import (
    BLOCKING_CHECKS,
    audit_first_kill_data,
    audit_formal_targets,
    audit_frozen_metrics,
    audit_prediction_replay,
    audit_reproduction_entrypoint,
    audit_stage_chain,
    build_progress_comparisons,
    build_progress_metrics,
    decide_acceptance,
    render_progress_report,
    run_acceptance,
)


def accepted_stage_summaries() -> dict[str, dict]:
    return {
        "M15": {"stage": "M15", "passed": True},
        "M16": {
            "stage": "M16",
            "acceptance": {"status": "passed", "ready_for_m17": True},
        },
        "M17": {
            "stage": "M17",
            "acceptance": {"status": "passed", "ready_for_m18": True},
        },
        "M18": {
            "stage": "M18",
            "acceptance": {"status": "passed", "ready_for_m19": True},
        },
        "M19": {
            "stage": "M19",
            "acceptance": {"status": "passed", "ready_for_m20": True},
        },
        "M20": {
            "stage": "M20",
            "acceptance": {"status": "passed", "ready_for_m21": True},
        },
    }


def tiny_first_kill_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["s1", "s1", "s2", "s3"],
            "game_id": ["g1", "g1", "g2", "g3"],
            "round_id": ["g1_1", "g1_2", "g2_1", "g3_1"],
            "split": ["train", "train", "val", "test"],
            "ct_win": [1, 0, 1, 0],
            "first_kill_advantage_ct": [1, -1, 1, -1],
            "first_kill_time": [12.0, 20.0, 8.0, 31.0],
            "first_kill_headshot": [0, 1, 0, 1],
            "first_kill_weapon": ["AK-47", "M4A1", "AWP", "Galil AR"],
        }
    )


class M21FirstKillAcceptanceTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_stage_chain_accepts_m15_through_m20(self) -> None:
        result = audit_stage_chain(accepted_stage_summaries())

        self.assertTrue(result["passed"])
        self.assertEqual(result["accepted_stages"], 6)
        self.assertEqual(result["failed_stages"], [])

    def test_stage_chain_rejects_a_missing_readiness_handoff(self) -> None:
        summaries = accepted_stage_summaries()
        summaries["M18"]["acceptance"]["ready_for_m19"] = False

        result = audit_stage_chain(summaries)

        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_stages"], ["M18"])

    def test_first_kill_data_audit_confirms_grouped_split_contract(self) -> None:
        result = audit_first_kill_data(
            tiny_first_kill_frame(),
            expected_split_rows={"train": 2, "val": 1, "test": 1},
            expected_split_series={"train": 1, "val": 1, "test": 1},
            required_features=[
                "first_kill_advantage_ct",
                "first_kill_time",
                "first_kill_headshot",
                "first_kill_weapon",
            ],
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["duplicate_key_rows"], 0)
        self.assertEqual(result["cross_split_series"], 0)
        self.assertEqual(result["cross_split_games"], 0)
        self.assertEqual(result["cross_split_rounds"], 0)

    def test_first_kill_data_audit_rejects_duplicate_and_cross_split_rows(self) -> None:
        frame = tiny_first_kill_frame()
        frame.loc[3, ["series_id", "game_id", "round_id"]] = ["s1", "g1", "g1_1"]

        result = audit_first_kill_data(
            frame,
            required_features=[
                "first_kill_advantage_ct",
                "first_kill_time",
                "first_kill_headshot",
                "first_kill_weapon",
            ],
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["duplicate_key_rows"], 1)
        self.assertGreater(result["cross_split_series"], 0)
        self.assertGreater(result["cross_split_games"], 0)
        self.assertGreater(result["cross_split_rounds"], 0)

    def test_prediction_replay_matches_by_complete_key_not_row_order(self) -> None:
        saved = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "game_id": ["g1", "g2"],
                "round_id": ["g1_1", "g2_1"],
                "ct_win": [1, 0],
                "xgboost_tuned_probability": [0.8, 0.2],
            }
        )
        replayed = pd.DataFrame(
            {
                "series_id": ["s2", "s1"],
                "game_id": ["g2", "g1"],
                "round_id": ["g2_1", "g1_1"],
                "ct_win": [0, 1],
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
                "xgboost_tuned_probability": [0.8],
            }
        )
        replayed = saved.rename(
            columns={"xgboost_tuned_probability": "ct_win_probability"}
        ).copy()
        replayed["ct_win_probability"] = 0.8000001

        result = audit_prediction_replay(saved, replayed, tolerance=1e-12)

        self.assertFalse(result["passed"])
        self.assertGreater(result["max_absolute_probability_difference"], 1e-12)

    def test_frozen_metric_audit_uses_strict_tolerance(self) -> None:
        expected = {"accuracy": 0.74, "auc": 0.81, "log_loss": 0.52}

        stable = audit_frozen_metrics(expected.copy(), expected, tolerance=1e-12)
        drifted = audit_frozen_metrics(
            {**expected, "auc": expected["auc"] + 1e-6},
            expected,
            tolerance=1e-12,
        )

        self.assertTrue(stable["passed"])
        self.assertEqual(stable["max_absolute_difference"], 0.0)
        self.assertFalse(drifted["passed"])

    def test_formal_target_audit_requires_identical_ten_passed_targets(self) -> None:
        rows = [
            {
                "target_id": f"target_{index}",
                "current": 0.8,
                "target": 0.7,
                "remaining": 0.0,
                "passed": True,
            }
            for index in range(10)
        ]
        m19 = {
            "passed_count": 10,
            "target_count": 10,
            "remaining_count": 0,
            "all_formal_targets_passed": True,
            "rows": rows,
        }

        accepted = audit_formal_targets(m19, {**m19, "rows": [dict(row) for row in rows]})
        changed = {**m19, "rows": [dict(row) for row in rows]}
        changed["rows"][0]["target"] = 0.6

        self.assertTrue(accepted["passed"])
        self.assertFalse(audit_formal_targets(m19, changed)["passed"])

    def test_progress_tables_separate_fair_and_timing_comparisons(self) -> None:
        m6 = {"accuracy": 0.6462, "auc": 0.7220, "log_loss": 0.5938}
        pre_round = {
            "accuracy": 0.6474,
            "auc": 0.7271,
            "log_loss": 0.5917,
            "brier_score": 0.2053,
            "ece10": 0.0232,
        }
        m16 = pd.DataFrame(
            [
                {
                    "model": "logistic_regression",
                    "split": "test",
                    "accuracy": 0.7434,
                    "auc": 0.8091,
                    "log_loss": 0.5266,
                    "brier_score": 0.1761,
                    "ece10": 0.0150,
                },
                {
                    "model": "xgboost_untuned",
                    "split": "test",
                    "accuracy": 0.7453,
                    "auc": 0.8089,
                    "log_loss": 0.5248,
                    "brier_score": 0.1763,
                    "ece10": 0.0109,
                },
            ]
        )
        current = {
            "accuracy": 0.7441,
            "auc": 0.8098,
            "log_loss": 0.5231,
            "brier_score": 0.1757,
            "ece10": 0.0154,
        }

        metrics = build_progress_metrics(m6, pre_round, m16, current)
        comparisons = build_progress_comparisons(metrics)

        self.assertEqual(len(metrics), 5)
        self.assertEqual(len(comparisons), 4)
        pre_round_change = comparisons.set_index("comparison_id").loc[
            "m6_to_m14_pre_round"
        ]
        timing_change = comparisons.set_index("comparison_id").loc[
            "m14_to_m21_prediction_time"
        ]
        self.assertEqual(pre_round_change["comparability"], "same_task_same_split")
        self.assertAlmostEqual(pre_round_change["auc_change"], 0.0051)
        self.assertEqual(timing_change["comparability"], "timing_change_not_algorithm_only")
        self.assertAlmostEqual(timing_change["auc_change"], 0.0827)

    def test_progress_report_explains_that_first_kill_gain_is_not_tuning_only(self) -> None:
        metrics = pd.DataFrame(
            [{"stage": "M6", "auc": 0.722}, {"stage": "M21", "auc": 0.8098}]
        )
        comparisons = pd.DataFrame(
            [
                {
                    "comparison_id": "m14_to_m21_prediction_time",
                    "comparability": "timing_change_not_algorithm_only",
                    "auc_change": 0.0827,
                }
            ]
        )

        report = render_progress_report(
            metrics,
            comparisons,
            context={
                "test_count": 145,
                "blocking_passed": 17,
                "blocking_total": 17,
                "data": {
                    "rows": 41027,
                    "series": 782,
                    "games": 1558,
                    "split_rows": {"train": 28489, "val": 8368, "test": 4170},
                    "split_series": {"train": 547, "val": 156, "test": 79},
                    "split_percentages": {
                        "train": 69.44,
                        "val": 20.40,
                        "test": 10.16,
                    },
                    "cross_split_series": 0,
                    "cross_split_games": 0,
                    "cross_split_rounds": 0,
                },
                "formal_target_rows": [
                    {
                        "label": "Test AUC",
                        "current": 0.8098,
                        "target": 0.78,
                        "remaining": 0.0,
                        "margin": 0.0298,
                        "passed": True,
                    }
                ],
                "closest_external": [
                    {
                        "source_title": "CS156 first-kill logistic regression",
                        "metric": "auc",
                        "current_value": 0.8091,
                        "reported_value": 0.76,
                        "raw_difference_ours_minus_reported": 0.0491,
                    }
                ],
            },
        )

        self.assertIn("M6", report)
        self.assertIn("M21", report)
        self.assertIn("不是纯调参增益", report)
        self.assertIn("LightGBM", report)
        self.assertIn("数据与切分进展", report)
        self.assertIn("M6-M21 模块记录", report)
        self.assertIn("首杀后正式目标", report)
        self.assertIn("外部模型差距", report)
        self.assertIn("工程成熟度变化", report)

    def test_reproduction_entrypoint_requires_all_three_modes_and_m15_to_m21(self) -> None:
        stage_scripts = "\n".join(
            [
                "run_pre_round_pipeline.ps1",
                "run_first_kill_data_stage.ps1",
                "run_first_kill_baselines.ps1",
                "run_first_kill_tuning.ps1",
                "run_first_kill_evaluation.ps1",
                "run_first_kill_explanation.ps1",
                "run_first_kill_interface.ps1",
                "src.csdemo.m21_first_kill_acceptance",
                "FullRebuild",
                "RebuildFirstKill",
            ]
        )

        complete = audit_reproduction_entrypoint(stage_scripts)
        incomplete = audit_reproduction_entrypoint(
            stage_scripts.replace("run_first_kill_tuning.ps1", "")
        )

        self.assertTrue(complete["passed"])
        self.assertEqual(complete["missing_tokens"], [])
        self.assertFalse(incomplete["passed"])
        self.assertIn("run_first_kill_tuning.ps1", incomplete["missing_tokens"])

    def test_final_decision_requires_every_declared_blocker(self) -> None:
        checks = {name: True for name in BLOCKING_CHECKS}

        accepted = decide_acceptance(checks)
        checks["prediction_replay"] = False
        rejected = decide_acceptance(checks)

        self.assertEqual(accepted["status"], "passed")
        self.assertTrue(accepted["first_kill_xgboost_complete"])
        self.assertEqual(rejected["blocking_failures"], ["prediction_replay"])
        self.assertFalse(rejected["ready_for_lightgbm_comparison"])

    def test_module_entrypoint_follows_all_function_definitions(self) -> None:
        source = Path(m21_module.__file__).read_text(encoding="utf-8")

        self.assertGreater(
            source.rfind('if __name__ == "__main__":'),
            source.rfind("def decide_acceptance"),
        )

    def test_real_artifact_runner_writes_complete_m21_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            report_dir = temporary / "m21"
            progress_path = temporary / "m6_to_m21_progress_report.md"

            summary = run_acceptance(
                project_root=self.root,
                esta_root=Path(r"C:\project1\data\esta"),
                report_dir=report_dir,
                progress_report_path=progress_path,
                run_tests=False,
            )

            self.assertEqual(summary["acceptance"]["status"], "passed")
            self.assertEqual(summary["acceptance"]["blocking_passed"], 17)
            self.assertTrue(summary["acceptance"]["first_kill_xgboost_complete"])
            self.assertLessEqual(
                summary["prediction_replay"][
                    "max_absolute_probability_difference"
                ],
                1e-12,
            )
            self.assertEqual(summary["formal_targets"]["passed_count"], 10)
            self.assertEqual(summary["xgboost_fit_calls"], 0)
            self.assertTrue(progress_path.is_file())
            for filename in (
                "m21_summary.json",
                "m21_checks.csv",
                "m21_experiment_manifest.json",
                "m21_first_kill_final_acceptance_report.md",
                "runtime_environment.json",
                "split_assignments.csv",
                "external_benchmark_comparison.csv",
                "external_benchmark_comparison.md",
                "m6_to_m21_stage_metrics.csv",
                "m6_to_m21_metric_changes.csv",
            ):
                self.assertTrue((report_dir / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()

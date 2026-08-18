import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.csdemo.features import make_first_kill_samples
from src.csdemo.m15_first_kill_data import (
    apply_split_manifest,
    audit_first_kill_data,
    fingerprint_file,
    resolve_previous_comparison,
)


def fixture_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rounds = pd.DataFrame(
        [
            {
                "series_id": "s1",
                "game_id": "g1",
                "round_id": "g1_1",
                "ct_win": 1,
                "ct_alive": 5,
                "t_alive": 5,
            },
            {
                "series_id": "s1",
                "game_id": "g1",
                "round_id": "g1_2",
                "ct_win": 0,
                "ct_alive": 5,
                "t_alive": 5,
            },
        ]
    )
    kills = pd.DataFrame(
        [
            {
                "series_id": "s1",
                "game_id": "g1",
                "round_id": "g1_1",
                "time": 18.0,
                "tick": 1_800,
                "killer_side": "CT",
                "victim_side": "T",
                "weapon": "M4A4",
                "headshot": 1,
                "is_first_kill": 1,
            }
        ]
    )
    manifest = pd.DataFrame([{"series_id": "s1", "split": "train"}])
    return rounds, kills, manifest


class M15FirstKillDataTests(unittest.TestCase):
    def test_split_manifest_is_applied_without_resplitting(self) -> None:
        samples = pd.DataFrame(
            {
                "series_id": ["s2", "s1"],
                "round_id": ["g2_1", "g1_1"],
                "split": ["old", "old"],
            }
        )
        manifest = pd.DataFrame(
            {
                "series_id": ["s1", "s2"],
                "split": ["test", "train"],
            }
        )

        result = apply_split_manifest(samples, manifest)

        self.assertEqual(result["split"].tolist(), ["train", "test"])

    def test_split_manifest_rejects_an_unassigned_series(self) -> None:
        samples = pd.DataFrame({"series_id": ["missing"], "round_id": ["g1_1"]})
        manifest = pd.DataFrame({"series_id": ["s1"], "split": ["train"]})

        with self.assertRaisesRegex(ValueError, "missing from the M14 split manifest"):
            apply_split_manifest(samples, manifest)

    def test_audit_passes_and_records_rounds_without_valid_kills(self) -> None:
        rounds, kills, manifest = fixture_tables()
        samples = apply_split_manifest(make_first_kill_samples(rounds, kills), manifest)

        summary, excluded = audit_first_kill_data(rounds, kills, samples, manifest)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["counts"]["sample_rows"], 1)
        self.assertEqual(summary["counts"]["excluded_rounds"], 1)
        self.assertEqual(excluded.iloc[0]["round_id"], "g1_2")
        self.assertEqual(excluded.iloc[0]["reason"], "no_valid_enemy_kill")

    def test_audit_rejects_an_event_from_the_wrong_kill(self) -> None:
        rounds, kills, manifest = fixture_tables()
        samples = apply_split_manifest(make_first_kill_samples(rounds, kills), manifest)
        samples.loc[0, "first_kill_weapon"] = "AWP"

        summary, _ = audit_first_kill_data(rounds, kills, samples, manifest)

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["event_linkage"]["passed"])
        self.assertEqual(summary["checks"]["event_linkage"]["mismatch_rows"], 1)

    def test_audit_rejects_a_split_that_differs_from_m14(self) -> None:
        rounds, kills, manifest = fixture_tables()
        samples = apply_split_manifest(make_first_kill_samples(rounds, kills), manifest)
        samples.loc[0, "split"] = "test"

        summary, _ = audit_first_kill_data(rounds, kills, samples, manifest)

        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["split_manifest"]["passed"])
        self.assertEqual(summary["checks"]["split_manifest"]["mismatch_rows"], 1)

    def test_rerun_preserves_the_original_repair_comparison(self) -> None:
        frame = pd.DataFrame(
            {
                "series_id": ["s1"],
                "game_id": ["g1"],
                "round_id": ["g1_1"],
                "first_kill_time": [10.0],
            }
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "first_kill.parquet"
            reports = root / "reports"
            reports.mkdir()
            output.write_bytes(b"stable-m15-artifact")
            saved = {
                "data_artifact": fingerprint_file(output),
                "previous_dataset_comparison": {
                    "available": True,
                    "event_changed_rows": 14_357,
                },
            }
            (reports / "m15_summary.json").write_text(
                json.dumps(saved), encoding="utf-8"
            )

            result = resolve_previous_comparison(
                frame, frame, output_path=output, report_dir=reports
            )

        self.assertEqual(result["event_changed_rows"], 14_357)
        self.assertTrue(result["preserved_from_first_m15_run"])


if __name__ == "__main__":
    unittest.main()

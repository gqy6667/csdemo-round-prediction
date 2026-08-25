import csv
import json
import re
import unittest
from pathlib import Path


class TeacherReviewReportTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports" / "teacher_review"

    def load_json(self, relative_path: str) -> dict:
        return json.loads((self.root / relative_path).read_text(encoding="utf-8"))

    def assert_local_markdown_links_exist(self, report_path: Path) -> None:
        text = report_path.read_text(encoding="utf-8")
        targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        local_targets = [
            target.split("#", maxsplit=1)[0]
            for target in targets
            if target
            and not target.startswith(("http://", "https://", "mailto:", "#"))
        ]

        self.assertGreater(len(local_targets), 0)
        missing = [
            target
            for target in local_targets
            if not (report_path.parent / target).resolve().exists()
        ]
        self.assertEqual(missing, [])

    def test_pre_round_xgboost_report_matches_frozen_sources(self) -> None:
        report_path = self.report_dir / "01_pre_round_xgboost_report.md"
        report = report_path.read_text(encoding="utf-8")
        m9 = self.load_json("reports/esta_full_m9/m9_summary.json")
        m10 = self.load_json("reports/esta_full_m10/m10_summary.json")
        m12 = self.load_json("reports/esta_full_m12/m12_summary.json")
        m13 = self.load_json("reports/esta_full_m13/m13_summary.json")
        m14 = self.load_json("reports/esta_full_m14/m14_experiment_manifest.json")

        self.assertIn("购买结束、交火前", report)
        self.assertIn("冻结时间结束后的第一个可用快照", report)
        self.assertIn("测试集不参与模型选择、调参或校准器选择", report)
        self.assertIn("不能把两个预测时点的指标差解释为算法优劣", report)

        for value in m9["metrics"].values():
            self.assertIn(f"{value:.6f}", report)

        with (self.root / "reports/esta_full_m9/bootstrap_95ci.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            confidence_intervals = list(csv.DictReader(handle))
        self.assertEqual(len(confidence_intervals), 5)
        for row in confidence_intervals:
            interval = (
                f"[{float(row['ci_lower_95']):.6f}, "
                f"{float(row['ci_upper_95']):.6f}]"
            )
            self.assertIn(interval, report)

        split = m14["data"]["split_contract"]
        self.assertIn(f"{m14['data']['identity']['pre_round_rows']:,}", report)
        for count in split["series_counts"].values():
            self.assertIn(f"{count:,}", report)
        for count in split["row_counts"].values():
            self.assertIn(f"{count:,}", report)
        self.assertEqual(split["cross_split_series"], 0)
        self.assertEqual(split["cross_split_games"], 0)
        self.assertEqual(split["cross_split_rounds"], 0)

        artifacts = m14["artifacts"]
        for name in ("pre_round", "model", "calibrator"):
            self.assertIn(artifacts[name]["sha256"], report)

        self.assertIn(str(m14["stage_evidence"]["training"]["best_tree_count"]), report)
        self.assertIn(m10["selected_method"], report)
        self.assertIn(str(m12["encoded_features"]), report)
        self.assertIn(
            str(m13["input_contract"]["pre_encoding_feature_count"]), report
        )
        self.assertIn("非因果", report)
        self.assertIn("未达到更高阶段目标", report)
        self.assert_local_markdown_links_exist(report_path)


if __name__ == "__main__":
    unittest.main()

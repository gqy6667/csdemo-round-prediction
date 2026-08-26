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

    def test_pre_round_lightgbm_report_matches_frozen_sources(self) -> None:
        report_path = self.report_dir / "02_pre_round_lightgbm_report.md"
        report = report_path.read_text(encoding="utf-8")
        m22 = self.load_json("reports/esta_full_m22/m22_summary.json")
        m23 = self.load_json("reports/esta_full_m23/m23_summary.json")
        m24 = self.load_json("reports/esta_full_m24/m24_summary.json")
        m25 = self.load_json("reports/esta_full_m25/m25_summary.json")
        m27 = self.load_json("reports/esta_full_m27/m27_summary.json")
        manifest = self.load_json(
            "reports/esta_full_m27/m27_experiment_manifest.json"
        )

        self.assertIn("购买结束、交火前", report)
        self.assertIn("只替换模型算法", report)
        self.assertIn("测试集不参与拟合、早停、调参或校准器选择", report)
        self.assertIn("不能宣称 LightGBM 显著优于 XGBoost", report)
        self.assertIn("不能把两个预测时点的指标差解释为算法优劣", report)

        for value in m27["metrics"].values():
            self.assertIn(f"{value:.6f}", report)

        with (self.root / "reports/esta_full_m24/global_bootstrap_95ci.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            global_intervals = list(csv.DictReader(handle))
        self.assertEqual(len(global_intervals), 5)
        for row in global_intervals:
            interval = (
                f"[{float(row['ci_lower_95']):.6f}, "
                f"{float(row['ci_upper_95']):.6f}]"
            )
            self.assertIn(interval, report)

        paired_path = (
            self.root
            / "reports/esta_full_m24/paired_lightgbm_vs_xgboost_bootstrap.csv"
        )
        with paired_path.open(newline="", encoding="utf-8") as handle:
            paired_intervals = list(csv.DictReader(handle))
        self.assertEqual(len(paired_intervals), 5)
        for row in paired_intervals:
            self.assertIn(
                f"{float(row['performance_advantage_lightgbm']):.6f}", report
            )
            interval = (
                f"[{float(row['performance_advantage_ci_lower_95']):.6f}, "
                f"{float(row['performance_advantage_ci_upper_95']):.6f}]"
            )
            self.assertIn(interval, report)
            self.assertEqual(row["ci_includes_zero"], "True")

        for count in m27["data"]["split_rows"].values():
            self.assertIn(f"{count:,}", report)
        for count in m27["data"]["split_series"].values():
            self.assertIn(f"{count:,}", report)
        for sha256 in m27["artifacts"].values():
            if isinstance(sha256, str):
                self.assertIn(sha256, report)

        self.assertEqual(m22["features"]["raw_count"], 36)
        self.assertEqual(m22["features"]["encoded_count"], 43)
        self.assertIn(str(m22["features"]["raw_count"]), report)
        self.assertIn(str(m22["features"]["encoded_count"]), report)
        self.assertIn(str(m23["search"]["phase_count"]), report)
        self.assertIn(str(m23["search"]["candidate_count"]), report)
        self.assertIn(str(m23["search"]["accepted_change_count"]), report)
        self.assertIn(str(m27["model_contract"]["deployment_tree_count"]), report)
        self.assertIn(m24["calibration"]["selected_method"], report)
        self.assertIn(
            f"{m25['shap_reconstruction_max_abs_error']:.6e}", report
        )
        self.assertIn("非因果", report)
        self.assertIn("19/19", report)

        pipeline = next(
            item
            for item in manifest["inputs"]
            if item["path"].endswith("run_pre_round_lightgbm_pipeline.ps1")
        )
        self.assertIn(pipeline["sha256"], report)
        self.assert_local_markdown_links_exist(report_path)

    def test_post_first_kill_xgboost_report_matches_frozen_sources(self) -> None:
        report_path = self.report_dir / "03_post_first_kill_xgboost_report.md"
        report = report_path.read_text(encoding="utf-8")
        m15 = self.load_json("reports/esta_full_m15/m15_summary.json")
        m16 = self.load_json("reports/esta_full_m16/m16_summary.json")
        m17 = self.load_json("reports/esta_full_m17/m17_summary.json")
        m18 = self.load_json("reports/esta_full_m18/m18_summary.json")
        m19 = self.load_json("reports/esta_full_m19/m19_summary.json")
        m21 = self.load_json("reports/esta_full_m21/m21_summary.json")
        manifest = self.load_json("reports/esta_full_m21/m21_experiment_manifest.json")

        self.assertIn("购买完成后最早一次有效敌对击杀刚刚发生", report)
        self.assertIn("不是算法提升", report)
        self.assertIn("测试集只在参数和种子协议冻结后评估一次", report)
        self.assertIn("公平比较见第四份报告", report)

        for value in m21["metrics"].values():
            self.assertIn(f"{value:.6f}", report)

        with (self.root / "reports/esta_full_m18/global_bootstrap_95ci.csv").open(
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

        self.assertIn(f"{m15['counts']['sample_rows']:,}", report)
        self.assertIn(str(m15["counts"]["excluded_rounds"]), report)
        self.assertIn(
            f"{m15['previous_dataset_comparison']['event_changed_rows']:,}", report
        )
        for count in m21["data"]["split_rows"].values():
            self.assertIn(f"{count:,}", report)
        for count in m21["data"]["split_series"].values():
            self.assertIn(f"{count:,}", report)

        self.assertIn(m21["data"]["sha256"], report)
        self.assertIn(m21["artifacts"]["model"]["sha256"], report)
        self.assertIn(m21["artifacts"]["calibrator"]["sha256"], report)
        self.assertIn(str(m16["features"]["raw_count"]), report)
        self.assertIn(str(m16["features"]["encoded_count"]), report)
        self.assertIn(str(m17["tuning"]["phase_count"]), report)
        self.assertIn(str(m17["tuning"]["candidate_count"]), report)
        self.assertIn(str(m17["frozen_model"]["best_tree_count"]), report)
        self.assertIn(m18["calibration"]["selected_method"], report)
        self.assertIn(
            f"{m19['shap_reconstruction_max_abs_error']:.6e}", report
        )
        self.assertIn("非因果", report)
        self.assertIn("17/17", report)
        self.assertIn(
            manifest["artifact_fingerprints"][
                "scripts/run_first_kill_pipeline.ps1"
            ]["sha256"],
            report,
        )
        self.assert_local_markdown_links_exist(report_path)

    def test_post_first_kill_lightgbm_report_matches_frozen_sources(self) -> None:
        report_path = self.report_dir / "04_post_first_kill_lightgbm_report.md"
        report = report_path.read_text(encoding="utf-8")
        m28 = self.load_json("reports/esta_full_m28/m28_summary.json")
        m29 = self.load_json("reports/esta_full_m29/m29_summary.json")
        m30 = self.load_json("reports/esta_full_m30/m30_summary.json")
        m31 = self.load_json("reports/esta_full_m31/m31_summary.json")
        m32 = self.load_json("reports/esta_full_m32/m32_summary.json")
        m33 = self.load_json("reports/esta_full_m33/m33_summary.json")
        manifest = self.load_json(
            "reports/esta_full_m33/m33_experiment_manifest.json"
        )

        self.assertIn("购买完成后最早一次有效敌对击杀刚刚发生", report)
        self.assertIn("只替换模型算法", report)
        self.assertIn("测试集不参与拟合、早停、调参或校准器选择", report)
        self.assertIn("不能宣称 LightGBM 或 XGBoost 稳定显著更优", report)
        self.assertIn("不能把两个预测时点的指标差解释为算法优劣", report)

        for value in m33["metrics"].values():
            self.assertIn(f"{value:.6f}", report)

        with (self.root / "reports/esta_full_m30/global_bootstrap_95ci.csv").open(
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

        with (
            self.root
            / "reports/esta_full_m30/paired_lightgbm_vs_xgboost_bootstrap.csv"
        ).open(newline="", encoding="utf-8") as handle:
            paired_intervals = list(csv.DictReader(handle))
        self.assertEqual(len(paired_intervals), 5)
        for row in paired_intervals:
            self.assertIn(
                f"{float(row['performance_advantage_lightgbm']):.6f}", report
            )
            interval = (
                f"[{float(row['performance_advantage_ci_lower_95']):.6f}, "
                f"{float(row['performance_advantage_ci_upper_95']):.6f}]"
            )
            self.assertIn(interval, report)
            self.assertEqual(row["ci_includes_zero"], "True")

        for count in m33["data"]["split_rows"].values():
            self.assertIn(f"{count:,}", report)
        for count in m33["data"]["split_series"].values():
            self.assertIn(f"{count:,}", report)
        for sha256 in m33["artifacts"].values():
            if isinstance(sha256, str):
                self.assertIn(sha256, report)

        self.assertEqual(m28["features"]["raw_count"], 40)
        self.assertEqual(m28["features"]["encoded_count"], 82)
        self.assertIn(str(m28["features"]["raw_count"]), report)
        self.assertIn(str(m28["features"]["encoded_count"]), report)
        self.assertIn(str(m29["search"]["phase_count"]), report)
        self.assertIn(str(m29["search"]["candidate_count"]), report)
        self.assertIn(str(m29["search"]["accepted_change_count"]), report)
        self.assertIn(str(m33["model_contract"]["deployment_tree_count"]), report)
        self.assertIn(m30["calibration"]["selected_method"], report)
        self.assertIn(
            f"{m31['shap_reconstruction_max_abs_error']:.6e}", report
        )
        self.assertIn(str(m32["input_contract"]["required_input_field_count"]), report)
        self.assertIn("非因果", report)
        self.assertIn("19/19", report)

        pipeline = next(
            item
            for item in manifest["inputs"]
            if item["path"].endswith(
                "run_post_first_kill_lightgbm_pipeline.ps1"
            )
        )
        self.assertIn(pipeline["sha256"], report)
        self.assert_local_markdown_links_exist(report_path)

    def test_teacher_review_index_links_four_reports_and_keeps_fair_groups(self) -> None:
        index_path = self.report_dir / "README.md"
        index = index_path.read_text(encoding="utf-8")

        expected_reports = (
            "01_pre_round_xgboost_report.md",
            "02_pre_round_lightgbm_report.md",
            "03_post_first_kill_xgboost_report.md",
            "04_post_first_kill_lightgbm_report.md",
        )
        for report in expected_reports:
            self.assertIn(report, index)
        self.assertEqual(index.count("已完成并可复核"), 4)
        self.assertIn("购买结束、交火前：XGBoost vs LightGBM", index)
        self.assertIn("首杀后：XGBoost vs LightGBM", index)
        self.assertIn("两个预测时点不能直接排行", index)
        self.assertIn("五项配对区间均包含 0", index)
        self.assert_local_markdown_links_exist(index_path)


if __name__ == "__main__":
    unittest.main()

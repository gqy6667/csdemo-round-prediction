from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from .benchmark_comparison import compare_benchmarks, write_markdown_report
from .io import read_table
from .m9_evaluation import METRIC_ORDER, bootstrap_metric_intervals
from .m10_calibration import (
    calibration_curves,
    cross_validated_comparison,
    evaluate_test_calibrators,
    fit_full_calibrators,
    select_calibration_method,
)
from .m11_robustness import group_metrics_with_intervals
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .m16_first_kill_baselines import (
    canonical_feature_names,
    prepare_profile_splits,
)
from .m18_first_kill_evaluation import (
    assess_calibration,
    assess_global_intervals,
    assess_group_robustness,
    bootstrap_source_auc_gap,
    enrich_high_confidence_errors,
    parse_source_subset,
    prepare_analysis_table,
    summarize_high_confidence_errors,
)
from .m24_pre_round_lightgbm_evaluation import (
    audit_calibration_protocol,
    audit_prediction_replay,
    build_paired_prediction_table,
    paired_model_bootstrap,
    run_compile_check,
)
from .m28_post_first_kill_lightgbm_baseline import (
    audit_data_contract,
    write_json,
)
from .metrics import probability_metrics
from .schema import ID_COLUMNS


KEY_COLUMNS = tuple(ID_COLUMNS)

BLOCKING_CHECKS = (
    "m29_prerequisite",
    "split_and_key_contract",
    "frozen_model_replay",
    "prediction_replay",
    "global_bootstrap",
    "global_metric_minimum",
    "paired_comparison",
    "group_outputs",
    "source_stability",
    "large_map_minimum",
    "calibration_protocol",
    "calibration_no_material_harm",
    "error_review",
    "external_report",
    "automated_tests",
    "source_compile",
    "reproduction_entrypoint",
    "artifact_manifest",
)


def verify_m29_prerequisite(
    data_path: str | Path,
    model_path: str | Path,
    m29_summary: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    data_artifact = fingerprint_file(data_path)
    model_artifact = fingerprint_file(model_path)
    expected_data_sha = m29_summary.get("data", {}).get("sha256")
    expected_model_sha = (
        m29_summary.get("model", {}).get("model_artifact", {}).get("sha256")
    )
    raw_features = list(bundle.get("raw_features", []))
    encoded_columns = list(bundle.get("columns", []))
    checks = {
        "m29_accepted": (
            m29_summary.get("acceptance", {}).get("status") == "passed"
            and m29_summary.get("acceptance", {}).get("ready_for_m30") is True
        ),
        "data_sha256": bool(expected_data_sha)
        and data_artifact["sha256"] == expected_data_sha
        and bundle.get("data_sha256") == expected_data_sha,
        "model_sha256": bool(expected_model_sha)
        and model_artifact["sha256"] == expected_model_sha,
        "task": bundle.get("task") == "post_first_kill",
        "model_name": bundle.get("model_name") == "lightgbm_tuned",
        "raw_feature_contract": raw_features == canonical_feature_names()
        and len(raw_features) == int(
            m29_summary.get("features", {}).get("raw_count", -1)
        ),
        "encoded_feature_contract": bool(encoded_columns)
        and len(encoded_columns) == len(set(encoded_columns))
        and len(encoded_columns)
        == int(m29_summary.get("features", {}).get("encoded_count", -1)),
        "parameter_contract": dict(bundle.get("params", {}))
        == dict(m29_summary.get("model", {}).get("params", {})),
        "predictor_available": hasattr(bundle.get("model"), "predict_proba"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "data_artifact": data_artifact,
        "model_artifact": model_artifact,
        "expected_data_sha256": expected_data_sha,
        "expected_model_sha256": expected_model_sha,
    }


def replay_frozen_model(
    data: pd.DataFrame,
    bundle: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_features = list(bundle["raw_features"])
    prepared = prepare_profile_splits(data, raw_features)
    encoded_columns = prepared["train"][0].columns.tolist()
    expected_columns = list(bundle["columns"])
    if encoded_columns != expected_columns:
        raise ValueError("M30 encoded feature columns do not match the M29 model")

    outputs: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        x, y, identity = prepared[split]
        probability = np.asarray(
            bundle["model"].predict_proba(x)[:, 1], dtype=float
        )
        if not np.isfinite(probability).all() or (
            (probability < 0) | (probability > 1)
        ).any():
            raise ValueError("M30 replay produced invalid probabilities")
        output = identity.copy()
        output["y_true"] = y.to_numpy(dtype=int)
        output["ct_win_probability"] = probability
        output["t_win_probability"] = 1.0 - probability
        output["predicted_label"] = (probability >= 0.5).astype(int)
        output["correct"] = output["predicted_label"].eq(output["y_true"])
        metadata = data.loc[
            data["split"].eq(split), list(KEY_COLUMNS) + ["map_name"]
        ].copy()
        output = output.merge(
            metadata,
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        output["source_subset"] = parse_source_subset(output["game_id"])
        outputs[split] = output

    return outputs, {
        "passed": True,
        "raw_feature_count": len(raw_features),
        "encoded_feature_count": len(encoded_columns),
        "encoded_columns_match_m29": encoded_columns == expected_columns,
        "split_rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "lightgbm_fit_calls": 0,
    }


def audit_reproduction_entrypoint(script_path: Path) -> dict[str, Any]:
    if not script_path.is_file():
        return {"passed": False, "missing_tokens": [script_path.as_posix()]}
    source = script_path.read_text(encoding="utf-8")
    required = (
        "src.csdemo.m30_post_first_kill_lightgbm_evaluation",
        "post_first_kill_lightgbm_tuned.joblib",
        "m29_summary.json",
        "[int]$BootstrapSamples = 2000",
    )
    missing = [token for token in required if token not in source]
    return {"passed": not missing, "missing_tokens": missing}


def decide_acceptance(checks: Mapping[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not bool(checks.get(name, False))]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "blocking_passed": len(BLOCKING_CHECKS) - len(failures),
        "blocking_total": len(BLOCKING_CHECKS),
        "m30_lightgbm_evaluation_complete": not failures,
        "ready_for_m31": not failures,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _manifest_key(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _collect_git_state(project_root: Path, report_dir: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain").splitlines()
    try:
        report_prefix = report_dir.relative_to(project_root).as_posix().rstrip("/") + "/"
    except ValueError:
        report_prefix = None
    retained = [
        line
        for line in status
        if report_prefix is None
        or report_prefix not in line[3:].replace("\\", "/")
    ]
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "remote": git("remote", "get-url", "origin"),
        "working_tree_status": retained,
    }


def _write_case_review(cases: pd.DataFrame, path: Path) -> None:
    lines = [
        "# M30 首杀后 LightGBM 高置信错误复核",
        "",
        "定义：预测错误且预测方概率不低于 0.80。模式是事后描述，不是因果结论。",
        "",
    ]
    for index, row in cases.reset_index(drop=True).iterrows():
        lines.append(
            f"{index + 1}. `{row['series_id']} / {row['game_id']} / "
            f"{row['round_id']}`：预测 {row['predicted_side']} "
            f"({row['assigned_side_probability']:.3f})，实际 {row['actual_winner']}；"
            f"首杀 {row['first_kill_side']}，时间 {row['first_kill_time']:.2f}s，"
            f"武器 {row['first_kill_weapon']}，装备差 CT "
            f"{row['eq_value_diff_ct']:+.0f}；`{row['signal_pattern']}`。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_source_auc_gap(source_gap: Mapping[str, Any]) -> str:
    return (
        f"LAN-online AUC 差为 `{float(source_gap['signed_difference']):+.6f}`，"
        f"95% CI `[{float(source_gap['ci_lower_95']):+.6f}, "
        f"{float(source_gap['ci_upper_95']):+.6f}]`。"
    )


def _render_report(
    summary: Mapping[str, Any],
    global_intervals: pd.DataFrame,
    paired: pd.DataFrame,
    map_metrics: pd.DataFrame,
    source_metrics: pd.DataFrame,
    validation_calibration: pd.DataFrame,
    test_calibration: pd.DataFrame,
    external: pd.DataFrame,
) -> str:
    lines = [
        "# M30 首杀后 LightGBM 冻结模型评估报告",
        "",
        "## 阶段结论",
        "",
        f"阻断验收：**{summary['acceptance']['status']}**（"
        f"{summary['acceptance']['blocking_passed']}/"
        f"{summary['acceptance']['blocking_total']}）；可进入 M31："
        f"**{summary['acceptance']['ready_for_m31']}**。",
        "本阶段回放 M29 冻结模型，LightGBM `fit` 调用为 0；测试集不参与模型、"
        "参数或校准方法选择。",
        "",
        "## 冻结合同",
        "",
        f"- 数据：{summary['data']['rows']:,} 行，SHA-256 "
        f"`{summary['data']['sha256']}`。",
        f"- train/val/test 行数：{summary['data']['split_rows']}；系列赛："
        f"{summary['data']['split_series']}。",
        f"- 模型 SHA-256：`{summary['prerequisite']['model_artifact']['sha256']}`。",
        f"- 4,170 条测试概率最大回放误差："
        f"`{summary['prediction_replay']['max_absolute_probability_difference']:.3e}`。",
        "",
        "## 整体指标与系列赛级 95% CI",
        "",
        "| 指标 | 点估计 | 95% CI | 成功次数 |",
        "|---|---:|---:|---:|",
    ]
    for row in global_intervals.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['point_estimate']:.6f} | "
            f"[{row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}] | "
            f"{int(row['successful_bootstraps'])} |"
        )

    lines.extend(
        [
            "",
            "## 与 M21 XGBoost 的公平配对比较",
            "",
            "性能优势统一为正值表示 LightGBM 更好。是否显著只由配对 95% CI 是否"
            "排除 0 决定，不由点指标决定。",
            "",
            "| 指标 | LightGBM | XGBoost | 优势 | 95% CI | 显著更好 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in paired.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['lightgbm']:.6f} | "
            f"{row['xgboost']:.6f} | {row['performance_advantage_lightgbm']:+.6f} | "
            f"[{row['performance_advantage_ci_lower_95']:+.6f}, "
            f"{row['performance_advantage_ci_upper_95']:+.6f}] | "
            f"{row['lightgbm_significantly_better']} |"
        )

    lines.extend(
        [
            "",
            "## 主要地图稳健性",
            "",
            "| 地图 | 回合 | 系列赛 | AUC | AUC 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in map_metrics.sort_values("auc").to_dict(orient="records"):
        lines.append(
            f"| {row['map_name']} | {int(row['rounds'])} | {int(row['series'])} | "
            f"{row['auc']:.6f} | [{row['auc_ci_lower_95']:.6f}, "
            f"{row['auc_ci_upper_95']:.6f}] |"
        )

    lines.extend(
        [
            "",
            "## LAN/online 稳健性",
            "",
            "| 来源 | 回合 | 系列赛 | AUC | Log Loss |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in source_metrics.to_dict(orient="records"):
        lines.append(
            f"| {row['source_subset']} | {int(row['rounds'])} | "
            f"{int(row['series'])} | {row['auc']:.6f} | {row['log_loss']:.6f} |"
        )
    gap = summary["source_auc_gap"]
    lines.extend(
        [
            "",
            format_source_auc_gap(gap),
            "",
            "## 校准",
            "",
            "| 方法 | Validation OOF Log Loss | Brier | ECE10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in validation_calibration.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['log_loss']:.6f} | "
            f"{row['brier_score']:.6f} | {row['ece10']:.6f} |"
        )
    selected = summary["calibration"]["selected_method"]
    lines.extend(
        [
            "",
            f"Validation OOF 选择 **`{selected}`**。冻结后测试比较：",
            "",
            "| 方法 | Log Loss | Brier | ECE10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in test_calibration.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['log_loss']:.6f} | "
            f"{row['brier_score']:.6f} | {row['ece10']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 错误分析",
            "",
            f"高置信错误共 **{summary['errors']['available']}** 个，已复核前 "
            f"**{summary['errors']['reviewed']}** 个。最常见信号组合只作描述，"
            "不作为返回调参的依据。",
            "",
            "## 与公开结果的边界",
            "",
            f"外部比较表共 {len(external)} 行。公开工作在数据、切分或预测时点上不同，"
            "不能据此建立纯算法排行榜。",
            "",
            "## 下一阶段",
            "",
            "M31 冻结 M29 模型和本阶段校准决定，执行 Gain、Permutation、TreeSHAP、"
            "泄漏审计和与 M19 XGBoost 的解释差异分析。",
            "",
            "复现命令：",
            "",
            "```powershell",
            "powershell -ExecutionPolicy Bypass -File "
            ".\\scripts\\run_post_first_kill_lightgbm_evaluation.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    project_root: str | Path,
    data_path: str | Path = "data/processed/esta_full/first_kill.parquet",
    model_path: str | Path = "models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib",
    m29_summary_path: str | Path = "reports/esta_full_m29/m29_summary.json",
    m29_predictions_path: str | Path = "reports/esta_full_m29/test_predictions.csv",
    benchmarks_path: str | Path = "benchmarks/external_first_kill_tuned_metrics.csv",
    model_dir: str | Path = "models/esta_full_m30",
    report_dir: str | Path = "reports/esta_full_m30",
    n_bootstrap: int = 2000,
    seed: int = 42,
    n_splits: int = 5,
    review_cases: int = 30,
    run_tests: bool = True,
    run_compile: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_path = _resolve(root, data_path)
    model_path = _resolve(root, model_path)
    m29_summary_path = _resolve(root, m29_summary_path)
    m29_predictions_path = _resolve(root, m29_predictions_path)
    benchmarks_path = _resolve(root, benchmarks_path)
    model_dir = _resolve(root, model_dir)
    report_dir = _resolve(root, report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m29_summary = _read_json(m29_summary_path)
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError("M30 expected the M29 model artifact to contain a bundle")
    prerequisite = verify_m29_prerequisite(
        data_path, model_path, m29_summary, bundle
    )
    if not prerequisite["passed"]:
        raise RuntimeError("M30 input does not match the accepted M29 artifacts")

    data = read_table(data_path)
    data_audit = audit_data_contract(data)
    split_and_key_contract = bool(
        data_audit["passed"]
        and data_audit["split_rows"] == m29_summary["data"]["split_rows"]
        and data_audit["split_series"] == m29_summary["data"]["split_series"]
    )
    if not split_and_key_contract:
        raise RuntimeError("M30 data identity or split membership differs from M29")

    replayed, model_replay = replay_frozen_model(data, bundle)
    saved_predictions = read_table(m29_predictions_path)
    prediction_replay = audit_prediction_replay(
        saved_predictions, replayed["test"], tolerance=1e-12
    )
    if not prediction_replay["passed"]:
        raise RuntimeError("M30 could not exactly replay the M29 test probabilities")

    analysis = prepare_analysis_table(replayed["test"], data)
    global_intervals = bootstrap_metric_intervals(
        analysis, n_bootstrap=n_bootstrap, seed=seed
    )
    global_assessment = assess_global_intervals(
        global_intervals, n_bootstrap=n_bootstrap
    )
    paired_predictions = build_paired_prediction_table(
        replayed["test"], saved_predictions
    )
    paired = paired_model_bootstrap(
        paired_predictions, n_bootstrap=n_bootstrap, seed=seed
    )
    paired_complete = bool(
        set(paired["metric"]) == set(METRIC_ORDER)
        and paired["successful_bootstraps"].eq(n_bootstrap).all()
    )

    group_columns = {
        "map": "map_name",
        "source": "source_subset",
        "round_stage": "round_stage",
        "equipment_band": "equipment_band",
        "first_kill_side": "first_kill_side",
        "first_kill_time_band": "first_kill_time_band",
        "first_kill_weapon_family": "first_kill_weapon_family",
        "first_kill_headshot": "first_kill_headshot_label",
    }
    grouped = {
        output_name: group_metrics_with_intervals(
            analysis, column, n_bootstrap=n_bootstrap, seed=seed
        )
        for output_name, column in group_columns.items()
    }
    source_gap = bootstrap_source_auc_gap(
        analysis, n_bootstrap=n_bootstrap, seed=seed
    )
    robustness = assess_group_robustness(grouped, source_gap)

    all_errors = enrich_high_confidence_errors(analysis)
    reviewed = all_errors.head(review_cases).copy()
    error_summary = summarize_high_confidence_errors(all_errors)

    validation_comparison, validation_oof = cross_validated_comparison(
        replayed["val"], n_splits=n_splits
    )
    selected_method = select_calibration_method(validation_comparison)
    calibrators = fit_full_calibrators(replayed["val"])
    test_calibration, calibrated_test = evaluate_test_calibrators(
        replayed["test"], calibrators, selected_method
    )
    curves = calibration_curves(calibrated_test)
    calibration_protocol = audit_calibration_protocol(
        replayed["val"],
        validation_comparison,
        validation_oof,
        selected_method,
        n_splits=n_splits,
    )
    calibration_assessment = assess_calibration(
        test_calibration, selected_method
    )

    calibrator_path = model_dir / "post_first_kill_lightgbm_calibrator.joblib"
    joblib.dump(
        {
            "calibrator": calibrators[selected_method],
            "method": selected_method,
            "task": "post_first_kill",
            "base_model_path": model_path.as_posix(),
            "base_model_sha256": prerequisite["model_artifact"]["sha256"],
            "data_sha256": prerequisite["data_artifact"]["sha256"],
            "selection_data": "validation only",
            "validation_folds": n_splits,
        },
        calibrator_path,
    )
    calibrator_artifact = fingerprint_file(calibrator_path)

    current_metrics = probability_metrics(
        analysis["y_true"], analysis["ct_win_probability"], n_bins=10
    )
    metric_replay_max_difference = max(
        abs(float(current_metrics[name]) - float(m29_summary["metrics"][name]))
        for name in METRIC_ORDER
    )
    external_benchmarks = read_table(benchmarks_path)
    external = compare_benchmarks(current_metrics, external_benchmarks)
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    write_markdown_report(
        external,
        current_metrics,
        report_dir / "external_benchmark_comparison.md",
        stage_label="M30 post-first-kill LightGBM",
    )

    global_intervals.to_csv(report_dir / "global_bootstrap_95ci.csv", index=False)
    paired.to_csv(
        report_dir / "paired_lightgbm_vs_xgboost_bootstrap.csv", index=False
    )
    analysis.to_csv(report_dir / "test_predictions_enriched.csv", index=False)
    for name, table in grouped.items():
        table.to_csv(report_dir / f"metrics_by_{name}_with_ci.csv", index=False)
    pd.DataFrame([source_gap]).to_csv(report_dir / "source_auc_gap.csv", index=False)
    validation_comparison.to_csv(
        report_dir / "validation_oof_calibration.csv", index=False
    )
    validation_oof.to_csv(
        report_dir / "validation_oof_predictions.csv", index=False
    )
    test_calibration.to_csv(
        report_dir / "test_calibration_comparison.csv", index=False
    )
    calibrated_test.to_csv(
        report_dir / "calibrated_test_predictions.csv", index=False
    )
    curves.to_csv(report_dir / "calibration_curves.csv", index=False)
    all_errors.to_csv(report_dir / "all_high_confidence_errors.csv", index=False)
    reviewed.to_csv(report_dir / "reviewed_top30_errors.csv", index=False)
    error_summary.to_csv(report_dir / "error_pattern_summary.csv", index=False)
    _write_case_review(reviewed, report_dir / "top30_error_review.md")

    script_path = root / "scripts/run_post_first_kill_lightgbm_evaluation.ps1"
    reproduction_entrypoint = audit_reproduction_entrypoint(script_path)
    if run_tests:
        automated = run_automated_tests(root)
        match = re.search(r"Ran (\d+) tests?", automated["output"])
        test_count = int(match.group(1)) if match else None
    else:
        automated = {
            "passed": True,
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "command": [],
            "output": "Skipped by caller; exercised separately.\n",
        }
        test_count = None
    compile_result = (
        run_compile_check(root)
        if run_compile
        else {
            "passed": True,
            "return_code": 0,
            "command": [],
            "output": "Skipped by caller; exercised separately.\n",
        }
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated["output"], encoding="utf-8"
    )
    (report_dir / "source_compile_output.txt").write_text(
        compile_result["output"], encoding="utf-8"
    )

    core_output_names = (
        "global_bootstrap_95ci.csv",
        "paired_lightgbm_vs_xgboost_bootstrap.csv",
        "test_predictions_enriched.csv",
        *(f"metrics_by_{name}_with_ci.csv" for name in group_columns),
        "source_auc_gap.csv",
        "validation_oof_calibration.csv",
        "validation_oof_predictions.csv",
        "test_calibration_comparison.csv",
        "calibrated_test_predictions.csv",
        "calibration_curves.csv",
        "all_high_confidence_errors.csv",
        "reviewed_top30_errors.csv",
        "error_pattern_summary.csv",
        "top30_error_review.md",
        "external_benchmark_comparison.csv",
        "external_benchmark_comparison.md",
        "automated_test_output.txt",
        "source_compile_output.txt",
    )
    artifact_manifest_passed = bool(
        calibrator_path.is_file()
        and all((report_dir / name).is_file() for name in core_output_names)
    )
    checks = {
        "m29_prerequisite": prerequisite["passed"],
        "split_and_key_contract": split_and_key_contract,
        "frozen_model_replay": (
            model_replay["passed"] and model_replay["lightgbm_fit_calls"] == 0
        ),
        "prediction_replay": (
            prediction_replay["passed"] and metric_replay_max_difference <= 1e-12
        ),
        "global_bootstrap": global_assessment["bootstrap_complete"],
        "global_metric_minimum": global_assessment["minimum_passed"],
        "paired_comparison": paired_complete,
        "group_outputs": robustness["outputs_complete"],
        "source_stability": robustness["source_gap_passed"],
        "large_map_minimum": robustness["large_map_minimum_passed"],
        "calibration_protocol": calibration_protocol["passed"],
        "calibration_no_material_harm": calibration_assessment[
            "no_material_probability_metric_harm"
        ],
        "error_review": len(reviewed) >= review_cases,
        "external_report": (
            not external.empty
            and (report_dir / "external_benchmark_comparison.md").is_file()
        ),
        "automated_tests": automated["passed"],
        "source_compile": compile_result["passed"],
        "reproduction_entrypoint": reproduction_entrypoint["passed"],
        "artifact_manifest": artifact_manifest_passed,
    }
    acceptance = decide_acceptance(checks)
    significant_metrics = paired.loc[
        paired["lightgbm_significantly_better"], "metric"
    ].tolist()
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "stage": "M30",
        "generated_at_utc": generated_at,
        "task": "post_first_kill",
        "definition": "purchase complete, immediately after earliest valid enemy kill",
        "model_policy": "M29 LightGBM frozen; no LightGBM training or tuning in M30",
        "acceptance": acceptance,
        "checks": checks,
        "stage_targets": {
            "global_interval_target": global_assessment["stage_passed"],
            "large_map_auc_target": robustness["large_map_stage_passed"],
            "large_map_ci_target": robustness["large_map_ci_stage_passed"],
            "source_gap_ci_includes_zero": robustness["source_gap_ci_includes_zero"],
            "calibration_ece_target": calibration_assessment[
                "test_ece_stage_passed"
            ],
        },
        "prerequisite": prerequisite,
        "data": {
            "path": data_path.as_posix(),
            "sha256": prerequisite["data_artifact"]["sha256"],
            "rows": int(len(data)),
            "series": int(data["series_id"].nunique()),
            "games": int(data["game_id"].nunique()),
            "split_rows": data_audit["split_rows"],
            "split_series": data_audit["split_series"],
            "cross_split_series": data_audit["cross_split_series"],
            "cross_split_games": data_audit["cross_split_games"],
            "cross_split_rounds": data_audit["cross_split_rounds"],
            "duplicate_key_rows": data_audit["duplicate_key_rows"],
            "key_columns": list(KEY_COLUMNS),
        },
        "model_replay": model_replay,
        "prediction_replay": prediction_replay,
        "metric_replay_max_absolute_difference": metric_replay_max_difference,
        "metrics": {name: float(current_metrics[name]) for name in METRIC_ORDER},
        "bootstrap": {
            "unit": "series_id",
            "samples": n_bootstrap,
            "seed": seed,
        },
        "global_assessment": global_assessment,
        "paired_comparison": {
            "bootstrap_complete": paired_complete,
            "bootstrap_unit": "series_id_paired",
            "significant_better_count": len(significant_metrics),
            "significant_better_metrics": significant_metrics,
            "superiority_required_for_acceptance": False,
        },
        "robustness": robustness,
        "source_auc_gap": source_gap,
        "group_table_rows": {
            name: int(len(table)) for name, table in grouped.items()
        },
        "calibration_protocol": calibration_protocol,
        "calibration": {
            **calibration_assessment,
            "calibrator_artifact": calibrator_artifact,
            "validation_oof_comparison": json.loads(
                validation_comparison.to_json(orient="records")
            ),
            "test_comparison": json.loads(
                test_calibration.to_json(orient="records")
            ),
        },
        "errors": {
            "definition": "wrong and assigned predicted-side probability >= 0.80",
            "available": int(len(all_errors)),
            "reviewed": int(len(reviewed)),
            "review_target": review_cases,
            "signal_pattern_counts": {
                str(name): int(count)
                for name, count in all_errors["signal_pattern"].value_counts().items()
            },
        },
        "external_comparison_rows": int(len(external)),
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "lightgbm_version": importlib.metadata.version("lightgbm"),
            "device_type": bundle["model"].get_params().get("device_type"),
            "cuda_required": False,
        },
        "automated_tests": {
            "passed": automated["passed"],
            "return_code": automated["return_code"],
            "elapsed_seconds": automated["elapsed_seconds"],
            "test_count": test_count,
            "skipped": not run_tests,
        },
        "source_compile": {
            "passed": compile_result["passed"],
            "return_code": compile_result["return_code"],
            "skipped": not run_compile,
        },
        "reproduction_entrypoint": reproduction_entrypoint,
        "next_stage": "M31 post-first-kill LightGBM explanation and leakage audit",
    }
    summary["stage_targets"]["all_passed"] = all(
        summary["stage_targets"].values()
    )
    pd.DataFrame(
        [
            {"check": name, "passed": bool(value), "blocking": True}
            for name, value in checks.items()
        ]
    ).to_csv(report_dir / "m30_checks.csv", index=False)
    summary_path = report_dir / "m30_summary.json"
    report_path = report_dir / "m30_post_first_kill_lightgbm_evaluation_report.md"
    write_json(summary, summary_path)
    report_path.write_text(
        _render_report(
            summary,
            global_intervals,
            paired,
            grouped["map"],
            grouped["source"],
            validation_comparison,
            test_calibration,
            external,
        ),
        encoding="utf-8",
    )

    input_paths = {
        "first_kill_data": data_path,
        "m29_model": model_path,
        "m29_summary": m29_summary_path,
        "m29_test_predictions": m29_predictions_path,
        "external_benchmarks": benchmarks_path,
        "m30_spec": root / "docs/m30_post_first_kill_lightgbm_evaluation_spec.md",
        "m30_module": root / "src/csdemo/m30_post_first_kill_lightgbm_evaluation.py",
        "m30_tests": root / "tests/test_m30_post_first_kill_lightgbm_evaluation.py",
        "reproduction_script": script_path,
        "environment": root / "environment.yml",
        "requirements_lock": root / "requirements-lock.txt",
    }
    output_paths = [
        calibrator_path,
        *(report_dir / name for name in core_output_names),
        report_dir / "m30_checks.csv",
        summary_path,
        report_path,
    ]
    manifest = {
        "stage": "M30",
        "generated_at_utc": generated_at,
        "policy": "M29 model frozen; grouped evaluation and validation-only calibration",
        "code": _collect_git_state(root, report_dir),
        "inputs": {name: fingerprint_file(path) for name, path in input_paths.items()},
        "outputs": {
            _manifest_key(root, path): fingerprint_file(path) for path in output_paths
        },
        "contract": {
            "model_frozen": True,
            "lightgbm_fit_calls": 0,
            "key_columns": list(KEY_COLUMNS),
            "bootstrap_unit": "series_id",
            "bootstrap_samples": n_bootstrap,
            "paired_model_bootstrap": True,
            "calibration_selection_data": "validation only",
            "calibration_folds": n_splits,
            "group_columns": group_columns,
        },
        "checks": checks,
        "acceptance": acceptance,
    }
    write_json(manifest, report_dir / "m30_experiment_manifest.json")
    if acceptance["status"] != "passed":
        raise RuntimeError(
            "M30 acceptance failed: " + ", ".join(acceptance["blocking_failures"])
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M30 frozen post-first-kill LightGBM evaluation."
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data", default="data/processed/esta_full/first_kill.parquet")
    parser.add_argument(
        "--model",
        default="models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib",
    )
    parser.add_argument("--m29-summary", default="reports/esta_full_m29/m29_summary.json")
    parser.add_argument(
        "--m29-predictions", default="reports/esta_full_m29/test_predictions.csv"
    )
    parser.add_argument(
        "--benchmarks", default="benchmarks/external_first_kill_tuned_metrics.csv"
    )
    parser.add_argument("--model-dir", default="models/esta_full_m30")
    parser.add_argument("--report-dir", default="reports/esta_full_m30")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--review-cases", type=int, default=30)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    summary = run(
        project_root=args.project_root,
        data_path=args.data,
        model_path=args.model,
        m29_summary_path=args.m29_summary,
        m29_predictions_path=args.m29_predictions,
        benchmarks_path=args.benchmarks,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
        n_splits=args.calibration_folds,
        review_cases=args.review_cases,
        run_tests=not args.skip_tests,
        run_compile=not args.skip_compile,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

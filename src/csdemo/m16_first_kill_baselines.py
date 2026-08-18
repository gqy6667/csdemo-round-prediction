from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import LABEL_COL
from .benchmark_comparison import compare_benchmarks
from .m7_baselines import make_constant_model, make_logistic_model
from .io import read_table
from .m15_first_kill_data import fingerprint_file, run_automated_tests
from .metrics import probability_metrics
from .schema import ID_COLUMNS, PRE_ROUND_FEATURES
from .train_xgb import align_columns, make_model, prepare_xy


SPLIT_ORDER = ("train", "val", "test")
FORMAL_MODEL_NAMES = (
    "constant_train_prior",
    "logistic_regression",
    "xgboost_untuned",
)
FIRST_KILL_MODEL_FEATURES = [
    "first_kill_advantage_ct",
    "first_kill_time",
    "first_kill_headshot",
    "first_kill_weapon",
]
REDUNDANT_FIRST_KILL_FEATURES = [
    "first_kill_is_ct",
    "first_death_is_ct",
    "ct_alive_after_fk",
    "t_alive_after_fk",
    "alive_diff_ct_after_fk",
]
METRIC_TARGETS = {
    "accuracy": {"minimum": 0.68, "stage": 0.70, "higher_is_better": True},
    "auc": {"minimum": 0.75, "stage": 0.78, "higher_is_better": True},
    "log_loss": {"minimum": 0.58, "stage": 0.55, "higher_is_better": False},
    "brier_score": {"minimum": 0.20, "stage": 0.185, "higher_is_better": False},
}
BLOCKING_CHECKS = (
    "m15_artifact",
    "data_contract",
    "feature_contract",
    "model_probabilities",
    "frozen_xgboost",
    "minimum_metrics",
    "automated_tests",
    "external_report",
)
HIGHER_IS_BETTER = {"accuracy", "auc"}
REPORT_METRICS = ("accuracy", "auc", "log_loss", "brier_score", "ece10")


def canonical_feature_names() -> list[str]:
    return [*PRE_ROUND_FEATURES, *FIRST_KILL_MODEL_FEATURES]


def build_feature_contract() -> pd.DataFrame:
    rows = [
        {
            "feature": feature,
            "group": "pre_round",
            "included": True,
            "reason": "accepted_M14_purchase_end_feature",
        }
        for feature in PRE_ROUND_FEATURES
    ]
    rows.extend(
        {
            "feature": feature,
            "group": "first_kill",
            "included": True,
            "reason": "canonical_post_first_kill_signal",
        }
        for feature in FIRST_KILL_MODEL_FEATURES
    )
    rows.extend(
        {
            "feature": feature,
            "group": "first_kill",
            "included": False,
            "reason": "deterministic_redundancy",
        }
        for feature in REDUNDANT_FIRST_KILL_FEATURES
    )
    return pd.DataFrame(rows)


def audit_training_data(df: pd.DataFrame) -> dict[str, Any]:
    required = set(ID_COLUMNS + [LABEL_COL, "split"])
    missing = sorted(required - set(df.columns))
    if missing:
        return {
            "passed": False,
            "missing_columns": missing,
            "duplicate_key_rows": 0,
            "cross_split_series": 0,
            "invalid_split_rows": int(len(df)),
            "null_identity_cells": 0,
            "split_rows": {},
        }

    duplicate_key_rows = int(df.duplicated(ID_COLUMNS).sum())
    cross_split_series = int((df.groupby("series_id")["split"].nunique() > 1).sum())
    invalid_split_rows = int((~df["split"].isin(SPLIT_ORDER)).sum())
    null_identity_cells = int(df[ID_COLUMNS].isna().sum().sum())
    split_rows = {
        split: int(df["split"].eq(split).sum()) for split in SPLIT_ORDER
    }
    all_splits_present = all(count > 0 for count in split_rows.values())
    passed = (
        duplicate_key_rows == 0
        and cross_split_series == 0
        and invalid_split_rows == 0
        and null_identity_cells == 0
        and all_splits_present
    )
    return {
        "passed": passed,
        "missing_columns": [],
        "duplicate_key_rows": duplicate_key_rows,
        "cross_split_series": cross_split_series,
        "invalid_split_rows": invalid_split_rows,
        "null_identity_cells": null_identity_cells,
        "all_splits_present": all_splits_present,
        "split_rows": split_rows,
    }


def prepare_profile_splits(
    df: pd.DataFrame, feature_names: list[str]
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Feature profile contains duplicate column names")
    forbidden = sorted(set(feature_names) & set(ID_COLUMNS + [LABEL_COL, "split"]))
    if forbidden:
        raise ValueError(f"Feature profile contains forbidden columns: {forbidden}")
    missing = sorted(set(feature_names) - set(df.columns))
    if missing:
        raise KeyError(f"Training data is missing profile features: {missing}")

    prepared: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
    for split in SPLIT_ORDER:
        split_df = df.loc[df["split"].eq(split)]
        if split_df.empty:
            raise ValueError(f"M16 data has no rows for split: {split}")
        model_frame = split_df[ID_COLUMNS + [LABEL_COL, "split"] + feature_names]
        x, y = prepare_xy(model_frame)
        prepared[split] = (x, y, split_df[ID_COLUMNS].reset_index(drop=True))

    reference = prepared["train"][0]
    for split in ("val", "test"):
        x, y, identity = prepared[split]
        prepared[split] = (align_columns(reference, x), y, identity)
    return prepared


def make_formal_models() -> dict[str, Any]:
    return {
        "constant_train_prior": make_constant_model(),
        "logistic_regression": make_logistic_model(),
        "xgboost_untuned": make_model(task="first_kill"),
    }


def fit_formal_models(
    prepared: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
) -> dict[str, Any]:
    models = make_formal_models()
    x_train, y_train, _ = prepared["train"]
    x_val, y_val, _ = prepared["val"]
    models["constant_train_prior"].fit(x_train, y_train)
    models["logistic_regression"].fit(x_train, y_train)
    models["xgboost_untuned"].fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return models


def evaluate_models(
    models: dict[str, Any],
    prepared: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    probabilities: dict[str, dict[str, np.ndarray]] = {
        split: {} for split in SPLIT_ORDER
    }
    for model_name in FORMAL_MODEL_NAMES:
        model = models[model_name]
        for split in SPLIT_ORDER:
            x, y, _ = prepared[split]
            probability = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
            probabilities[split][model_name] = probability
            rows.append(
                {
                    "model": model_name,
                    "profile": "canonical_event",
                    "split": split,
                    **probability_metrics(y, probability, n_bins=10),
                }
            )
    return pd.DataFrame(rows), probabilities


def fit_pre_round_control(
    prepared: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
) -> Any:
    model = make_model(task="first_kill")
    x_train, y_train, _ = prepared["train"]
    x_val, y_val, _ = prepared["val"]
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def build_feature_control_comparison(
    canonical_comparison: pd.DataFrame,
    control_model: Any,
    control_prepared: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
) -> pd.DataFrame:
    canonical = canonical_comparison.loc[
        canonical_comparison["model"].eq("xgboost_untuned")
    ].copy()
    canonical["profile"] = "canonical_event"
    rows = [canonical]
    control_rows = []
    for split in SPLIT_ORDER:
        x, y, _ = control_prepared[split]
        probability = np.asarray(control_model.predict_proba(x)[:, 1], dtype=float)
        control_rows.append(
            {
                "model": "xgboost_untuned",
                "profile": "pre_round_control",
                "split": split,
                **probability_metrics(y, probability, n_bins=10),
            }
        )
    rows.append(pd.DataFrame(control_rows))
    return pd.concat(rows, ignore_index=True)


def assess_metric_targets(metrics: dict[str, float]) -> dict[str, Any]:
    assessed: dict[str, dict[str, Any]] = {}
    for name, target in METRIC_TARGETS.items():
        value = float(metrics[name])
        if target["higher_is_better"]:
            minimum_passed = value >= target["minimum"]
            stage_passed = value >= target["stage"]
            minimum_gap = max(target["minimum"] - value, 0.0)
            stage_gap = max(target["stage"] - value, 0.0)
        else:
            minimum_passed = value <= target["minimum"]
            stage_passed = value <= target["stage"]
            minimum_gap = max(value - target["minimum"], 0.0)
            stage_gap = max(value - target["stage"], 0.0)
        assessed[name] = {
            "value": value,
            **target,
            "minimum_passed": bool(minimum_passed),
            "stage_passed": bool(stage_passed),
            "minimum_gap": float(minimum_gap),
            "stage_gap": float(stage_gap),
        }
    return {
        "all_minimum_passed": all(item["minimum_passed"] for item in assessed.values()),
        "all_stage_passed": all(item["stage_passed"] for item in assessed.values()),
        "minimum_passed_count": sum(item["minimum_passed"] for item in assessed.values()),
        "stage_passed_count": sum(item["stage_passed"] for item in assessed.values()),
        "metrics": assessed,
    }


def build_prediction_table(
    test_rows: pd.DataFrame, probabilities: dict[str, np.ndarray]
) -> pd.DataFrame:
    result = test_rows[ID_COLUMNS + [LABEL_COL]].reset_index(drop=True).copy()
    for model_name in FORMAL_MODEL_NAMES:
        if model_name not in probabilities:
            raise KeyError(f"Missing probabilities for model: {model_name}")
        values = np.asarray(probabilities[model_name], dtype=float).reshape(-1)
        if len(values) != len(result):
            raise ValueError(f"Probability row count differs for model: {model_name}")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Probabilities for {model_name} must be finite and between 0 and 1")
        result[f"{model_name}_probability"] = values
        result[f"{model_name}_prediction"] = (values >= 0.5).astype(int)
    return result


def compare_external_models(
    local_comparison: pd.DataFrame, benchmarks: pd.DataFrame
) -> pd.DataFrame:
    if "current_model" not in benchmarks.columns:
        raise KeyError("External first-kill benchmarks require current_model")
    if benchmarks["benchmark_id"].duplicated().any():
        raise ValueError("External benchmark_id values must be unique")

    test = local_comparison.loc[local_comparison["split"].eq("test")]
    results = []
    for source_order, (model_name, group) in enumerate(
        benchmarks.groupby("current_model", sort=False)
    ):
        model_rows = test.loc[test["model"].eq(model_name)]
        if len(model_rows) != 1:
            raise ValueError(
                f"Expected one test metric row for external comparison model: {model_name}"
            )
        local_metrics = {
            column: float(model_rows.iloc[0][column])
            for column in ("accuracy", "auc", "log_loss", "brier_score", "ece10")
            if column in model_rows.columns
        }
        compared = compare_benchmarks(local_metrics, group.copy())
        compared["_model_group_order"] = source_order
        results.append(compared)
    if not results:
        raise ValueError("External first-kill benchmark table must not be empty")
    return (
        pd.concat(results, ignore_index=True)
        .sort_values(["_model_group_order"])
        .drop(columns="_model_group_order")
        .reset_index(drop=True)
    )


def verify_m15_artifact(
    data_path: str | Path, m15_summary: dict[str, Any]
) -> dict[str, Any]:
    expected = m15_summary.get("data_artifact", {})
    expected_sha256 = expected.get("sha256")
    expected_bytes = expected.get("bytes")
    actual = fingerprint_file(data_path)
    passed = (
        bool(expected_sha256)
        and actual["sha256"] == expected_sha256
        and (expected_bytes is None or actual["bytes"] == expected_bytes)
    )
    return {
        "passed": passed,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual["sha256"],
        "expected_bytes": expected_bytes,
        "actual_bytes": actual["bytes"],
    }


def decide_acceptance(checks: dict[str, bool]) -> dict[str, Any]:
    failures = [name for name in BLOCKING_CHECKS if not checks.get(name, False)]
    return {
        "status": "passed" if not failures else "failed",
        "blocking_failures": failures,
        "ready_for_m17": not failures,
    }


def model_metric_differences(
    comparison: pd.DataFrame,
    left_model: str,
    right_model: str,
    *,
    split: str = "test",
) -> dict[str, dict[str, float]]:
    indexed = comparison.loc[comparison["split"].eq(split)].set_index("model")
    if left_model not in indexed.index or right_model not in indexed.index:
        raise KeyError("Both comparison models must have one row in the requested split")
    result = {}
    for metric in REPORT_METRICS:
        left = float(indexed.loc[left_model, metric])
        right = float(indexed.loc[right_model, metric])
        raw = left - right
        advantage = raw if metric in HIGHER_IS_BETTER else -raw
        result[metric] = {
            "left": left,
            "right": right,
            "raw_left_minus_right": raw,
            "performance_advantage_left": advantage,
        }
    return result


def feature_control_gains(control_comparison: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for split in SPLIT_ORDER:
        indexed = control_comparison.loc[
            control_comparison["split"].eq(split)
        ].set_index("profile")
        canonical = indexed.loc["canonical_event"]
        control = indexed.loc["pre_round_control"]
        split_result = {}
        for metric in REPORT_METRICS:
            raw = float(canonical[metric] - control[metric])
            split_result[metric] = raw if metric in HIGHER_IS_BETTER else -raw
        result[split] = split_result
    return result


def audit_frozen_xgboost(model: Any) -> dict[str, Any]:
    params = model.get_params()
    expected = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
    }
    mismatches = {
        name: {"expected": value, "actual": params.get(name)}
        for name, value in expected.items()
        if params.get(name) != value
    }
    if params.get("early_stopping_rounds") is not None:
        mismatches["early_stopping_rounds"] = {
            "expected": None,
            "actual": params.get("early_stopping_rounds"),
        }
    return {"passed": not mismatches, "expected": expected, "mismatches": mismatches}


def audit_predictions(predictions: pd.DataFrame, expected_rows: int) -> dict[str, Any]:
    probability_columns = [
        f"{model_name}_probability" for model_name in FORMAL_MODEL_NAMES
    ]
    probability = predictions[probability_columns]
    invalid_cells = int(
        (~np.isfinite(probability.to_numpy(dtype=float))).sum()
        + ((probability < 0) | (probability > 1)).sum().sum()
    )
    duplicate_keys = int(predictions.duplicated(ID_COLUMNS).sum())
    passed = (
        len(predictions) == expected_rows
        and invalid_cells == 0
        and duplicate_keys == 0
    )
    return {
        "passed": passed,
        "rows": int(len(predictions)),
        "expected_rows": int(expected_rows),
        "invalid_probability_cells": invalid_cells,
        "duplicate_key_rows": duplicate_keys,
    }


def render_external_report(comparison: pd.DataFrame) -> str:
    lines = [
        "# M16 首杀后外部模型指标对照",
        "",
        "差值统一为“我们的指标 - 外部指标”。Accuracy/AUC 同时换算为百分点。",
        "`current_model` 指明应使用本阶段逻辑回归还是 XGBoost；不同数据、切分和预测",
        "时点仍使这些结果无法成为受控模型排行榜。",
        "",
        "| 可比性 | 本地模型 | 外部工作 | 指标 | 我们 | 外部 | 差值 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        title = row.get("source_title", row["benchmark_id"])
        url = row.get("source_url", "")
        source = f"[{title}]({url})" if url else title
        difference = row["raw_difference_ours_minus_reported"]
        if row["metric"] in {"accuracy", "auc"}:
            difference_text = f"{difference * 100:+.2f} 个百分点"
        else:
            difference_text = f"{difference:+.6f}"
        lines.append(
            f"| {row.get('comparability', '')} | `{row['current_model']}` | {source} | "
            f"{row['metric']} | {row['current_value']:.6f} | "
            f"{row['reported_value']:.6f} | {difference_text} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `closest_task` 只表示预测时点最接近。424 回合个人数据和随机行切分的方差、",
            "  难度都不同于本项目的 782 个系列赛分组切分。",
            "- 实时 WPA 工作混合整回合时点，并使用 HP、人数、炸弹和空间信息；只报告",
            "  数值差，不判断模型优劣。",
            "- freeze-time DNN 的预测时点早于首杀，因此任务更难；差值不能解释为 XGBoost",
            "  优于 DNN。",
            "",
        ]
    )
    return "\n".join(lines)


def render_m16_report(
    comparison: pd.DataFrame,
    control_comparison: pd.DataFrame,
    external: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    test = comparison.loc[comparison["split"].eq("test")].set_index("model")
    target = summary["metric_targets"]
    difference = summary["xgboost_vs_logistic_test"]
    gains = summary["feature_control_gains"]
    lines = [
        "# M16 首杀后固定切分基线报告",
        "",
        "## 阶段决定",
        "",
        f"验收状态：**{summary['acceptance']['status']}**。",
        f"可以进入 M17：**{summary['acceptance']['ready_for_m17']}**。",
        "本阶段没有调参；三个正式模型使用完全相同的样本、特征、编码列和指标代码。",
        "",
        "## 数据与特征",
        "",
        f"- M15 样本：{summary['data']['rows']:,}；编码后特征：{summary['features']['encoded_count']}。",
        f"- train/val/test：{summary['data']['split_rows']['train']:,} / "
        f"{summary['data']['split_rows']['val']:,} / {summary['data']['split_rows']['test']:,}。",
        "- 正式首杀特征：CT 优势、首杀时间、爆头、武器；五个确定性重复字段全部排除。",
        "- ID、split 和 label 不进入模型。",
        "",
        "## 三模型测试结果",
        "",
        "| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in FORMAL_MODEL_NAMES:
        row = test.loc[model_name]
        lines.append(
            f"| `{model_name}` | {row['accuracy']:.6f} | {row['auc']:.6f} | "
            f"{row['log_loss']:.6f} | {row['brier_score']:.6f} | {row['ece10']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 预先目标验收",
            "",
            "| 指标 | 当前 XGBoost | 最低门槛 | 最低通过 | 阶段目标 | 目标通过 |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for metric, item in target["metrics"].items():
        lines.append(
            f"| {metric} | {item['value']:.6f} | {item['minimum']:.3f} | "
            f"{item['minimum_passed']} | {item['stage']:.3f} | {item['stage_passed']} |"
        )

    lines.extend(
        [
            "",
            "## XGBoost 与逻辑回归",
            "",
            "差值为 XGBoost 减逻辑回归；Log Loss/Brier/ECE 的负值代表 XGBoost 更低。",
            "",
            "| 指标 | 原始差值 | XGBoost 是否更好 |",
            "|---|---:|---|",
        ]
    )
    for metric, item in difference.items():
        lines.append(
            f"| {metric} | {item['raw_left_minus_right']:+.6f} | "
            f"{item['performance_advantage_left'] > 0} |"
        )

    lines.extend(
        [
            "",
            "逻辑回归和 XGBoost 的 AUC 几乎相同。树模型不能仅凭模型复杂度宣称胜出；",
            "M17 调参应优先改善概率损失与泛化，而不是追逐极小的 AUC 波动。",
            "",
            "## 首杀信息控制组",
            "",
            "两个 XGBoost 使用相同参数和相同回合，唯一变量是是否加入四个首杀特征。",
            "",
            "| profile | split | Accuracy | AUC | Log Loss | Brier |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in control_comparison.loc[
        control_comparison["split"].isin(["val", "test"])
    ].to_dict(orient="records"):
        lines.append(
            f"| `{row['profile']}` | {row['split']} | {row['accuracy']:.6f} | "
            f"{row['auc']:.6f} | {row['log_loss']:.6f} | {row['brier_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Validation AUC 增益：**{gains['val']['auc']:+.6f}**；"
            f"测试 AUC 增益：**{gains['test']['auc']:+.6f}**。",
            "这项控制说明性能提高来自首杀事件信息，而不是样本集合或切分变化。",
            "",
            "## 与外部模型相差多少",
            "",
            "| 本地模型 | 外部工作 | 指标 | 我们 | 外部 | 差值 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in external.to_dict(orient="records"):
        if row["comparability"] not in {"closest_task", "not_comparable"}:
            continue
        difference_value = row["raw_difference_ours_minus_reported"]
        difference_text = (
            f"{difference_value * 100:+.2f} 个百分点"
            if row["metric"] in {"accuracy", "auc"}
            else f"{difference_value:+.6f}"
        )
        lines.append(
            f"| `{row['current_model']}` | {row['source_title']} | {row['metric']} | "
            f"{row['current_value']:.6f} | {row['reported_value']:.6f} | "
            f"{difference_text} |"
        )

    lines.extend(
        [
            "",
            "最近的同任务公开项目样本仅 424 回合且按行随机切分；实时 WPA 工作使用",
            "更丰富的整回合状态。上表只回答数值差，不能把差值归因于模型本身。",
            "完整来源和 freeze-time 参考见 `external_benchmark_comparison.md`。",
            "",
            "## 历史结果关系",
            "",
            f"旧主键/旧事件选择的首杀 XGBoost 测试 AUC 为 0.774750；当前为 "
            f"{test.loc['xgboost_untuned', 'auc']:.6f}，原始差值 "
            f"{test.loc['xgboost_untuned', 'auc'] - 0.7747496424780327:+.6f}。",
            "旧值无效，因此这不是受控提升，只用于说明为什么必须先完成 M15。",
            "",
            "## 下一阶段",
            "",
            "M17 只看 train/validation 做控制变量调参。由于逻辑回归 AUC 已与 XGBoost",
            "相当，M17 必须同时观察 Log Loss、Brier、过拟合差距和首杀特征消融，不能只",
            "以测试 AUC 反复选择参数。",
            "",
            "复现命令：",
            "",
            "```powershell",
            ".\\scripts\\run_first_kill_baselines.ps1",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    def default(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=default)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _split_expectation(m15_summary: dict[str, Any]) -> dict[str, int]:
    return {
        str(row["split"]): int(row["samples"])
        for row in m15_summary.get("split_summary", [])
    }


def _test_metric_rows(comparison: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for row in comparison.loc[comparison["split"].eq("test")].to_dict(
        orient="records"
    ):
        result[row["model"]] = {
            metric: float(row[metric]) for metric in REPORT_METRICS
        }
    return result


def _save_model_bundle(
    model: Any,
    path: Path,
    *,
    model_name: str,
    profile: str,
    raw_features: list[str],
    encoded_columns: list[str],
    data_sha256: str,
) -> dict[str, Any]:
    bundle = {
        "model": model,
        "task": "first_kill",
        "model_name": model_name,
        "profile": profile,
        "raw_features": raw_features,
        "columns": encoded_columns,
        "data_sha256": data_sha256,
    }
    joblib.dump(bundle, path)
    return fingerprint_file(path)


def run(
    data_path: str | Path,
    m15_summary_path: str | Path,
    benchmarks_path: str | Path,
    model_dir: str | Path,
    report_dir: str | Path,
    project_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_path = Path(data_path)
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    project_root = Path(project_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    m15_summary = _read_json(m15_summary_path)
    artifact_audit = verify_m15_artifact(data_path, m15_summary)
    if not artifact_audit["passed"]:
        raise RuntimeError("M16 input does not match the accepted M15 data fingerprint")

    data = read_table(data_path)
    data_audit = audit_training_data(data)
    expected_rows = int(m15_summary.get("counts", {}).get("sample_rows", -1))
    expected_split_rows = _split_expectation(m15_summary)
    actual_split_rows = data_audit.get("split_rows", {})
    counts_match_m15 = (
        len(data) == expected_rows
        and expected_split_rows
        and actual_split_rows == expected_split_rows
    )
    if not data_audit["passed"] or not counts_match_m15:
        raise RuntimeError("M16 data identity or split counts differ from M15")

    feature_contract = build_feature_contract()
    canonical_features = canonical_feature_names()
    missing_features = sorted(set(canonical_features) - set(data.columns))
    if missing_features:
        raise KeyError(f"M16 canonical features are missing: {missing_features}")

    canonical_prepared = prepare_profile_splits(data, canonical_features)
    control_features = list(PRE_ROUND_FEATURES)
    control_prepared = prepare_profile_splits(data, control_features)
    encoded_columns = canonical_prepared["train"][0].columns.tolist()
    forbidden_encoded = sorted(
        column
        for column in encoded_columns
        if column in set(ID_COLUMNS + [LABEL_COL, "split"])
        or column in REDUNDANT_FIRST_KILL_FEATURES
    )
    feature_contract_passed = not missing_features and not forbidden_encoded

    models = fit_formal_models(canonical_prepared)
    comparison, probabilities = evaluate_models(models, canonical_prepared)
    control_model = fit_pre_round_control(control_prepared)
    control_comparison = build_feature_control_comparison(
        comparison, control_model, control_prepared
    )

    test_rows = data.loc[data["split"].eq("test")]
    predictions = build_prediction_table(test_rows, probabilities["test"])
    prediction_audit = audit_predictions(predictions, expected_split_rows["test"])
    xgboost_audit = audit_frozen_xgboost(models["xgboost_untuned"])

    test_metrics = _test_metric_rows(comparison)
    xgboost_test = test_metrics["xgboost_untuned"]
    metric_targets = assess_metric_targets(xgboost_test)
    xgboost_vs_logistic = model_metric_differences(
        comparison, "xgboost_untuned", "logistic_regression"
    )
    control_gains = feature_control_gains(control_comparison)

    external_benchmarks = pd.read_csv(benchmarks_path)
    external = compare_external_models(comparison, external_benchmarks)
    external_report = render_external_report(external)
    external_report_passed = not external.empty and bool(external_report.strip())

    automated_tests = run_automated_tests(project_root)
    test_count_match = re.search(r"Ran (\d+) tests?", automated_tests["output"])
    automated_test_count = int(test_count_match.group(1)) if test_count_match else None

    checks = {
        "m15_artifact": artifact_audit["passed"],
        "data_contract": data_audit["passed"] and counts_match_m15,
        "feature_contract": feature_contract_passed,
        "model_probabilities": prediction_audit["passed"],
        "frozen_xgboost": xgboost_audit["passed"],
        "minimum_metrics": metric_targets["all_minimum_passed"],
        "automated_tests": automated_tests["passed"],
        "external_report": external_report_passed,
    }
    acceptance = decide_acceptance(checks)

    model_artifacts = {}
    for model_name, model in models.items():
        model_artifacts[model_name] = _save_model_bundle(
            model,
            model_dir / f"first_kill_{model_name}.joblib",
            model_name=model_name,
            profile="canonical_event",
            raw_features=canonical_features,
            encoded_columns=encoded_columns,
            data_sha256=artifact_audit["actual_sha256"],
        )
    control_columns = control_prepared["train"][0].columns.tolist()
    model_artifacts["xgboost_pre_round_control"] = _save_model_bundle(
        control_model,
        model_dir / "first_kill_xgboost_pre_round_control.joblib",
        model_name="xgboost_untuned",
        profile="pre_round_control",
        raw_features=control_features,
        encoded_columns=control_columns,
        data_sha256=artifact_audit["actual_sha256"],
    )

    comparison.to_csv(report_dir / "m16_model_comparison.csv", index=False)
    control_comparison.to_csv(report_dir / "m16_feature_control.csv", index=False)
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    feature_contract.to_csv(report_dir / "feature_contract.csv", index=False)
    pd.DataFrame({"encoded_feature": encoded_columns}).to_csv(
        report_dir / "encoded_feature_columns.csv", index=False
    )
    external.to_csv(report_dir / "external_benchmark_comparison.csv", index=False)
    (report_dir / "external_benchmark_comparison.md").write_text(
        external_report, encoding="utf-8"
    )
    (report_dir / "automated_test_output.txt").write_text(
        automated_tests["output"], encoding="utf-8"
    )

    canonical_history = models["xgboost_untuned"].evals_result()["validation_0"][
        "logloss"
    ]
    control_history = control_model.evals_result()["validation_0"]["logloss"]
    pd.DataFrame(
        {
            "iteration": range(len(canonical_history)),
            "canonical_event_val_logloss": canonical_history,
            "pre_round_control_val_logloss": control_history,
        }
    ).to_csv(report_dir / "xgboost_training_history.csv", index=False)

    checks_rows = [
        {"check": name, "passed": passed, "blocking": True}
        for name, passed in checks.items()
    ]
    pd.DataFrame(checks_rows).to_csv(report_dir / "m16_checks.csv", index=False)

    xgboost_rows = comparison.loc[comparison["model"].eq("xgboost_untuned")].set_index(
        "split"
    )
    summary = {
        "stage": "M16",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "post_first_kill",
        "definition": "immediately after earliest valid enemy kill by demo tick",
        "acceptance": acceptance,
        "checks": checks,
        "data": {
            "path": data_path.as_posix(),
            "sha256": artifact_audit["actual_sha256"],
            "rows": int(len(data)),
            "split_rows": actual_split_rows,
            "series": int(data["series_id"].nunique()),
            "games": int(data["game_id"].nunique()),
            "duplicate_key_rows": data_audit["duplicate_key_rows"],
            "cross_split_series": data_audit["cross_split_series"],
        },
        "features": {
            "profile": "canonical_event",
            "raw_count": len(canonical_features),
            "encoded_count": len(encoded_columns),
            "raw_features": canonical_features,
            "excluded_redundant_features": REDUNDANT_FIRST_KILL_FEATURES,
            "forbidden_encoded_features": forbidden_encoded,
            "control_encoded_count": len(control_columns),
        },
        "models": {
            "formal": list(FORMAL_MODEL_NAMES),
            "xgboost_params": models["xgboost_untuned"].get_params(),
            "model_artifacts": model_artifacts,
        },
        "metrics": xgboost_test,
        "test_metrics_by_model": test_metrics,
        "metric_targets": metric_targets,
        "xgboost_vs_logistic_test": xgboost_vs_logistic,
        "feature_control_gains": control_gains,
        "xgboost_train_minus_val_auc": float(
            xgboost_rows.loc["train", "auc"] - xgboost_rows.loc["val", "auc"]
        ),
        "prediction_audit": prediction_audit,
        "xgboost_parameter_audit": xgboost_audit,
        "external_comparison_rows": int(len(external)),
        "automated_tests": {
            "passed": automated_tests["passed"],
            "return_code": automated_tests["return_code"],
            "elapsed_seconds": automated_tests["elapsed_seconds"],
            "test_count": automated_test_count,
        },
        "next_stage": "M17 validation-only controlled XGBoost tuning",
    }
    write_json(summary, report_dir / "m16_summary.json")
    (report_dir / "m16_first_kill_baseline_report.md").write_text(
        render_m16_report(comparison, control_comparison, external, summary),
        encoding="utf-8",
    )
    return comparison, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and audit M16 post-first-kill fixed-split baselines."
    )
    parser.add_argument(
        "--data", default="data/processed/esta_full/first_kill.parquet"
    )
    parser.add_argument(
        "--m15-summary", default="reports/esta_full_m15/m15_summary.json"
    )
    parser.add_argument(
        "--benchmarks", default="benchmarks/external_first_kill_metrics.csv"
    )
    parser.add_argument("--model-dir", default="models/esta_full_m16")
    parser.add_argument("--report-dir", default="reports/esta_full_m16")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    comparison, summary = run(
        data_path=args.data,
        m15_summary_path=args.m15_summary,
        benchmarks_path=args.benchmarks,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        project_root=args.project_root,
    )
    print(comparison.loc[comparison["split"].eq("test")].round(6).to_string(index=False))
    print(
        f"M16 {summary['acceptance']['status']}; "
        f"ready_for_m17={summary['acceptance']['ready_for_m17']}"
    )
    if not summary["acceptance"]["ready_for_m17"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

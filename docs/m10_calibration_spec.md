# M10 概率校准验收

## 目的

检查 XGBoost 输出的 CT 胜率是否需要二次校准，并保证校准方法只由验证集决定。
测试集只用于对预先选定的方法做一次最终评价。

## 方法

比较三个预先定义的方法：

1. `uncalibrated`：Identity，不改变 XGBoost 原始概率。
2. `sigmoid`：对原始概率的 logit 拟合逻辑回归，也称 Platt scaling。
3. `isotonic`：单调非参数校准。

验证集包含 8,380 个回合和 156 个系列赛。方法选择使用 5 折 GroupKFold，
同一 `series_id` 的所有回合始终处于同一折。每种方法生成完整的验证集 OOF
概率，按 Log Loss 最低、Brier 次低排序。测试指标不参与方法选择。

## 验证集 OOF 结果

| 方法 | Log Loss | Brier | ECE10 | AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| 不校准 | **0.596038** | **0.207575** | 0.011285 | **0.718596** | **0.650358** |
| Sigmoid | 0.596250 | 0.207634 | **0.008777** | 0.718221 | 0.648926 |
| Isotonic | 0.605117 | 0.208231 | 0.012645 | 0.713903 | 0.644511 |

Sigmoid 的 ECE 略低，但 Log Loss 和 Brier 都略差；Isotonic 的 Log Loss 明显
变差。根据预先规定的验证集 OOF Log Loss，选择 `uncalibrated`。

## 固定测试集结果

| 方法 | Log Loss | Brier | ECE10 | AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| 不校准（验证集选择） | **0.591733** | **0.205294** | 0.023198 | **0.727122** | 0.647411 |
| Sigmoid | 0.591890 | 0.205347 | 0.022049 | **0.727122** | 0.648130 |
| Isotonic | 0.594422 | 0.205676 | **0.010360** | 0.726040 | **0.652924** |

Isotonic 虽然降低测试 ECE 并提高 0.5 阈值 Accuracy，但同时损害 Log Loss、
Brier 和 AUC，因此不能为了单个指标采用它。Sigmoid 同样没有改善主要概率指标。

## 验收结论

- 校准器只使用验证集拟合：通过。
- 方法选择使用按系列赛分组的验证集 OOF 概率：通过。
- 选定方法测试 ECE <= 0.04：通过，实际 0.023198。
- 选定方法测试 ECE <= 0.03：通过。
- 没有为了降低 ECE 明显损害 Log Loss 或 Brier：通过。
- 校准模型可以在独立 Python 进程加载：通过。

M10 的结论是当前 XGBoost 原始概率已经足够校准，不增加二次概率变换。项目
仍保存 Identity 校准器，明确记录验证集决策并保持后续预测接口一致。

## 输出文件

```text
models\esta_full_m10\pre_round_calibrator.joblib
reports\esta_full_m10\validation_oof_comparison.csv
reports\esta_full_m10\validation_oof_predictions.csv
reports\esta_full_m10\test_calibration_comparison.csv
reports\esta_full_m10\calibrated_test_predictions.csv
reports\esta_full_m10\calibration_curves.csv
reports\esta_full_m10\reliability_comparison.png
reports\esta_full_m10\m10_summary.json
reports\esta_full_m10\m10_calibration_report.md
```

## 复现命令

```powershell
cd C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m10_calibration --data data\processed\esta_full\pre_round.parquet --base-model models\esta_full_m8_tuned\pre_round_xgb.joblib --model-dir models\esta_full_m10 --report-dir reports\esta_full_m10 --folds 5
```

下一阶段是 M11：为地图和 LAN/online 分组补系列赛级置信区间，并分析至少
30 个高置信度错误回合。

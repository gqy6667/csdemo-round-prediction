# M18 首杀后固定模型评估规格

## 1. 目标

M18 对 M17 已冻结的首杀后 XGBoost 做一次独立、可复现的验收。此阶段不重新训练、
不调 XGBoost 参数，也不根据测试集更改特征。需要回答四个问题：

1. 整体指标的不确定范围有多大；
2. 模型在不同地图、LAN/online 和首杀情境下是否稳定；
3. 高置信度错误集中在哪些已定义情境；
4. 概率是否需要校准，以及校准能否在不伤害概率指标的前提下使用。

## 2. 冻结输入和数据边界

- 数据：`data/processed/esta_full/first_kill.parquet`。
- 模型：`models/esta_full_m17/first_kill_xgboost_tuned.joblib`。
- M17 测试预测：`reports/esta_full_m17/test_predictions.csv`。
- M17 验收摘要：`reports/esta_full_m17/m17_summary.json`。
- 主键：`series_id + game_id + round_id`，禁止只用 `round_id` 连接。
- 标签：`ct_win=1` 表示 CT 最终赢得该回合。
- 验证集只用于校准方法选择；测试集只用于固定模型、固定校准方法的最终评估。

运行前必须校验数据和模型 SHA-256、split 行数、测试主键唯一性，并由模型重新生成
测试概率。重新生成的概率与 M17 保存概率最大绝对误差必须不超过 `1e-12`。

## 3. 整体统计不确定性

对测试集按 `series_id` 有放回抽样 2,000 次，计算 Accuracy、AUC、Log Loss、Brier
和 ECE10 的 95% percentile bootstrap 区间。回合不能作为独立 bootstrap 单位，因为
同一系列赛中的回合存在相关性。

预先固定目标：

| 项目 | 最低验收 | 阶段目标 |
|---|---:|---:|
| AUC 95% CI 下界 | >= 0.780 | >= 0.790 |
| Log Loss 95% CI 上界 | <= 0.550 | <= 0.540 |
| 每项成功 bootstrap 次数 | 2,000 | 2,000 |

## 4. 固定分组

所有分组都报告回合数、系列数、CT 胜率、五项指标及系列级 95% CI：

- 地图：`map_name`；
- 来源：从 `game_id` 严格解析为 `lan` 或 `online`；
- 回合阶段：1-10、11-20、21+；
- 装备差：T 大优势、T 中优势、均衡、CT 中优势、CT 大优势；
- 首杀方：CT 或 T；
- 首杀时间：`[0,15)`、`[15,30)`、`[30,60)`、`[60,+inf)` 秒；
- 首杀武器族：步枪、狙击枪、手枪、冲锋枪/霰弹枪、投掷物/其他；
- 是否爆头：否或是。

地图门槛只约束测试回合数至少 300 的主要地图：每张地图点估计 AUC 必须 >= 0.740，
阶段目标为 >= 0.770；95% CI 下界最低目标为 >= 0.700。LAN 与 online 的点估计
AUC 绝对差必须 <= 0.040。来源差值 CI 是否包含 0 同时报告，但不因统计功效不足单独
改变模型。

## 5. 校准协议

比较三种固定方法：不校准、Sigmoid 和 Isotonic。

1. 在 validation 中按 `series_id` 做 5 折 GroupKFold；
2. 每种方法生成完整 OOF 概率；
3. 只按 OOF Log Loss 最低、Brier 次低、方法名最后排序选方法；
4. 用完整 validation 拟合所选校准器；
5. 只在选择冻结后评估 test。

测试集不参与方法选择。所选方法测试 ECE10 阶段目标为 <= 0.030；相对不校准模型，
Log Loss 变差不得超过 `0.002`，Brier 变差不得超过 `0.001`。若不校准胜出，则持久化
Identity 校准器，这也是有效结论。

## 6. 错误分析

高置信度错误定义为预测错误且预测方概率 >= 0.80。保存全部错误，并按置信度选择前
30 个案例人工复核。固定的信号组合只使用预测时已经存在的信息：

- `first_kill_and_equipment_agree`：预测方拿到首杀且装备优势至少 1,500；
- `first_kill_only`：只有首杀支持预测方；
- `equipment_only`：只有装备优势支持预测方；
- `neither`：两者均不支持预测方。

报告预测方、首杀方、首杀时间、武器族、爆头与信号组合的错误数量。这些是错误模式，
不是因果解释，也不作为重新选择模型的依据。

## 7. 外部模型比较

继续使用 `benchmarks/external_first_kill_tuned_metrics.csv`。M18 没有改变模型，点指标
应与 M17 完全一致；本阶段新增价值是置信区间、分组稳健性和校准诊断。所有外部差值
均标注数据集、切分和预测时点不同，不能解释为纯算法差异。

## 8. 阻断验收项

1. M17 数据和模型哈希、任务、特征契约校验通过；
2. 70/20/10 split 与 M17 一致，完整主键唯一且集合完全匹配；
3. 重新生成的测试概率与 M17 保存值一致；
4. 五项整体指标均完成 2,000 次系列级 bootstrap；
5. 固定八类分组均成功生成；
6. LAN/online AUC 绝对差不超过 0.040；
7. 所有主要地图点估计 AUC 不低于 0.740；
8. 校准选择严格来自 validation OOF，且测试概率指标无明显伤害；
9. 全部高置信度错误和前 30 个复核表已生成；
10. 外部比较、中文报告、自动化测试和源码编译全部通过。

未达到阶段目标但仍超过最低验收线时，记录为“接受现阶段、保留改进项”；阻断项失败
则 M18 失败，不进入 M19。

## 9. 交付物

```text
src/csdemo/m18_first_kill_evaluation.py
tests/test_m18_first_kill_evaluation.py
scripts/run_first_kill_evaluation.ps1
models/esta_full_m18/first_kill_calibrator.joblib
reports/esta_full_m18/global_bootstrap_95ci.csv
reports/esta_full_m18/metrics_by_*.csv
reports/esta_full_m18/source_auc_gap.csv
reports/esta_full_m18/validation_oof_calibration.csv
reports/esta_full_m18/test_calibration_comparison.csv
reports/esta_full_m18/all_high_confidence_errors.csv
reports/esta_full_m18/reviewed_top30_errors.csv
reports/esta_full_m18/m18_checks.csv
reports/esta_full_m18/m18_summary.json
reports/esta_full_m18/m18_first_kill_evaluation_report.md
```

## 10. 下一阶段

M18 通过后，M19 对首杀后模型执行特征重要性、Permutation Importance、SHAP 与泄漏
审计，解释哪些购买状态和首杀事件真正推动了概率变化。

## 11. 实际验收结果

M18 于 2026-08-19 完整运行并通过：

- 28,489/8,368/4,170 行对应 547/156/79 个系列赛，跨 split 系列数和重复主键均为 0；
- M17 测试概率回放最大绝对误差为 `1.11e-16`，XGBoost 训练调用为 0；
- 测试 AUC `0.809837`，系列赛级 95% CI `[0.797731, 0.822081]`；
- 测试 Log Loss `0.523146`，95% CI `[0.509747, 0.536146]`；
- LAN-online AUC 差 `-0.010276`，95% CI `[-0.034586, 0.014829]`；
- 七张至少 300 回合的地图最低 AUC 为 Ancient 的 `0.783901`，最低 CI 下界
  `0.750719`；
- validation OOF 选择 `uncalibrated`，测试 ECE10 为 `0.015450`；
- 90 个高置信度错误全部保存并复核前 30 个，其中 81 个同时得到首杀与装备信号支持；
- 13 个阻断检查、全部阶段目标、108 项自动化测试和源码编译全部通过。

详细结果见 `reports/esta_full_m18/m18_first_kill_evaluation_report.md`。

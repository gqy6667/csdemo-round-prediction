# M30 首杀后 LightGBM 冻结模型评估规格

## 1. 目标

M30 对 M29 冻结的首杀后 LightGBM 做独立、可复现的正式评估。本阶段不训练
LightGBM、不调参、不改特征，也不得根据测试结果返回 M29 改网格。需要回答：

1. 五项整体指标的系列赛级 95% 置信区间；
2. LightGBM 相对同一测试行 M21 XGBoost 的差异是否有稳定配对证据；
3. 地图、LAN/online、回合阶段、装备及首杀情境下是否稳定；
4. validation-only 校准是否值得采用；
5. 高置信错误集中在哪些首杀与装备组合。

## 2. 冻结输入

| 输入 | 路径或合同 |
|---|---|
| 数据 | `data/processed/esta_full/first_kill.parquet` |
| 数据 SHA-256 | `06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492` |
| 模型 | `models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib` |
| M29 摘要 | `reports/esta_full_m29/m29_summary.json` |
| M29 测试概率 | `reports/esta_full_m29/test_predictions.csv` |
| 主键 | `series_id + game_id + round_id` |
| 样本与切分 | 41,027；28,489/8,368/4,170 行；547/156/79 系列赛 |
| 特征 | 40 个原始、82 个编码特征 |
| 冻结参数变化 | `max_depth=3`，正式 seed 42，211 棵树 |

必须核验 M29 已通过、数据与模型哈希、任务、特征顺序、完整测试键和 split。
重新回放的 4,170 个 LightGBM 测试概率与 M29 保存值最大绝对误差不超过 `1e-12`，
LightGBM `fit` 调用次数必须为 0。

## 3. 整体与配对 Bootstrap

按 `series_id` 有放回抽样 2,000 次，计算 Accuracy、AUC、Log Loss、Brier、ECE10
的 95% percentile 区间。预先固定首杀后最低线和阶段目标：

| 项目 | 最低验收 | 阶段目标 |
|---|---:|---:|
| AUC 95% CI 下界 | >= 0.780 | >= 0.790 |
| Log Loss 95% CI 上界 | <= 0.550 | <= 0.540 |
| 每项成功次数 | 2,000 | 2,000 |

同一批系列赛抽样同时计算 M29 LightGBM 与 M21 XGBoost。保存五项原始差值、统一为
越大越好的性能优势、95% CI 及是否排除 0。配对结果完整是阻断项，但 LightGBM
显著胜出不是阻断项；点指标或单项 ECE10 不得替代配对结论。

## 4. 八类固定分组

所有分组报告回合数、系列赛数、CT 胜率、五项指标和系列赛级 95% CI：

- 地图；
- LAN/online；
- 回合阶段 1-10、11-20、21+；
- 装备差五档；
- 首杀方 CT/T；
- 首杀时间 `[0,15)`、`[15,30)`、`[30,60)`、`[60,+inf)`；
- 首杀武器族；
- 是否爆头。

LAN 与 online AUC 绝对差必须 <= 0.040。至少 300 回合的主要地图点估计 AUC
必须 >= 0.740，阶段目标 >= 0.770；主要地图最低 AUC CI 下界 >= 0.700 是阶段目标。

## 5. 校准协议

固定比较不校准、Sigmoid、Isotonic：

1. validation 按 `series_id` 做 5 折 GroupKFold；
2. 生成三种方法的完整 OOF 概率；
3. 只按 OOF Log Loss、Brier、方法名排序选择；
4. 用完整 validation 拟合所选校准器；
5. 方法冻结后才在 test 上比较。

所选方法相对原始概率的测试 Log Loss 变差不得超过 0.002，Brier 变差不得超过
0.001；ECE10 <= 0.030 是阶段目标。选择 Identity 校准器也是有效结论。

## 6. 错误分析

高置信错误定义为预测错误且预测方概率 >= 0.80。保存全部错误和置信度最高的 30 个。
固定信号组合为：首杀与装备都支持预测方、仅首杀支持、仅装备支持、两者均不支持。
这些是描述性错误模式，不是因果解释，也不用于重选模型。

## 7. 阻断验收

1. `m29_prerequisite`：M29 数据、模型、任务、参数和特征合同通过；
2. `split_and_key_contract`：系列赛级 70/20/10 及完整主键一致；
3. `frozen_model_replay`：validation/test 回放且 LightGBM fit=0；
4. `prediction_replay`：M29 测试概率与五项指标精确复现；
5. `global_bootstrap`：五项均完成 2,000 次；
6. `global_metric_minimum`：AUC 与 Log Loss 区间最低线通过；
7. `paired_comparison`：LightGBM-XGBoost 五项配对区间完整；
8. `group_outputs`：八类固定分组均生成；
9. `source_stability`：LAN/online AUC 差通过；
10. `large_map_minimum`：主要地图最低 AUC 通过；
11. `calibration_protocol`：仅由 validation GroupKFold OOF 选择；
12. `calibration_no_material_harm`：测试概率指标无明显伤害；
13. `error_review`：全部高置信错误和前 30 个案例保存；
14. `external_report`：外部比较保留可比性限制；
15. `automated_tests`：全量测试通过；
16. `source_compile`：源码和测试编译通过；
17. `reproduction_entrypoint`：一键脚本通过合同检查；
18. `artifact_manifest`：模型、校准器及正式产物哈希齐全。

## 8. 产物

```text
src/csdemo/m30_post_first_kill_lightgbm_evaluation.py
tests/test_m30_post_first_kill_lightgbm_evaluation.py
scripts/run_post_first_kill_lightgbm_evaluation.ps1
models/esta_full_m30/post_first_kill_lightgbm_calibrator.joblib
reports/esta_full_m30/global_bootstrap_95ci.csv
reports/esta_full_m30/paired_lightgbm_vs_xgboost_bootstrap.csv
reports/esta_full_m30/metrics_by_*.csv
reports/esta_full_m30/validation_oof_calibration.csv
reports/esta_full_m30/test_calibration_comparison.csv
reports/esta_full_m30/all_high_confidence_errors.csv
reports/esta_full_m30/reviewed_top30_errors.csv
reports/esta_full_m30/m30_summary.json
reports/esta_full_m30/m30_experiment_manifest.json
reports/esta_full_m30/m30_post_first_kill_lightgbm_evaluation_report.md
```

## 9. 下一阶段

M30 通过后进入 M31：冻结 M29 模型和 M30 校准决定，执行 Gain、Permutation
Importance、TreeSHAP、泄漏审计以及与 M19 XGBoost 的解释差异分析。

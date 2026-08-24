# M23 开局前 LightGBM 控制变量调参规格

## 1. 阶段目标

M23 在 M22 已通过的开局前 LightGBM 基线上做 validation-only 控制变量调参。目标是
寻找 validation Log Loss 更低且种子稳定的配置，同时保持实验可归因，不保证最终 test
一定提升。

本阶段不改变数据、预测时点、标签、70/20/10 系列赛切分、36 个原始特征、43 个编码
列、概率阈值或指标代码。不加入战队、选手、位置、血量或首杀信息。

## 2. 固定输入

| 项目 | 固定值 |
|---|---|
| 数据 | `data/processed/esta_full/pre_round.parquet` |
| 定义 | 购买完毕、冻结时间结束、正式交火前 |
| 行数 | 41,074 |
| train/validation/test | 28,522 / 8,380 / 4,172 |
| 系列赛 | 547 / 156 / 79 |
| 原始/编码特征 | 36 / 43 |
| M22 validation Log Loss | 0.595621 |
| M22 validation AUC | 0.718575 |
| 正式随机种子 | 42 |
| 选择指标 | validation Log Loss |
| 最小接受改善 | 0.0001 |

所有候选只接收 train 和 validation。候选结果、阶段选择和种子稳定性表禁止出现任何
`test_*` 列。test 只能传给参数冻结后的最终评估函数。

## 3. 逐阶段候选

使用 greedy sequential search：每一阶段从上一阶段已接受参数继续，只改变表中一个
参数。当前值必须出现在候选中；若最佳候选相对当前值的 validation Log Loss 改善不足
`0.0001`，保留当前值。

| 顺序 | 阶段 | 候选值 |
|---:|---|---|
| 1 | `num_leaves` | 7, 15, 31, 63 |
| 2 | `max_depth` | -1, 3, 4, 5, 6 |
| 3 | `min_child_samples` | 10, 20, 40, 80, 160 |
| 4 | `reg_lambda` | 0, 1, 3, 10 |
| 5 | `reg_alpha` | 0, 0.1, 0.5, 1 |
| 6 | `subsample` | 0.7, 0.85, 1.0 |
| 7 | `colsample_bytree` | 0.7, 0.85, 1.0 |
| 8 | `min_split_gain` | 0, 0.01, 0.05, 0.1 |
| 9 | `learning_rate` | 0.01, 0.02, 0.03, 0.05 |

总候选数为 36。所有模型继续使用最多 3,000 棵树、validation `binary_logloss` 和 100
轮 early stopping。`subsample_freq=1`、CPU、确定性模式和其他 M22 参数保持不变。

## 4. 选择规则

每阶段执行：

1. 训练全部候选，只计算 train/validation 五项指标；
2. 按 validation Log Loss 从低到高排序，同值按预先候选顺序决定；
3. 计算当前参数减最佳候选的 Log Loss；
4. 改善大于等于 `0.0001` 时接受，否则保留当前参数；
5. 把已接受参数带入下一阶段。

AUC、Accuracy、Brier 和 ECE10 只作诊断，不参与候选选择。不能因为候选 test 指标更好
而改变选择。

## 5. 种子稳定性

参数冻结后，以种子 42、43、44、45、46 分别在 train/validation 重训。正式模型仍使用
种子 42。阻断门槛：

| 项目 | 门槛 |
|---|---:|
| validation Log Loss 五种子范围 | <= 0.002 |
| validation AUC 五种子范围 | <= 0.003 |

种子表同样禁止出现 test 指标。

## 6. 阶段目标

以下是研究目标，不作为“必须调出提升”的阻断条件：

| 项目 | M23 目标 |
|---|---:|
| validation Log Loss 相对 M22 改善 | >= 0.0005 |
| validation AUC | >= 0.716575（最多下降 0.002） |
| train-validation AUC 差 | <= 0.030 |

最终 test 仍按 M22 的五项最低门槛验收：Accuracy >= 0.64、AUC >= 0.70、Log Loss
<= 0.61、Brier <= 0.21、ECE10 <= 0.05。是否胜过 M22 或 XGBoost不是阻断条件，必须
如实报告全部差值。

## 7. 外部比较

继续使用 `benchmarks/external_round_model_metrics.csv`。差值统一为“M23 LightGBM -
外部报告”，并保留可比性标签。同预测时点公开 DNN 的数据和随机行切分不同，不能据此
判断纯模型优劣。

## 8. 阻断检查

1. `m22_prerequisite`：M22 13/13 通过，数据与特征合同未漂移；
2. `data_contract`：行数、主键、split 和数据 SHA-256 与 M22 一致；
3. `feature_contract`：36/43 特征与 M22 完全一致，无 ID/label/split；
4. `candidate_grid`：9 阶段、36 候选、每阶段一个参数且包含当前值；
5. `validation_only`：候选表、选择表和种子表均无 test 列；
6. `phase_selection`：九阶段选择全部符合 `0.0001` 规则；
7. `frozen_model`：保存参数、正式种子和选择表一致；
8. `seed_stability`：五种子范围通过固定门槛；
9. `final_predictions`：4,172 条 test 键与 M22 完全一致，概率有效；
10. `minimum_metrics`：最终 test 五项最低门槛通过；
11. `controlled_comparison`：M22、XGBoost 和 M23 使用同一测试键与指标；
12. `external_report`：生成公开指标差值报告；
13. `automated_tests`：完整测试通过；
14. `reproduction_entrypoint`：一键脚本存在。

## 9. 阶段产物

```text
src/csdemo/m23_pre_round_lightgbm_tuning.py
tests/test_m23_pre_round_lightgbm_tuning.py
scripts/run_pre_round_lightgbm_tuning.ps1
models/esta_full_m23/pre_round_lightgbm_tuned.joblib
reports/esta_full_m23/tuning_candidates.csv
reports/esta_full_m23/phase_selections.csv
reports/esta_full_m23/seed_stability.csv
reports/esta_full_m23/test_predictions.csv
reports/esta_full_m23/m23_model_comparison.csv
reports/esta_full_m23/m23_summary.json
reports/esta_full_m23/m23_experiment_manifest.json
reports/esta_full_m23/m23_pre_round_lightgbm_tuning_report.md
```

模型二进制保留在本地并由 Git 忽略；参数、哈希、指标、概率和报告进入版本控制。

## 10. 下一阶段

M23 通过后进入 M24：不再调参，冻结 M23 模型并做系列赛 bootstrap 置信区间、地图与
LAN/online 稳健性、校准方法选择和错误分析。M24 不得根据 test 结果返回 M23 改网格。

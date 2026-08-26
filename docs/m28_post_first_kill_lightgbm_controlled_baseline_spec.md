# M28 首杀后 LightGBM 控制变量基线规格

## 1. 阶段目的

M28 在已经验收的 M21 首杀后 XGBoost 基线上，只替换模型算法为固定参数 LightGBM，建立第二组可公平比较的控制变量实验。

本阶段不是调参阶段，也不以测试集上战胜 XGBoost 为通过条件。目标是证明数据、预测时点、系列赛切分、特征和指标口径完全不变，并产出一次冻结的 LightGBM 基线测试结果及配对统计不确定性。

## 2. 冻结前提

| 项目 | 冻结值 |
|---|---|
| M21 状态 | `passed`，可进入 LightGBM 对照 |
| 预测时点 | 购买完成后，当前回合最早有效敌对击杀刚发生 |
| 数据 | `data/processed/esta_full/first_kill.parquet` |
| 数据 SHA-256 | `06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492` |
| 行数 | 41,027 |
| 系列赛 | 782 |
| XGBoost 模型 | `models/esta_full_m17/first_kill_xgboost_tuned.joblib` |
| XGBoost SHA-256 | `ecfaaf93031e78207f81ab5ad9674020657018c0601953238ee6b68e367e8279` |
| XGBoost 测试概率 | `reports/esta_full_m17/test_predictions.csv` |
| 主键 | `series_id + game_id + round_id` |
| 标签 | `ct_win` |

若数据或 XGBoost 哈希与 M21 不同，M28 必须停止，不能静默重建后继续比较。

## 3. 公平比较合同

- 复用 M21 的系列赛分配：train/validation/test 系列赛为 547/156/79。
- 行数固定为 train 28,489、validation 8,368、test 4,170。
- 使用 M16 `canonical_event` 的 40 个原始特征和训练集学习出的 82 个编码列。
- 36 个购买结束特征和 4 个首杀事件特征保持不变。
- 不加入身份、位置、后续击杀、伤害、血量、炸弹或任何预测时点后的字段。
- 冻结 XGBoost 只允许 `predict_proba` 回放，XGBoost `fit` 调用必须为 0。
- 测试概率必须按完整三列主键配对，禁止依赖行顺序。

## 4. 固定 LightGBM 基线

M28 使用项目 `train_lgbm.py` 已冻结的 CPU 参数，不搜索候选：

```text
boosting_type=gbdt
n_estimators=3000
learning_rate=0.03
num_leaves=15
min_child_samples=20
subsample=0.85
subsample_freq=1
colsample_bytree=0.85
reg_alpha=0.0
reg_lambda=1.0
objective=binary
random_state=42
device_type=cpu
deterministic=True
force_col_wise=True
```

只在 train 上拟合，以 validation `binary_logloss` 早停，`early_stopping_rounds=100`。测试集不参与拟合、早停、参数选择、种子选择或校准选择。

## 5. 指标与门槛

固定报告五项测试指标：Accuracy、AUC、Log Loss、Brier、ECE10。

| 指标 | 最低通过 | 阶段目标 | 方向 |
|---|---:|---:|---|
| Accuracy | 0.68 | 0.70 | 越高越好 |
| AUC | 0.75 | 0.78 | 越高越好 |
| Log Loss | 0.58 | 0.55 | 越低越好 |
| Brier | 0.20 | 0.185 | 越低越好 |
| ECE10 | 0.05 | 0.03 | 越低越好 |

最低门槛属于阻断条件；更高阶段目标只记录差距。LightGBM 相对 XGBoost 的点指标优劣不属于阻断条件。

## 6. 统计不确定性

- 对 LightGBM 五项指标按完整 `series_id` 做 2,000 次 bootstrap，种子 42。
- 对同一测试回合的 LightGBM 与 M21 XGBoost 概率按完整 `series_id` 做 2,000 次配对 bootstrap。
- 统一把正的 performance advantage 定义为 LightGBM 更好。
- 只有配对 95% CI 完全大于 0，才可称 LightGBM 在该指标上显著更好。
- 区间包含 0 时，只能报告点估计方向和统计不确定性，不能宣布算法胜负。

## 7. 阻断检查

M28 必须同时通过：

1. M21 前提与输入哈希；
2. 数据身份和固定 split；
3. 40/82 特征合同；
4. LightGBM 环境与固定参数；
5. train 拟合、validation 早停、test 隔离；
6. M21 XGBoost 完整主键概率回放；
7. LightGBM 概率和主键合同；
8. 五项最低指标门槛；
9. 同样本控制变量比较；
10. 五项全局系列赛 bootstrap；
11. 五项配对系列赛 bootstrap 与诚实结论；
12. 外部指标可比性说明；
13. 专项与全量测试；
14. 源码编译；
15. 一键复现入口；
16. 输入输出产物清单与 SHA-256。

## 8. 正式产物

```text
models/esta_full_m28/post_first_kill_lightgbm_baseline.joblib
reports/esta_full_m28/m28_summary.json
reports/esta_full_m28/m28_experiment_manifest.json
reports/esta_full_m28/m28_post_first_kill_lightgbm_controlled_baseline_report.md
reports/esta_full_m28/m28_model_comparison.csv
reports/esta_full_m28/m28_test_predictions.csv
reports/esta_full_m28/global_bootstrap_95ci.csv
reports/esta_full_m28/paired_lightgbm_vs_xgboost_bootstrap.csv
reports/esta_full_m28/feature_contract.csv
reports/esta_full_m28/encoded_feature_columns.csv
reports/esta_full_m28/lightgbm_training_history.csv
reports/esta_full_m28/m28_checks.csv
scripts/run_post_first_kill_lightgbm_baseline.ps1
```

## 9. 后续边界

M28 通过后再进入 validation-only LightGBM 受控调参、固定模型校准与稳健性、解释、接口和最终验收。第四份老师报告只能在这些证据完成并冻结后生成；M28 基线结果不得提前包装成最终 LightGBM 结论。

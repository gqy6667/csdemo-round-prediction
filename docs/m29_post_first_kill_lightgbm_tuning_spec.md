# M29 首杀后 LightGBM 控制变量调参规格

## 1. 阶段目标

M29 在 M28 已验收的首杀后 LightGBM 基线上执行 validation-only 控制变量调参。
目标是寻找 validation Log Loss 更低且跨随机种子稳定的配置，同时保持每一步可归因。
调参本身不保证测试集提升，也不把 LightGBM 必须胜过 XGBoost 设为验收条件。

本阶段只改变 LightGBM 超参数，不改变数据、预测时点、标签、系列赛级 70/20/10
切分、特征、指标实现、分类阈值或首杀事件定义。

## 2. 冻结输入合同

| 项目 | 固定值 |
|---|---|
| 数据 | `data/processed/esta_full/first_kill.parquet` |
| 数据 SHA-256 | `06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492` |
| 预测时点 | 第一条有效敌方击杀发生后立即预测 |
| 样本量 | 41,027 |
| train/validation/test 行数 | 28,489 / 8,368 / 4,170 |
| train/validation/test 系列赛 | 547 / 156 / 79 |
| 原始/编码特征 | 40 / 82 |
| M28 validation Log Loss | 0.528706 |
| M28 validation AUC | 0.802863 |
| 正式随机种子 | 42 |
| 选择指标 | validation Log Loss |
| 每阶段最小接受改善 | 0.0001 |

所有候选和种子稳定性实验只接收 train 与 validation。候选表、阶段选择表和种子表
禁止出现任何 `test_*` 列。test 只能在参数和正式种子冻结后评估一次。

## 3. 逐阶段候选

采用与 M23 相同、已验收的 greedy sequential search。每阶段只改变一个参数，当前值
必须包含在候选中；最佳候选相对当前配置的 validation Log Loss 改善不足 `0.0001`
时保留当前配置。

| 顺序 | 参数 | 候选值 |
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

共 9 阶段、36 个候选。模型继续使用最多 3,000 棵树、validation
`binary_logloss`、100 轮 early stopping、CPU 确定性模式和正式种子 42。

## 4. 选择与稳定性规则

每阶段按 validation Log Loss 从小到大排序，同值按预先冻结的候选顺序决定。
Accuracy、AUC、Brier 和 ECE10 只作诊断，不参与选择。

参数冻结后用种子 42、43、44、45、46 分别重训，正式模型仍为 seed 42。

| 稳定性项目 | 阻断门槛 |
|---|---:|
| validation Log Loss 五种子范围 | <= 0.002 |
| validation AUC 五种子范围 | <= 0.003 |

## 5. 阶段目标与测试集边界

以下 validation 目标用于判断调参是否有研究价值，不要求为了过关强行改变参数：

| 项目 | 目标 |
|---|---:|
| validation Log Loss 相对 M28 改善 | >= 0.0005 |
| validation AUC | >= 0.800863（最多比 M28 下降 0.002） |
| train-validation AUC 绝对差 | <= 0.030 |

冻结后测试集仍按 M28 门槛验收：Accuracy >= 0.68、AUC >= 0.75、Log Loss <= 0.58、
Brier <= 0.20、ECE10 <= 0.05。必须同时报告更高阶段目标，但是否胜过 M28 或 M21
XGBoost 不是阻断条件。M29 点指标不得替代 M30 的系列赛 bootstrap 配对结论。

## 6. 阻断检查

1. `m28_prerequisite`：M28 16/16 通过，数据、模型与特征合同可复核；
2. `data_contract`：样本、主键、切分和数据哈希与 M28 一致；
3. `feature_contract`：40/82 特征完全一致，无 ID、标签或 split；
4. `candidate_grid`：9 阶段、36 候选、每阶段只改一个参数且包含当前值；
5. `validation_only`：搜索、选择、种子表均无 test 列；
6. `phase_selection`：九阶段选择符合 `0.0001` 规则；
7. `frozen_model`：保存参数、正式种子和选择记录一致；
8. `seed_stability`：五种子范围通过固定门槛；
9. `final_predictions`：4,170 个测试键与 M28 完全一致且概率有效；
10. `minimum_metrics`：冻结模型测试集五项最低门槛通过；
11. `controlled_comparison`：M28、M21 XGBoost、M29 使用同一测试键和指标实现；
12. `external_report`：公开结果比较保留可比性标签；
13. `automated_tests`：完整测试通过；
14. `source_compile`：`src` 全量编译通过；
15. `reproduction_entrypoint`：一键脚本存在，文档明确使用执行策略绕过参数；
16. `artifact_manifest`：模型与正式核心产物齐全并写入哈希清单。

## 7. 正式产物

```text
src/csdemo/m29_post_first_kill_lightgbm_tuning.py
tests/test_m29_post_first_kill_lightgbm_tuning.py
scripts/run_post_first_kill_lightgbm_tuning.ps1
models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib
reports/esta_full_m29/tuning_candidates.csv
reports/esta_full_m29/phase_selections.csv
reports/esta_full_m29/seed_stability.csv
reports/esta_full_m29/test_predictions.csv
reports/esta_full_m29/m29_model_comparison.csv
reports/esta_full_m29/m29_summary.json
reports/esta_full_m29/m29_experiment_manifest.json
reports/esta_full_m29/m29_post_first_kill_lightgbm_tuning_report.md
```

模型二进制由 Git 忽略；规格、参数、模型哈希、概率、指标和报告进入版本控制。

## 8. 下一阶段

M29 通过后进入 M30。M30 冻结 M29 模型，不返回修改网格，并执行系列赛 bootstrap
置信区间、与 M21 XGBoost 的配对不确定性、地图及 LAN/online 稳健性、校准选择和错误分析。

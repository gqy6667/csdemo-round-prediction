# M31 首杀后 LightGBM 模型解释与泄漏审计规格

## 1. 目标

M31 解释 M29/M30 已冻结的首杀后 LightGBM，并回答：

1. 购买状态与四个首杀事件特征分别如何影响预测；
2. Gain、固定测试集 Permutation Importance、TreeSHAP 是否给出一致方向；
3. 82 个编码列能否全部追溯到 40 个预测时点可用的原始特征；
4. LightGBM 与 M19 XGBoost 在同一数据和特征上的解释有多大差异。

本阶段不训练、不调参、不改变阈值、校准、特征或 split。M30 的 4,170 条测试概率与
五项指标必须精确回放，M29 模型运行前后 SHA-256 必须相同。

## 2. 冻结输入

- 数据：`data/processed/esta_full/first_kill.parquet`；
- 模型：`models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib`；
- 前置验收：`reports/esta_full_m30/m30_summary.json`；
- M30 测试概率：`reports/esta_full_m30/test_predictions_enriched.csv`；
- M19 XGBoost 解释：`reports/esta_full_m19/importance_comparison.csv`、
  `macro_group_permutation_auc.csv`；
- M30 外部比较：`reports/esta_full_m30/external_benchmark_comparison.csv`；
- 完整主键：`series_id + game_id + round_id`。

必须验证 M30 已通过、数据和模型 SHA-256、任务名、模型名、40/82 特征、211 棵部署树
和 28,489/8,368/4,170 行 split 均一致。

## 3. 解释方法

### 3.1 LightGBM Gain 与 Split

从冻结 Booster 的 211 棵部署树读取 Gain 和 split count。82 个编码列全部输出，包括
未参与分裂的零重要性列；归一化 Gain 总和必须为 1。

### 3.2 Permutation Importance

- 逐个打乱 82 个编码列，每列重复 20 次；
- 将地图和首杀武器独热列映射回原始特征，同一原始特征的编码列共同打乱，40 组各
  重复 20 次；
- 按预测时点分为 `purchase_end` 与 `first_kill_event` 两个宏观组，各重复 20 次；
- 记录固定测试集 AUC 平均下降、标准差、最小值、最大值，负值必须保留。

测试集只用于解释冻结模型，不依据重要性结果删除特征或重选模型。

### 3.3 TreeSHAP

使用 LightGBM 原生 `pred_contrib=True`，单位为 log-odds。必须验证：

```text
sigmoid(base_value + sum(feature_contributions)) == frozen_probability
```

最大绝对误差不得超过 `1e-10`。

## 4. 特征时间与泄漏合同

允许字段只有 M21/M29 的 40 个原始特征：36 个购买结束字段，加上
`first_kill_advantage_ct`、`first_kill_time`、`first_kill_headshot`、
`first_kill_weapon`。首杀必须是最早有效敌方击杀。

禁止进入模型：

- `series_id`、`game_id`、`round_id` 等身份字段；
- `ct_win`、winner、label、split；
- 首杀之后的后续击杀、伤害、血量、存活人数、位置；
- 下包、拆包、回合结束等未来信息；
- 战队和选手身份。

82 个完整模型列与 SHAP 前 20 列的泄漏失败数必须为 0；40 个原始特征以及两个宏观
时间组必须全部有编码列覆盖。

## 5. 与 M19 XGBoost 的解释对照

两种模型使用同一 41,027 行数据、同一系列赛 split 和同一 82 个编码列。对 Gain、
编码列 Permutation、TreeSHAP 和三者平均排名分别计算：

- 82 列 Spearman 排名相关系数；
- Top 10 交集数量与 Jaccard；
- 每个编码列的排名差。

排序一致性是描述结果，不是阻断目标，不解释为因果关系。

## 6. 案例解释

固定选择三类测试回合：CT 高概率且正确、T 高概率且正确、最高置信错误。每例保存绝对
贡献最大的 10 个编码特征、贡献方向、值、基准 log-odds 和重建概率。主键只用于定位，
不进入模型。

## 7. 阻断验收

1. `m30_prerequisite`：M30 状态、数据/模型哈希、任务与 40/82 特征通过；
2. `model_replay`：测试概率和五项指标最大误差 <= `1e-12`，fit=0；
3. `model_unchanged`：模型运行前后 SHA-256 相同；
4. `importance_methods`：82 列 Gain、20 次编码 permutation、40 组原始 permutation、
   两组宏观 permutation 和 TreeSHAP 完整；
5. `feature_mapping_and_leakage`：82 列全部映射且泄漏失败数为 0；
6. `shap_reconstruction`：最大概率重建误差 <= `1e-10`；
7. `source_and_macro_groups`：40 个原始组与两个时间组覆盖完整；
8. `xgboost_explanation_comparison`：M19 的 82 列和四种排名对照完整；
9. `case_explanations`：三类案例、每类 10 个贡献生成；
10. `external_report`：M30 外部差距原样保留；
11. `automated_tests`：全量测试通过；
12. `source_compile`：源码和测试编译通过；
13. `reproduction_entrypoint`：一键脚本通过合同检查；
14. `artifact_manifest`：输入输出哈希齐全。

## 8. 产物与命令

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_post_first_kill_lightgbm_explanation.ps1
```

```text
src/csdemo/m31_post_first_kill_lightgbm_explanation.py
tests/test_m31_post_first_kill_lightgbm_explanation.py
scripts/run_post_first_kill_lightgbm_explanation.ps1
reports/esta_full_m31/gain_importance.csv
reports/esta_full_m31/permutation_importance_auc.csv
reports/esta_full_m31/grouped_permutation_importance_auc.csv
reports/esta_full_m31/macro_group_permutation_auc.csv
reports/esta_full_m31/shap_importance.csv
reports/esta_full_m31/model_importance_comparison_summary.csv
reports/esta_full_m31/all_feature_leakage_audit.csv
reports/esta_full_m31/case_explanations.csv
reports/esta_full_m31/m31_summary.json
reports/esta_full_m31/m31_experiment_manifest.json
reports/esta_full_m31/m31_post_first_kill_lightgbm_explanation_report.md
```

## 9. 下一阶段

M31 通过后进入 M32：建立单条 JSON/CSV 首杀后 LightGBM 预测接口，复用 M30 仅由
validation 选择的 Identity 校准器，并对输入、派生特征、列顺序和错误返回做验收。

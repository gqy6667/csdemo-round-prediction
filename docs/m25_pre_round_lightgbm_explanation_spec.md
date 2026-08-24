# M25 开局前 LightGBM 模型解释与泄漏审计规格

## 1. 目标

M25 解释 M23/M24 已冻结的购买结束、交火前 LightGBM，并回答：

1. 哪些购买状态推动 CT 或 T 的回合胜率；
2. Gain、测试集 AUC Permutation Importance 和 TreeSHAP 是否给出一致的大方向；
3. 43 个编码列能否全部追溯到 36 个允许的购买结束特征；
4. LightGBM 与 M12 XGBoost 对同一批特征的解释有多大差异。

本阶段不训练、不调参、不改变阈值、校准、特征或 split。M24 的 4,172 条测试概率和
五项指标必须原样回放，模型文件在运行前后的 SHA-256 必须相同。

## 2. 冻结输入

- 数据：`data/processed/esta_full/pre_round.parquet`；
- 模型：`models/esta_full_m23/pre_round_lightgbm_tuned.joblib`；
- 前置验收：`reports/esta_full_m24/m24_summary.json`；
- M24 测试概率：`reports/esta_full_m24/test_predictions_enriched.csv`；
- M12 XGBoost 解释：`reports/esta_full_m12/gain_importance.csv`、
  `permutation_importance_auc.csv`、`shap_importance.csv` 和
  `importance_comparison.csv`；
- M24 外部指标：`reports/esta_full_m24/external_benchmark_comparison.csv`；
- 完整主键：`series_id + game_id + round_id`。

必须验证 M24 已通过、数据和模型 SHA-256、任务名、模型名、36 个原始特征、43 个编码
列、115 棵部署树和 28,522/8,380/4,172 行 split 均匹配。

## 3. 解释方法

### 3.1 LightGBM Gain 与 Split

从冻结 Booster 读取 115 棵部署树的 gain 和 split count。43 个编码列全部输出，包括
未参与分裂的零重要性列；gain 归一化之和必须为 1。

### 3.2 编码列与原始特征 Permutation Importance

在固定测试集上逐个打乱 43 个编码列，每列重复 20 次，记录 AUC 平均下降、标准差、
最小值和最大值。负值必须保留。

将 `map_name_*` 归回 `map_name`。打乱原始特征时，同一原始特征的全部编码列使用同一
行排列，保持独热结构。36 个原始特征各重复 20 次。再按 M14 合同聚合为 context、
score、economy、armor_utility 和 weapons 五个宏观组，各重复 20 次。

测试集只用于解释已冻结模型，不根据重要性结果重新选择特征或模型。

### 3.3 TreeSHAP

使用 LightGBM 原生 `pred_contrib=True` TreeSHAP，单位为 log-odds。最后一列为基准值，
前 43 列为特征贡献。必须验证：

```text
sigmoid(base_value + sum(feature_contributions)) == frozen_probability
```

最大绝对误差不得超过 `1e-10`。

## 4. 特征时间与泄漏合同

唯一允许的原始输入是 `PRE_ROUND_FEATURES` 的 36 列，快照定义为购买结束、交火前。
允许字段包括地图、回合与比分、经济、护甲道具和武器计数。

以下内容禁止进入模型：

- `series_id`、`game_id`、`round_id` 等身份字段；
- `ct_win`、winner、label、split；
- 首杀、后续击杀、伤害、血量、存活人数变化；
- 下包、拆包、回合结束等未来信息；
- 战队和选手身份。

每个编码列必须且只能映射到一个允许原始特征。43 个完整模型列和 TreeSHAP 前 20 列
的泄漏失败数都必须为 0，36 个原始特征及五个宏观组都必须有编码列覆盖。

## 5. 与 M12 XGBoost 的解释对照

两种模型使用同一数据、split 和 43 个编码列。对 Gain、编码列 Permutation、TreeSHAP
及三者平均排名分别计算：

- 43 列 Spearman 排名相关系数；
- Top 10 交集数量；
- Top 10 Jaccard；
- 每个编码列的排名差。

排序一致性是观察结果，不是阻断目标。树算法的 gain 定义及相关特征的归因分配会不同，
因此不把排名差异解释成泄漏，也不把高相关性解释成因果证据。

## 6. 案例解释

固定选择三个测试回合：CT 高概率且正确、T 高概率且正确、最高置信错误。每个案例保存
绝对贡献最大的 10 个特征、方向、编码值、基准 log-odds、原始输出和重建概率。完整
主键用于定位案例，身份字段不进入模型。

## 7. 外部模型差距

M25 不改变预测概率，因此外部指标差距必须与 M24 完全一致。继续输出同口径外部比较，
并明确不同数据、预测时点和切分方式不能用于证明算法排名。

## 8. 技术与命令

```text
src/csdemo/m25_pre_round_lightgbm_explanation.py
tests/test_m25_pre_round_lightgbm_explanation.py
scripts/run_pre_round_lightgbm_explanation.ps1
reports/esta_full_m25/
```

正式命令：

```powershell
.\scripts\run_pre_round_lightgbm_explanation.ps1
```

测试与编译：

```powershell
C:\Users\admin\11\envs\game\python.exe -m unittest tests.test_m25_pre_round_lightgbm_explanation -v
C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests -v
C:\Users\admin\11\envs\game\python.exe -m compileall src tests
```

## 9. 测试策略

- 编码映射：数值列和地图独热列正确，未知列失败；
- 泄漏审计：购买结束字段通过，ID、标签、首杀和未来字段失败；
- 分组打乱：独热列共同打乱，真实信号高于噪声；
- LightGBM：Gain/Split 全列输出，TreeSHAP 形状和概率重建正确；
- 模型对照：完整特征集合、Spearman、Top 10 交集和排名差正确；
- 验收：前置哈希、零 fit、三种解释、案例、外部表、测试、编译和入口完整。

## 10. 阻断验收

1. M24 前置状态、数据哈希、模型哈希、任务和特征合同通过；
2. M24 测试概率及五项指标最大回放误差不超过 `1e-12`，LightGBM fit 调用为 0；
3. 模型文件运行前后 SHA-256 相同；
4. 43 列 Gain/Split、20 次编码 permutation、36 组原始 permutation、五组宏观
   permutation 和 TreeSHAP 全部完成；
5. 43 个编码列全部映射到 36 个原始特征，完整审计和 SHAP 前 20 失败数为 0；
6. TreeSHAP 概率重建最大误差不超过 `1e-10`；
7. M12 的 43 个同名特征和四种排名对照完整；
8. 三类案例及每类前 10 个贡献生成；
9. 外部差距、中文阶段报告、实验清单、自动化测试、源码编译和一键入口通过。

## 11. 边界与后续

始终执行：冻结模型、完整特征映射、保留负 permutation、如实报告算法解释差异。

另立阶段：删除或增加特征、加入战队/选手身份、重新调参、首杀后 LightGBM、实时状态。

禁止执行：根据测试重要性重训、把 SHAP 当成因果关系、提交 ESTA 原始数据或模型。

M25 通过后进入 M26：为冻结的购买结束 LightGBM 建立单条 JSON/CSV 预测接口，并复用
M24 validation-only 选择的 identity 校准器。

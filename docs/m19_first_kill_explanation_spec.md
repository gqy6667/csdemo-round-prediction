# M19 首杀后模型解释与泄漏审计规格

## 1. 目标

M19 解释 M17/M18 已冻结的首杀后 XGBoost，并回答：

1. 哪些购买状态和首杀事件推动 CT/T 胜率；
2. Gain、Permutation Importance 和 TreeSHAP 是否给出一致的大方向；
3. 82 个编码模型列能否全部追溯到 40 个允许的原始特征；
4. 模型离正式阶段目标还差多少，以及与公开模型指标相差多少。

本阶段不训练、不调参、不改变阈值或校准方法，因此 M18 点指标必须保持不变。

## 2. 冻结输入

- 数据：`data/processed/esta_full/first_kill.parquet`；
- 模型：`models/esta_full_m17/first_kill_xgboost_tuned.joblib`；
- 前置验收：`reports/esta_full_m18/m18_summary.json`；
- 内部模型比较：`reports/esta_full_m17/model_comparison.csv`；
- 外部指标：`benchmarks/external_first_kill_tuned_metrics.csv`；
- 完整主键：`series_id + game_id + round_id`；
- 正式测试集：4,170 回合、79 个系列赛。

数据和模型 SHA-256、任务名、原始特征、编码列顺序及 M18 `ready_for_m19` 必须匹配。

## 3. 解释方法

### 3.1 Gain

只统计部署时实际使用的 `best_iteration + 1` 棵树。输出编码列的平均 gain、归一化
gain、分裂次数和排名。

### 3.2 编码列 Permutation Importance

在固定测试集逐列打乱 82 个编码列，每列重复 20 次，记录 AUC 平均下降和标准差。
测试集用于解释已冻结模型，不根据结果重新选择特征或模型。

### 3.3 原始特征分组 Permutation Importance

将 `map_name_*` 归回 `map_name`，将 `first_kill_weapon_*` 归回
`first_kill_weapon`，其余编码列归回同名原始特征。打乱一个原始特征时，其全部编码列
使用同一个行排列，保持独热列之间的结构。40 个原始特征各重复 20 次。

另外把原始特征合并为购买结束特征和四个首杀事件特征，报告两个宏观组的 AUC 下降。

### 3.4 TreeSHAP

使用 XGBoost 原生 TreeSHAP，单位为 log-odds。必须限制到部署树，并验证
`sigmoid(base + sum(SHAP))` 与冻结模型概率的最大绝对误差不超过 `1e-5`。

## 4. 特征时间与泄漏合同

允许的 40 个原始特征只能来自：

- 36 个 M14 购买结束、交火前特征；
- `first_kill_advantage_ct`；
- `first_kill_time`；
- `first_kill_headshot`；
- `first_kill_weapon`。

首杀事件特征在首个有效敌方击杀发生后可用，因此对本任务合法，但对开局前任务非法。
ID、标签、split、第二次及后续击杀、伤害/血量变化、下包状态、回合结束和胜者字段都
禁止进入模型。确定性冗余列 `first_kill_is_ct`、`first_death_is_ct`、
`ct_alive_after_fk`、`t_alive_after_fk`、`alive_diff_ct_after_fk` 继续排除。

每个编码列必须且只能映射到一个允许原始特征；全部 82 列和 TreeSHAP 前 20 列的失败
数都必须为 0。

## 5. 目标距离定义

输出 `target_gap.csv`。对越高越好的指标：

```python
remaining = max(target - current, 0)
margin = max(current - target, 0)
```

对越低越好的指标交换方向。固定检查以下正式阶段目标：

| 项目 | 目标 |
|---|---:|
| Test Accuracy | >= 0.700 |
| Test AUC | >= 0.780 |
| Test Log Loss | <= 0.550 |
| Test Brier | <= 0.185 |
| Test ECE10 | <= 0.030 |
| AUC 95% CI 下界 | >= 0.790 |
| Log Loss 95% CI 上界 | <= 0.540 |
| LAN-online AUC 绝对差 | <= 0.040 |
| 主要地图最低 AUC | >= 0.770 |
| 主要地图最低 AUC CI 下界 | >= 0.700 |

XGBoost 相对逻辑回归的 AUC、Log Loss、Brier 差继续报告，但 M16 没有把领先逻辑
回归设为阻断目标，不虚构通过线。

## 6. 案例解释

固定选择三个测试回合：CT 高概率且正确、T 高概率且正确、最高置信错误。每个案例保存
前 10 个绝对 SHAP 贡献、方向、特征值、基准 log-odds 和重建概率。案例解释回答模型
为什么这样预测，不解释后续交火的因果过程。

## 7. 外部模型差距

继续生成七行外部比较：最接近的 CS156 首杀后逻辑回归、Xenopoulos 实时 WPA
XGBoost、Aakerholt 购买结束 DNN。差值统一为“本项目 - 外部报告”，并保留
`closest_task`、`partial`、`not_comparable` 标签。不同数据和切分不能解释为算法排名。

## 8. 技术与命令

- Python 3.10、XGBoost 3.2、pandas、scikit-learn、matplotlib；
- TreeSHAP 使用 XGBoost 原生接口，不新增依赖；
- 复用 `src/csdemo/m12_explanation.py` 中已测试的通用解释函数；
- 新逻辑位于 `src/csdemo/m19_first_kill_explanation.py`；
- 单元测试位于 `tests/test_m19_first_kill_explanation.py`。

```powershell
.\scripts\run_first_kill_explanation.ps1
C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests -v
C:\Users\admin\11\envs\game\python.exe -m compileall src tests
```

## 9. 测试策略

- 编码映射：数值、地图和武器独热列映射正确，未知列失败；
- 泄漏审计：合法首杀字段通过，ID、标签和未来字段失败；
- 分组打乱：同一原始特征的独热列一起打乱，真实信号高于噪声；
- 目标距离：高/低方向、remaining、margin 和无正式目标的内部比较正确；
- 验收：前置哈希、三种解释、SHAP 重建、案例、外部报告和自动化测试。

## 10. 边界

始终执行：固定模型、完整特征映射、保留负 permutation 值、报告所有未达到目标。

另立阶段：删除/增加特征、加入战队和选手身份、LightGBM、实时状态建模。

禁止执行：根据测试重要性重训、将 SHAP 解释为因果关系、提交本地 ESTA 数据或模型。

## 11. 阻断验收

1. M18 前置状态、数据哈希和模型哈希通过；
2. 模型回放指标与 M18 最大差不超过 `1e-12`，XGBoost fit 调用为 0；
3. Gain、20 次编码 permutation、20 次原始分组 permutation 和 TreeSHAP 完成；
4. 82 个编码列全部映射到 40 个原始特征，泄漏失败数为 0；
5. SHAP 概率重建最大误差不超过 `1e-5`；
6. 三类案例及每类前 10 个贡献生成；
7. 十项正式目标距离、内部模型差和外部七行比较全部生成；
8. 中文报告、CSV、图表、自动化测试和源码编译通过。

## 12. 交付物与后续

```text
reports/esta_full_m19/gain_importance.csv
reports/esta_full_m19/permutation_importance_auc.csv
reports/esta_full_m19/grouped_permutation_importance_auc.csv
reports/esta_full_m19/shap_importance.csv
reports/esta_full_m19/importance_comparison.csv
reports/esta_full_m19/encoded_feature_contract.csv
reports/esta_full_m19/all_feature_leakage_audit.csv
reports/esta_full_m19/target_gap.csv
reports/esta_full_m19/internal_model_gap.csv
reports/esta_full_m19/selected_cases.csv
reports/esta_full_m19/case_explanations.csv
reports/esta_full_m19/m19_summary.json
reports/esta_full_m19/m19_first_kill_explanation_report.md
reports/esta_full_m19/external_benchmark_comparison.csv/.md
```

M19 通过后，M20 建立首杀后单条 JSON/CSV 预测接口；M21 做首杀后 XGBoost 最终验收。
之后进入 LightGBM 同数据对照，再开始实时胜率数据模块。

## 13. 实际结果（2026-08-20）

- 阻断检查 `9/9` 通过，自动化测试 `118` 项通过，可进入 M20；
- M18 概率与指标被原样回放，XGBoost fit 调用为 `0`；
- 82 个编码列全部映射到 40 个允许原始特征，全部特征和 SHAP 前 20 泄漏失败数为 `0`；
- TreeSHAP 概率重建最大绝对误差为 `4.03e-7`；
- 十项正式目标全部通过，所有 `remaining` 均为 `0`；
- 最紧的两个余量是 Log Loss CI 上界 `0.003854`、AUC CI 下界 `0.007731`；
- `first_kill_advantage_ct` 三种方法均排第 1，原始特征分组打乱使 AUC 平均下降 `0.138412`；
- `eq_value_diff_ct` 三种方法均排第 2，分组打乱使 AUC 平均下降 `0.024478`；
- XGBoost 相比逻辑回归只领先 `0.000778` AUC，当前证据不支持“树模型明显更强”；
- 完整中文结果见 `reports/esta_full_m19/m19_first_kill_explanation_report.md`。

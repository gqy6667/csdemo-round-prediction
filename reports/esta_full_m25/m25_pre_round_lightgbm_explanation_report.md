# M25 开局前 LightGBM 模型解释与泄漏审计

## 结论

M25 阻断检查 14/14 通过，状态为 `passed`。本阶段没有训练、调参、删特征或修改
阈值；解释对象仍是 M23/M24 冻结的购买结束、交火前 LightGBM。

测试集仍为 4,172 回合。Accuracy 0.650767、AUC 0.727846、Log Loss 0.591437、Brier 0.205201、ECE10 0.018875，与 M24 最大差 0.000e+00。

## 解释完整性

冻结模型包含并部署 115 棵树。LightGBM 原生
TreeSHAP 重建概率的最大绝对误差为 7.772e-16；模型文件运行前后 SHA-256 一致为 `3a95983ed73cd99ae0178a16009036d48510e1ad091d33994cf296dfc69244fd`。

43 个编码列全部映射到 36 个 M14 购买结束特征和五个宏观组。完整审计失败 0，SHAP 前 20 失败 0。首杀、伤害、血量、下包、标签、
ID、战队和选手身份均未进入模型。

## 原始特征重要性

| 特征 | 组 | Gain 排名 | 分组置换排名 | SHAP 排名 | 测试 AUC 平均下降 |
|---|---|---:|---:|---:|---:|
| eq_value_diff_ct | economy | 1 | 1 | 1 | 0.074344 |
| helmet_diff_ct | armor_utility | 2 | 2 | 4 | 0.003992 |
| ct_eq_value | economy | 4 | 3 | 2 | 0.002855 |
| t_eq_value | economy | 5 | 6 | 3 | 0.002174 |
| ct_armor | armor_utility | 8 | 8 | 7 | 0.001145 |
| rifle_diff_ct | weapons | 7 | 10 | 6 | 0.000836 |
| score_diff_ct | score | 9 | 5 | 9 | 0.002315 |
| grenade_diff_ct | armor_utility | 3 | 16 | 5 | 0.000075 |
| map_name | context | 10 | 7 | 8 | 0.001780 |
| armor_diff_ct | armor_utility | 6 | 9 | 11 | 0.000920 |
| ct_m4a1_s | weapons | 15 | 4 | 10 | 0.002484 |
| t_armor | armor_utility | 11 | 12 | 12 | 0.000321 |
| ct_cash | economy | 14 | 19 | 14 | 0.000045 |
| t_rifles | weapons | 18 | 17 | 16 | 0.000075 |
| cash_diff_ct | economy | 12 | 26 | 15 | 0.000003 |

三种方法回答不同问题：Gain 统计分裂带来的训练损失下降，Permutation
衡量固定测试 AUC 对打乱输入的敏感度，TreeSHAP 衡量单回合 log-odds
贡献。相关经济列和差值列会分摊信号，因此不应只看一种排名。

## 宏观特征组

| M14 特征组 | 编码列数 | AUC 平均下降 | 标准差 |
|---|---:|---:|---:|
| economy | 6 | 0.133319 | 0.009084 |
| armor_utility | 10 | 0.010098 | 0.003648 |
| weapons | 15 | 0.003807 | 0.002053 |
| score | 3 | 0.002964 | 0.001791 |
| context | 9 | 0.002014 | 0.000963 |

## 与 M12 XGBoost 的解释差异

| 方法 | 43 列 Spearman | Top 10 交集 | Top 10 Jaccard |
|---|---:|---:|---:|
| gain | 0.730 | 8/10 | 0.667 |
| permutation_auc | 0.751 | 9/10 | 0.818 |
| tree_shap | 0.866 | 9/10 | 0.818 |
| mean_rank | 0.835 | 9/10 | 0.818 |

排名差异不构成验收失败。LightGBM 和 XGBoost 的树生长策略、分裂方式及
相关特征归因方式不同；本对照只说明两个冻结模型如何使用同一批输入，不能
证明某个特征对胜负具有因果作用。

## SHAP 前 20 泄漏检查

| 排名 | 编码列 | 原始特征 | 组 | 结果 |
|---:|---|---|---|---|
| 1 | eq_value_diff_ct | eq_value_diff_ct | economy | pass |
| 2 | ct_eq_value | ct_eq_value | economy | pass |
| 3 | t_eq_value | t_eq_value | economy | pass |
| 4 | helmet_diff_ct | helmet_diff_ct | armor_utility | pass |
| 5 | grenade_diff_ct | grenade_diff_ct | armor_utility | pass |
| 6 | rifle_diff_ct | rifle_diff_ct | weapons | pass |
| 7 | ct_armor | ct_armor | armor_utility | pass |
| 8 | score_diff_ct | score_diff_ct | score | pass |
| 9 | ct_m4a1_s | ct_m4a1_s | weapons | pass |
| 10 | armor_diff_ct | armor_diff_ct | armor_utility | pass |
| 11 | t_armor | t_armor | armor_utility | pass |
| 12 | map_name_de_inferno | map_name | context | pass |
| 13 | t_cash | t_cash | economy | pass |
| 14 | ct_cash | ct_cash | economy | pass |
| 15 | map_name_de_overpass | map_name | context | pass |
| 16 | cash_diff_ct | cash_diff_ct | economy | pass |
| 17 | map_name_de_dust2 | map_name | context | pass |
| 18 | t_rifles | t_rifles | weapons | pass |
| 19 | t_ak47 | t_ak47 | weapons | pass |
| 20 | ct_helmets | ct_helmets | armor_utility | pass |

## 三个固定案例

| 类型 | series_id | game_id | round_id | 地图 | 真实标签 | CT 概率 |
|---|---|---|---|---|---:|---:|
| ct_high_probability | f7282b88-c653-4ce5-b4b6-8a2963548662 | lan:00b7d807-ecf2-49c5-aa70-0ee1bfc0478c | lan:00b7d807-ecf2-49c5-aa70-0ee1bfc0478c_17 | de_nuke | 1 | 0.970559 |
| t_high_probability | f9f86156-544c-4b95-b4ca-cb5b33386198 | online:58775236-4620-43ad-8433-594691181531 | online:58775236-4620-43ad-8433-594691181531_17 | de_vertigo | 0 | 0.044116 |
| high_confidence_error | 90da2c53-5a02-4f16-8abe-f2235da5ffbd | online:478d378e-e7c1-4d64-a3f3-679ee18f27b5 | online:478d378e-e7c1-4d64-a3f3-679ee18f27b5_17 | de_nuke | 0 | 0.970418 |

案例中的正 SHAP 推向 CT，负 SHAP 推向 T。它解释模型为什么输出该概率，
不解释该回合后来发生的交火原因。

## 外部指标

外部比较共 4 行，与 M24 完全一致。最接近的购买结束 DNN
仍使用不同数据和随机行级切分，其 Accuracy 和 Log Loss 只能作为背景参照。
完整逐行差值见 `external_benchmark_comparison.csv`。

## 验收与下一步

自动化测试 192 项通过，源码编译通过。
M25 的结论是模型解释链路完整、无特征时间泄漏、冻结预测未漂移。解释结果
不会触发 test 驱动的重训。下一阶段 M26 建立购买结束 LightGBM 的单条
JSON/CSV 预测接口，并复用 M24 validation-only 选择的 identity 校准器。

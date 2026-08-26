# M31 首杀后 LightGBM 模型解释与泄漏审计

## 结论

M31 阻断检查 14/14 通过，状态为 `passed`。本阶段只解释 M29/M30 已冻结模型，
没有训练、调参、删特征、修改阈值或改变校准方法。

测试集仍为 4,170 回合。Accuracy 0.742926、AUC 0.808255、Log Loss 0.524063、Brier 0.176003、ECE10 0.014191；与 M30 的概率最大差为 1.110e-16。

## 模型与泄漏审计

冻结模型部署 211 棵树。LightGBM 原生 TreeSHAP 重建概率最大绝对误差为 7.772e-16；
模型运行前后 SHA-256 均为 `35ce17435a3716efcfdd49dd5ca13ff441e75c65512322627249e8920546a5b5`。

82 个编码列全部追溯到 40 个原始特征；完整审计失败 0，TreeSHAP 前 20 失败 0。主键、标签、首杀后的伤害/击杀、下包与回合结束信息、战队和选手身份均未进入模型。

## 原始特征重要性

| 特征 | 时点组 | Gain 排名 | 分组置换排名 | SHAP 排名 | 测试 AUC 平均下降 |
|---|---|---:|---:|---:|---:|
| first_kill_advantage_ct | first_kill_event | 1 | 1 | 1 | 0.129081 |
| eq_value_diff_ct | purchase_end | 2 | 2 | 2 | 0.025300 |
| ct_eq_value | purchase_end | 3 | 3 | 3 | 0.005917 |
| t_eq_value | purchase_end | 6 | 4 | 4 | 0.001461 |
| helmet_diff_ct | purchase_end | 4 | 6 | 5 | 0.000748 |
| rifle_diff_ct | purchase_end | 7 | 8 | 6 | 0.000677 |
| map_name | purchase_end | 12 | 5 | 7 | 0.001438 |
| armor_diff_ct | purchase_end | 9 | 10 | 11 | 0.000461 |
| ct_armor | purchase_end | 10 | 12 | 8 | 0.000306 |
| grenade_diff_ct | purchase_end | 5 | 16 | 9 | 0.000092 |
| score_diff_ct | purchase_end | 14 | 7 | 12 | 0.000746 |
| ct_m4a1_s | purchase_end | 15 | 9 | 13 | 0.000549 |
| first_kill_time | first_kill_event | 13 | 11 | 14 | 0.000428 |
| ct_grenades | purchase_end | 16 | 13 | 18 | 0.000272 |
| t_armor | purchase_end | 11 | 23 | 15 | 0.000023 |

Gain、固定测试集置换重要性和 TreeSHAP 回答的问题不同，相关经济列与差值列也会分摊信号。因此排名用于解释冻结模型，不用于依据测试集删特征。

## 购买结束与首杀事件

| 时点组 | 编码列数 | AUC 平均下降 | 标准差 |
|---|---:|---:|---:|
| purchase_end | 43 | 0.144799 | 0.007229 |
| first_kill_event | 39 | 0.132496 | 0.005668 |

`purchase_end` 是首杀发生前已知的购买和比分状态；`first_kill_event` 仅包含最早有效敌方击杀的阵营优势、时间、爆头和武器。两组同时打乱的结果不是因果效应。

## 与 M19 XGBoost 的解释对照

| 方法 | 82 列 Spearman | Top 10 交集 | Top 10 Jaccard |
|---|---:|---:|---:|
| gain | 0.818 | 8/10 | 0.667 |
| permutation_auc | 0.512 | 8/10 | 0.667 |
| tree_shap | 0.871 | 8/10 | 0.667 |
| mean_rank | 0.840 | 9/10 | 0.818 |

两模型使用完全相同的样本、split 和 82 个编码列。排名一致性只描述两种树模型如何分配同一批信号，不是验收门槛，也不表示特征具有因果作用。

## TreeSHAP 前 20 泄漏检查

| 排名 | 编码列 | 原始特征 | 时点组 | 结果 |
|---:|---|---|---|---|
| 1 | first_kill_advantage_ct | first_kill_advantage_ct | first_kill_event | pass |
| 2 | eq_value_diff_ct | eq_value_diff_ct | purchase_end | pass |
| 3 | ct_eq_value | ct_eq_value | purchase_end | pass |
| 4 | t_eq_value | t_eq_value | purchase_end | pass |
| 5 | helmet_diff_ct | helmet_diff_ct | purchase_end | pass |
| 6 | rifle_diff_ct | rifle_diff_ct | purchase_end | pass |
| 7 | ct_armor | ct_armor | purchase_end | pass |
| 8 | grenade_diff_ct | grenade_diff_ct | purchase_end | pass |
| 9 | armor_diff_ct | armor_diff_ct | purchase_end | pass |
| 10 | first_kill_weapon_AK-47 | first_kill_weapon | first_kill_event | pass |
| 11 | score_diff_ct | score_diff_ct | purchase_end | pass |
| 12 | map_name_de_inferno | map_name | purchase_end | pass |
| 13 | ct_m4a1_s | ct_m4a1_s | purchase_end | pass |
| 14 | first_kill_time | first_kill_time | first_kill_event | pass |
| 15 | t_armor | t_armor | purchase_end | pass |
| 16 | smg_diff_ct | smg_diff_ct | purchase_end | pass |
| 17 | map_name_de_overpass | map_name | purchase_end | pass |
| 18 | t_grenades | t_grenades | purchase_end | pass |
| 19 | ct_grenades | ct_grenades | purchase_end | pass |
| 20 | cash_diff_ct | cash_diff_ct | purchase_end | pass |

## 三个固定案例

| 类型 | series_id | game_id | round_id | 地图 | 真实标签 | CT 概率 |
|---|---|---|---|---|---:|---:|
| ct_high_probability | 63e0d98d-fd95-4698-9d12-808508005ee2 | online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b | online:1f4b1506-6f45-43ee-8b4c-82961eb4bd5b_8 | de_vertigo | 1 | 0.991469 |
| t_high_probability | 343edef3-916d-4f85-93b4-3b32c0c70d00 | lan:f912d2b7-3bec-47a2-9609-0ab73fb263e3 | lan:f912d2b7-3bec-47a2-9609-0ab73fb263e3_17 | de_mirage | 0 | 0.014661 |
| high_confidence_error | 90da2c53-5a02-4f16-8abe-f2235da5ffbd | online:478d378e-e7c1-4d64-a3f3-679ee18f27b5 | online:478d378e-e7c1-4d64-a3f3-679ee18f27b5_17 | de_nuke | 0 | 0.989839 |

案例表中的正 SHAP 推向 CT，负 SHAP 推向 T，单位是 log-odds。主键只用于定位案例，不进入模型。每例的 10 个主要贡献见 `case_explanations.csv`。

## 验收与下一步

自动化测试 254 项通过，源码编译通过；外部比较 7 行与 M30 原样保留。M31 证明解释链完整、预测未漂移且没有时间泄漏。下一阶段 M32 建立单条 JSON/CSV 首杀后 LightGBM 推理接口，并复用 M30 由 validation 选择的 identity 校准器。

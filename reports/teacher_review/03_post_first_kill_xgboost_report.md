# 首杀后 XGBoost 独立报告

## 报告结论

本报告对应 M15–M21 已验收的“首杀后 XGBoost”模型。M21 最终状态为 **passed**，17/17 个阻断检查全部通过；最终验收只回放冻结的 M17 模型与 M18 校准器，XGBoost `fit` 调用次数为 0，4,170 条测试概率最大回放误差为 `1.110223e-16`。

冻结测试结果为：Accuracy `0.744125`、AUC `0.809837`、Log Loss `0.523146`、Brier `0.175656`、ECE10 `0.015450`。五项点指标、全局区间、来源差异和主要地图共 10 项正式目标全部通过。

该结果回答的是“首杀刚发生后”的胜率问题，不是购买结束模型的直接替代。首杀后 LightGBM 现已完成 M33 最终验收；本报告仍只陈述 XGBoost 自身证据，公平比较见第四份报告：[首杀后 LightGBM 报告](04_post_first_kill_lightgbm_report.md)。

正式证据见 [M21 最终验收报告](../esta_full_m21/m21_first_kill_final_acceptance_report.md)、[M21 summary](../esta_full_m21/m21_summary.json) 和 [M21 实验清单](../esta_full_m21/m21_experiment_manifest.json)。

## 1. 研究问题与预测时点

研究问题是：**双方购买完成后最早一次有效敌对击杀刚刚发生，此时根据购买结束信息和该首杀事件，预测 CT 最终赢得本回合的概率。**

严格预测时点定义为：

- 购买结束快照仍是经济、装备和比分信息的来源；
- 在当前地图、当前回合内按最小有效 `tick` 选择第一条敌对击杀；
- 队友击杀、自杀和无法可靠排序的事件不作为首杀；
- 模型只使用这一首杀事件当时已经可知的字段，不使用第二次击杀、后续伤害、炸弹结果或回合胜负；
- 输出 `P(CT win)`，T 方概率为其补数。

这里的“首杀后”表示事件刚刚发生后的离散快照，不是持续更新的实时胜率。实时课题还需要按后续事件或时间间隔生成更多快照。

## 2. M15 数据修复与样本形成

首杀后模型不能直接沿用早期错误事件表。M15 先用完整主键 `series_id + game_id + round_id` 连接回合与击杀，再按 `tick` 选择当前回合最早的有效敌对击杀。

| 项目 | 数量 |
|---|---:|
| 购买结束标准回合 | 41,074 |
| 有有效首杀的样本 | 41,027 |
| 无有效首杀而排除的回合 | 47 |
| 系列赛 | 782 |
| 地图比赛 `game_id` | 1,558 |

与修复前的同键数据比较，有 14,357 条首杀事件发生变化，占 34.994%；其中首杀阵营变化 6,697 条、首杀武器变化 9,669 条。原因是旧逻辑可能按不可靠的秒数或错误回合关联选择事件，而修复后以地图内回合主键和最小 tick 为准。

M15 审计确认：重复完整键 0、事件字段链接不一致 0、标签不一致 0、击杀后人数推导错误 0、跨 split 系列赛/比赛/回合均为 0。被排除的 47 回合保存在 [excluded rounds](../esta_full_m15/excluded_rounds.csv)，完整修复证据见 [M15 数据报告](../esta_full_m15/m15_first_kill_data_report.md)。

## 3. 70/20/10 系列赛级切分

首杀后数据复用 M14 冻结的 782 条 `series_id -> split` 清单，不重新随机分配。因为 47 个无有效首杀回合被排除，回合数与购买结束数据略有不同，但系列赛数量保持一致。

| split | 系列赛 | 首杀后样本 | 样本比例 | 用途 |
|---|---:|---:|---:|---|
| train | 547 | 28,489 | 69.440% | 拟合候选模型 |
| validation | 156 | 8,368 | 20.396% | 调参、早停、校准选择 |
| test | 79 | 4,170 | 10.164% | 冻结最终评估 |
| 合计 | 782 | 41,027 | 100% | |

跨 split 的 `series_id`、`game_id`、`round_id` 均为 0，完整键重复行为 0。测试集只在参数和种子协议冻结后评估一次，不参与模型拟合、早停、候选选择或校准器选择。冻结分配见 [M21 split assignments](../esta_full_m21/split_assignments.csv)。

## 4. 标签与特征合同

标签仍为 `ct_win`：CT 赢得回合记为 1，T 赢得回合记为 0。

模型使用 40 个编码前特征和 82 个编码后特征：

- 36 个购买结束特征：地图、回合序号、比分、经济、护甲/道具和武器；
- 4 个首杀事件特征：`first_kill_advantage_ct`、`first_kill_time`、`first_kill_headshot`、`first_kill_weapon`。

`first_kill_is_ct`、`first_death_is_ct`、`ct_alive_after_fk`、`t_alive_after_fk`、`alive_diff_ct_after_fk` 与阵营优势存在确定性冗余，正式 `canonical_event` 合同将它们排除，避免同一件事以多列重复表达。

身份列、split、标签、后续击杀、伤害、血量、位置和炸弹结果均被禁止进入模型。训练集确定地图和首杀武器类别，validation/test 只对齐已有编码列。机器可读合同见 [M16 feature contract](../esta_full_m16/feature_contract.csv)。

## 5. 基线与首杀信息对照

M16 在同一批 41,027 个首杀后样本、相同 split 和相同未经调参 XGBoost 参数下，比较“只用 36 个购买结束特征”与“再加入 4 个首杀事件特征”。这是信息增量对照，不是算法提升：

| 测试指标 | 购买结束特征控制组 | 加入首杀事件 | 变化 |
|---|---:|---:|---:|
| Accuracy | 0.649400 | 0.745324 | +0.095923 |
| AUC | 0.722889 | 0.808896 | +0.086007 |
| Log Loss | 0.593823 | 0.524753 | -0.069069 |
| Brier | 0.206026 | 0.176265 | -0.029761 |
| ECE10 | 0.020274 | 0.010908 | -0.009365 |

这个对照说明首杀阵营、时间、爆头和武器携带了大量新的回合内信息。它不能证明“首杀后 XGBoost 算法优于购买结束 XGBoost 算法”，因为真正改变的是可用信息时点。控制表见 [M16 feature control](../esta_full_m16/m16_feature_control.csv)。

M16 同时保留常数先验、逻辑回归和未调参 XGBoost；测试 AUC 分别约为 0.5000、0.8091、0.8089，说明首杀信号对简单模型也很强，不能把全部提升归因于树模型复杂度。

## 6. XGBoost 训练与受控调参

M17 只使用 train 与 validation，以 validation Log Loss 为选择指标，最小阶段改进门槛为 `0.0001`。搜索包含 8 个顺序阶段、39 个候选；接受了树上限/早停策略、`max_depth` 和 `subsample` 三项变化，其余阶段保留 incumbent。

正式参数为：

| 参数 | 冻结值 |
|---|---:|
| `n_estimators` 上限 | 1,500 |
| `early_stopping_rounds` | 50 |
| `max_depth` | 2 |
| `min_child_weight` | 1 |
| `learning_rate` | 0.03 |
| `subsample` | 0.90 |
| `colsample_bytree` | 0.85 |
| `reg_alpha` | 0 |
| `reg_lambda` | 1 |
| `random_state` | 42 |

validation 早停得到 `best_iteration = 408`，部署固定为 409 棵树。种子 42–46 的 validation Log Loss 极差为 `0.000130`，AUC 极差为 `0.000318`，通过稳定性门槛。

相对 M16 未调参 XGBoost，正式模型测试 AUC 改善 `0.000941`、Log Loss 改善 `0.001607`、Brier 改善 `0.000609`，但 Accuracy 下降 `0.001199`、ECE10 恶化 `0.004541`。这说明调参收益较小，首杀信息本身才是主要变化。完整过程见 [M17 调参报告](../esta_full_m17/m17_first_kill_tuning_report.md)。

## 7. 校准规则

M18 只在 validation 上以 `series_id` 分组做 5 折 OOF，对比未校准、sigmoid、isotonic，并按 OOF Log Loss、Brier 和预先固定顺序选择。最终方法为 `uncalibrated`，即保留原始 XGBoost 概率。

测试集上其他方法即使某个单项数值更好，也不能推翻 validation 的选择。校准器仅保存这个已冻结决策并绑定数据及模型哈希。详见 [M18 评估报告](../esta_full_m18/m18_first_kill_evaluation_report.md)。

## 8. 冻结测试指标与不确定性

以下点估计来自 4,170 个测试回合。95% 区间以 79 个完整系列赛为重采样单位，执行 2,000 次 bootstrap，随机种子 42。

| 指标 | 点估计 | 95% 系列赛 bootstrap CI | 方向 |
|---|---:|---:|---|
| Accuracy | `0.744125` | `[0.731877, 0.756081]` | 越高越好 |
| AUC | `0.809837` | `[0.797731, 0.822081]` | 越高越好 |
| Log Loss | `0.523146` | `[0.509747, 0.536146]` | 越低越好 |
| Brier | `0.175656` | `[0.170040, 0.181184]` | 越低越好 |
| ECE10 | `0.015450` | `[0.012111, 0.031797]` | 越低越好 |

完整精度来源见 [M18 global bootstrap CSV](../esta_full_m18/global_bootstrap_95ci.csv)。AUC 区间下界 `0.797731` 高于阶段目标 0.79，Log Loss 区间上界 `0.536146` 低于阶段目标 0.54。

## 9. 正式目标与稳健性

M21 冻结的 10 项正式目标全部通过：

| 目标 | 当前 | 门槛 | 结果 |
|---|---:|---:|---|
| Accuracy | 0.744125 | >= 0.700 | 通过 |
| AUC | 0.809837 | >= 0.780 | 通过 |
| Log Loss | 0.523146 | <= 0.550 | 通过 |
| Brier | 0.175656 | <= 0.185 | 通过 |
| ECE10 | 0.015450 | <= 0.030 | 通过 |
| AUC 95% CI 下界 | 0.797731 | >= 0.790 | 通过 |
| Log Loss 95% CI 上界 | 0.536146 | <= 0.540 | 通过 |
| LAN/Online AUC 绝对差 | 0.010276 | <= 0.040 | 通过 |
| 主要地图最低 AUC | 0.783901 | >= 0.770 | 通过 |
| 主要地图最低 AUC CI 下界 | 0.750719 | >= 0.700 | 通过 |

LAN AUC 为 `0.803798`，Online AUC 为 `0.814074`，LAN - Online 差为 `-0.010276`，95% CI `[-0.034586, 0.014829]` 包含 0。模型有 90 个预测方概率至少 0.80 但结果错误的案例，结构化复核 30 个；高置信并不等于确定结果。

按来源、地图、首杀阵营、时间、武器、爆头、回合阶段和装备档位的明细均在 [M18 评估报告](../esta_full_m18/m18_first_kill_evaluation_report.md)。

## 10. 模型解释

M19 使用 XGBoost gain、20 次 grouped permutation 和原生 TreeSHAP。40 个原始特征全部映射到 82 个编码列，全部特征与 Top 20 泄漏失败数均为 0；TreeSHAP 概率重构最大误差为 `4.029525e-07`。

三种方法都把 `first_kill_advantage_ct` 排在首位，随后是购买结束装备价值差、头盔/护甲差、双方装备价值和地图等。宏观组 permutation 中，购买结束组平均 AUC 下降 `0.142717`，首杀事件组下降 `0.136379`，说明模型同时依赖首杀和先前经济状态，而不是只看首杀阵营。

这些重要性结果属于非因果解释：它们描述模型对当前数据的依赖，不能证明首杀或某种武器在所有条件下会产生同样的因果效应。详见 [M19 解释报告](../esta_full_m19/m19_first_kill_explanation_report.md)。

## 11. 预测接口

M20 的单条 JSON/CSV 接口要求 27 个购买结束基础字段和 4 个首杀字段，共 31 个输入；程序派生 9 个差值，形成 40 个原始特征并对齐 82 个编码列。接口验证模型、校准器和数据哈希，拒绝未来字段、非法事件值、未知类别及不一致输入。

JSON 与 CSV 对同一快照概率完全一致。示例输出 CT/T 为 0.718764/0.281236，但示例只证明接口工作，不改变固定测试指标。见 [M20 接口报告](../esta_full_m20/m20_first_kill_interface_report.md)。

## 12. 外部参照与比较边界

M21 外部表中，公开 action-value 工作报告 AUC 0.7913、Log Loss 0.5353、Brier 0.1842；本模型的点指标分别好 `0.018537`、`0.012154`、`0.008544`。但该工作的问题、状态表示和数据不同，被标记为 `not_comparable`，不能据此宣称算法领先。

公开 DNN 的购买结束任务报告 Accuracy 0.679220、Log Loss 0.567860；本模型相差 +6.49 个百分点和 -0.044714，但预测时点不同，只能算部分参照。完整来源与可比性见 [M21 外部对照](../esta_full_m21/external_benchmark_comparison.md)。

同理，M14 购买结束 XGBoost 到 M21 首杀后 XGBoost 的指标变化包含新增首杀信息和两条测试样本差异，不是纯调参收益，也不是算法提升。同一首杀后合同的 LightGBM 公平比较现已完成，见[第四份报告](04_post_first_kill_lightgbm_report.md)和[老师查收总索引](README.md)。

## 13. 局限性

- 47 个没有有效敌对首杀的回合被排除，模型的适用总体是“存在可识别首杀的回合”。
- 当前首杀快照只保留四个 canonical 事件字段，尚未使用首杀发生位置、双方剩余血量、已消耗道具或战术空间控制。
- 系列赛级随机切分控制同场泄漏，但尚未做按比赛时间的未来外推测试。
- 未加入战队和选手身份；在时间切分、未知身份回退和阵容变更处理完成前不应加入。
- 高置信错误仍存在，模型概率不能解释为对单个回合的确定判断。
- ESTA 覆盖的是特定时期的职业比赛，游戏版本、地图池和经济规则变化可能导致分布漂移。
- 当前是单一首杀后时点，不是连续实时系统。

## 14. 一键复现与关键哈希

环境为 Python `3.10.20`、XGBoost `3.2.0`、scikit-learn `1.7.2`，解释器为 `C:\Users\admin\11\envs\game\python.exe`。正式模型使用 CPU，不要求 CUDA。

核验冻结的 M15–M21 产物：

```powershell
.\scripts\run_first_kill_pipeline.ps1
```

从 M14 冻结产物重建首杀后流水线：

```powershell
.\scripts\run_first_kill_pipeline.ps1 -RebuildFirstKill
```

从 `C:\project1\data\esta` 完整重建：

```powershell
.\scripts\run_first_kill_pipeline.ps1 -FullRebuild
```

入口见 [run_first_kill_pipeline.ps1](../../scripts/run_first_kill_pipeline.ps1)，环境锁定见 [environment.yml](../../environment.yml) 和 [requirements-lock.txt](../../requirements-lock.txt)。M21 冻结实验对应 Git commit `5f608a20f0cb9b39e85af87c825b990535646f91`。

| 关键产物 | SHA-256 |
|---|---|
| 原始 ESTA 目录清单 | `07391e1aa728921fa84cc832bec81383d3ea3c6dfe684c1e05e7303d66a4c68e` |
| `first_kill.parquet` | `06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492` |
| `first_kill_xgboost_tuned.joblib` | `ecfaaf93031e78207f81ab5ad9674020657018c0601953238ee6b68e367e8279` |
| `first_kill_calibrator.joblib` | `661db6964786dde1276dbeb6c0cf3f175858ad0bf02ecb109bb2f72c45074157` |
| `run_first_kill_pipeline.ps1` | `59402fde18ef944a99d3d6b44155c31135fc009e25658bdc3a611ae3086161a2` |

## 15. 最终判断

首杀后 XGBoost 已完成“事件主键修复 -> 固定系列赛切分 -> canonical 特征合同 -> validation-only 调参 -> 系列赛级不确定性与校准 -> 解释与接口 -> 最终哈希回放”的完整证据链。它达到 M21 的全部正式目标，并作为首杀后 LightGBM 控制变量实验的冻结基线；两种算法的配对比较应以第四份报告的同回合系列赛 bootstrap 为准。

报告状态：**已完成并可复核**。

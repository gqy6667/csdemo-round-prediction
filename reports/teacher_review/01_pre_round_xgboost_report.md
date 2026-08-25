# 购买结束、交火前 XGBoost 独立报告

## 报告结论

本报告对应项目 M14 已验收的“购买结束、交火前 XGBoost”模型。模型在冻结测试集上达到阶段最低验收门槛，M14 状态为 **passed**，可以作为第一个正式课题结果；但 Accuracy、AUC、Log Loss、Brier 四项更高阶段目标均未达到，因此结论是“达到当前可接受基线”，不是“模型已经最优”。

冻结测试结果为：Accuracy `0.647411`、AUC `0.727122`、Log Loss `0.591733`、Brier `0.205294`、ECE10 `0.023198`。完整验收证据见 [M14 最终验收报告](../esta_full_m14/m14_final_acceptance_report.md) 和 [M14 实验清单](../esta_full_m14/m14_experiment_manifest.json)。

## 1. 研究问题与预测时点

研究问题是：**只利用双方购买完成后的装备、经济、比分和地图信息，预测本回合最终由 CT 方获胜的概率。**

严格预测时点定义为：

- 使用冻结时间结束后的第一个可用快照；
- 此时双方已经完成本回合购买；
- 快照必须仍在首次交火、首次伤害和首次击杀之前；
- 模型输出 `P(CT win)`，`P(T win) = 1 - P(CT win)`；
- 0.5 仅作为 Accuracy 和混淆矩阵的分类阈值，主要研究对象仍是概率质量。

这个课题与“首杀后预测”不是同一个信息条件。后者已经观察到一次击杀和人数变化，因此不能把两个预测时点的指标差解释为算法优劣。

## 2. 数据来源与样本量

数据来自本地 ESTA 数据集的 LAN 与 Online 两部分：

| 项目 | 数量 |
|---|---:|
| 原始 `.json.xz` 文件 | 1,558 |
| LAN 文件 | 680 |
| Online 文件 | 878 |
| 标准化回合 | 41,074 |
| 标准化击杀事件 | 268,640 |
| 最终购买结束样本 | 41,074 |

数据身份审计结果为：重复 `round_id` 0、孤立击杀 0、非法标签 0。质量闸门为 error 0、warning 0、info 47。用于模型的数据文件是 `data/processed/esta_full/pre_round.parquet`，其冻结 SHA-256 见第 11 节。

预测时点与样本合同的正式来源是 [M14 实验清单](../esta_full_m14/m14_experiment_manifest.json)；数据质量与前序修复路径可从 [项目路径文档](../../docs/project_paths.md) 继续追溯。

## 3. 70/20/10 切分与泄漏控制

切分单位是 `series_id`，不是单回合随机行。比例按系列赛数量约为 70%/20%/10%，固定分配如下：

| split | 系列赛 | 回合样本 | 用途 |
|---|---:|---:|---|
| train | 547 | 28,522 | 拟合候选模型 |
| validation | 156 | 8,380 | 早停、候选选择、校准器选择 |
| test | 79 | 4,172 | 冻结后的最终一次评估 |
| 合计 | 782 | 41,074 | 100% |

泄漏审计结果：跨 split 的系列赛 0、比赛地图文件 `game_id` 0、回合 0，重复回合键 0。测试集不参与模型选择、调参或校准器选择，也没有把测试指标写入候选参数表。

系列赛级切分的意义是：同一场系列赛中的相近回合不会被拆到训练集和测试集，从而避免模型借助同场比赛的重复环境获得过于乐观的结果。冻结分配表见 [M14 split assignments](../esta_full_m14/split_assignments.csv)。

## 4. 标签与特征合同

标签为 `ct_win`：CT 赢得该回合记为 1，T 赢得该回合记为 0。

模型使用 36 个编码前特征，地图独热编码后为 43 个特征，分为五组：

| 特征组 | 主要内容 |
|---|---|
| context | 地图、回合序号 |
| score | CT/T 当前比分及 CT 视角比分差 |
| economy | 双方装备价值、现金及差值 |
| armor_utility | 护甲、头盔、拆弹器、手雷及差值 |
| weapons | AK、M4A4、M4A1-S、AWP、步枪、冲锋枪及差值 |

以下内容不进入模型：`series_id`、`game_id`、`round_id`、`split`、标签本身，以及首次伤害、首次击杀、首杀武器、首杀时间、击杀后存活人数、炸弹结果等预测时点之后的信息。`ct_alive`、`t_alive` 和人数差在合格快照中恒为 5v5，也作为常量从正式模型特征中排除。

特征合同和接口检查均已通过，完整定义见 [M13 接口报告](../esta_full_m13/m13_interface_report.md) 与 [当前 schema](../../src/csdemo/schema.py)。

## 5. 模型与训练规则

正式模型是 `XGBClassifier` 二分类树模型，目标函数为 `binary:logistic`。基线阶段同时保留训练集先验常数模型、逻辑回归和 XGBoost，以确认复杂模型没有脱离基础参照。

受控调参只读取 train 与 validation：每轮只改变声明的参数，按 validation Log Loss 选择；测试集保持封存。最终关键参数为：

| 参数 | 冻结值 |
|---|---:|
| `learning_rate` | 0.03 |
| `max_depth` | 2 |
| `min_child_weight` | 3 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `reg_alpha` | 0 |
| `reg_lambda` | 1 |
| 最大 `n_estimators` | 3,000 |
| `early_stopping_rounds` | 100 |
| `random_state` | 42 |

validation 早停得到 `best_iteration = 212`，部署时固定使用 213 棵树。调参记录见 [M8 受控调参报告](../esta_full_m8_tuned/m8_controlled_tuning_report.md) 和 [训练摘要](../esta_full_m8_tuned/pre_round_xgb_training_summary.json)。

## 6. 校准规则

校准方法只能在 validation 上选择。M10 以 `series_id` 分组做 5 折 OOF，对比未校准、sigmoid 和 isotonic，并按 OOF Log Loss、再按 Brier 排序。最终选择 `uncalibrated`，即保留 XGBoost 原始概率；测试集概率没有被事后修改。

该结果不是“没有做校准实验”，而是校准实验表明额外映射没有改善主要概率指标。证据见 [M10 校准报告](../esta_full_m10/m10_calibration_report.md)。

## 7. 冻结测试指标与不确定性

以下点估计来自 4,172 个测试回合。95% 区间通过对 79 个测试系列赛进行 2,000 次 bootstrap 获得，抽样单位为完整 `series_id`，随机种子为 42。

| 指标 | 点估计 | 95% 系列赛 bootstrap CI | 方向 |
|---|---:|---:|---|
| Accuracy | `0.647411` | `[0.632426, 0.662448]` | 越高越好 |
| AUC | `0.727122` | `[0.713125, 0.740907]` | 越高越好 |
| Log Loss | `0.591733` | `[0.580192, 0.603874]` | 越低越好 |
| Brier | `0.205294` | `[0.200853, 0.209890]` | 越低越好 |
| ECE10 | `0.023198` | `[0.016944, 0.039134]` | 越低越好 |

Accuracy 表示 0.5 阈值下预测胜方的比例；AUC 衡量模型把 CT 胜利回合排在 T 胜利回合之前的能力；Log Loss 对自信但错误的概率惩罚较重；Brier 是概率与 0/1 结果的均方误差；ECE10 比较 10 个概率区间中的平均预测置信度与实际胜率。

完整精度点指标见 [M9 summary](../esta_full_m9/m9_summary.json)，置信区间见 [M9 bootstrap CSV](../esta_full_m9/bootstrap_95ci.csv)。

## 8. 验收目标与对照结果

| 指标 | 当前 | 最低门槛 | 最低门槛 | 更高阶段目标 | 距离目标 |
|---|---:|---:|---|---:|---:|
| Accuracy | 0.647411 | 0.640 | 通过 | 0.660 | 低 1.259 个百分点 |
| AUC | 0.727122 | 0.700 | 通过 | 0.730 | 低 0.288 个百分点 |
| Log Loss | 0.591733 | 0.610 | 通过 | 0.580 | 高 0.011733 |
| Brier | 0.205294 | 0.210 | 通过 | 0.195 | 高 0.010294 |

四项最低门槛全部通过，但四项均未达到更高阶段目标。XGBoost 测试 AUC 比同合同逻辑回归低 `0.000107`，因此没有证明 XGBoost 在该数据上优于逻辑回归；M14 的完成含义是工程与最低性能门槛通过。

外部论文 DNN 报告了 Accuracy 0.679220、Log Loss 0.567860，本项目分别相差 -3.18 个百分点和 +0.023873。由于外部工作使用不同数据与随机行切分，这只是背景参照，不能作为公平模型排名。来源和差异计算见 [M14 外部对照](../esta_full_m14/external_benchmark_comparison.md)。

## 9. 稳健性与解释性

LAN AUC 为 `0.731693`，Online AUC 为 `0.722690`，差值 LAN - Online 为 `0.009003`，系列赛 bootstrap 95% CI 为 `[-0.018130, 0.036925]`，包含 0，不能断言两个来源存在稳定性能差异。

七张大样本地图的最低 AUC 为 `0.695993`，但最低 AUC 区间下界为 `0.658606`，说明地图级不确定性仍不可忽略。高置信错误共 90 个，人工结构化复核其中 30 个；首次击杀仅用于事后诊断，绝未作为本模型输入。详见 [M11 稳健性报告](../esta_full_m11/m11_robustness_report.md)。

解释性使用三种互补方法：XGBoost gain、测试集 permutation AUC decrease、原生 TreeSHAP。三者共同把装备经济差、双方装备价值、头盔/护甲、M4A1-S、比分差等列为重要信号；平均排名最高的特征是 `eq_value_diff_ct`。TreeSHAP 的最大重构误差为 `2.511671e-07`，全部 43 个编码特征及 Top 20 均通过泄漏审计。

这些结果属于非因果解释，只表示模型在当前数据中依赖哪些变量。例如“装备价值差重要”不等于人为增加某个记录值就会直接造成胜利。完整证据见 [M12 解释报告](../esta_full_m12/m12_explanation_report.md)。

## 10. 局限性与适用范围

- M0 尚缺正式保存的 20 回合人工快照核验记录；现有自动测试和原始帧抽查不能完全替代该记录。
- 当前 Parquet 未同时保留 `freezeTimeEndTick` 与最终 snapshot tick，后续全量重建应保存二者并输出 tick 偏移分布。
- Accuracy、AUC、Log Loss、Brier 已达到最低门槛，但未达到更高阶段目标。
- XGBoost 没有在冻结测试 AUC 上超过同合同逻辑回归。
- 部分地图的置信区间较宽，固定系列赛级随机切分也不能代表未来时间段的外推能力。
- 战队和选手身份未纳入；在没有时间切分与未知身份回退策略前加入这些字段，容易变成记忆历史队伍而不是学习可迁移规律。
- 该模型适用于 ESTA 当前数据覆盖的职业比赛分布，不应直接解释为任意版本、任意水平玩家或实时中局场景的已验证胜率。

## 11. 复现命令与关键哈希

环境为 Python `3.10.20`、XGBoost `3.2.0`、scikit-learn `1.7.2`，解释器为 `C:\Users\admin\11\envs\game\python.exe`。当前 XGBoost 模型不需要 GPU。冻结实验对应 Git commit `40cc2424e82bc8aab06e4cb4da881e12435e89c3`。

在项目根目录使用已有中间数据执行默认复现：

```powershell
.\scripts\run_pre_round_pipeline.ps1
```

从 `C:\project1\data\esta` 原始 ESTA 数据完整重建：

```powershell
.\scripts\run_pre_round_pipeline.ps1 -FullRebuild
```

入口脚本为 [run_pre_round_pipeline.ps1](../../scripts/run_pre_round_pipeline.ps1)，环境锁定文件为 [environment.yml](../../environment.yml) 和 [requirements-lock.txt](../../requirements-lock.txt)。

| 关键产物 | SHA-256 |
|---|---|
| 原始数据目录清单 | `07391e1aa728921fa84cc832bec81383d3ea3c6dfe684c1e05e7303d66a4c68e` |
| `pre_round.parquet` | `85f8154ce27a5b5d88da6b0abba29057ad9989e0ceed46c7543b089da36d5d72` |
| `pre_round_xgb.joblib` | `bf958cd64fd5a398894c286f2db77db4bed7c762054cc04bae9477b82f8d003d` |
| `pre_round_calibrator.joblib` | `8f888d90303d464b12874df968c5daad81c438c349eb5e6df9c2997be63f2507` |
| `run_pre_round_pipeline.ps1` | `5c0e2bc59fd87b800eb273e1adc7dbf544756039bf6a5991471dfd8e54eeca0b` |

哈希、环境、数据身份、切分和验收检查的机器可读来源是 [M14 实验清单](../esta_full_m14/m14_experiment_manifest.json)。

## 12. 最终判断

购买结束、交火前 XGBoost 已形成完整的“数据合同 -> 系列赛级切分 -> validation 受控调参 -> validation 校准选择 -> 冻结测试评估 -> 稳健性与解释 -> 接口与复现”证据链。它达到 M14 约定的最低验收条件，适合作为老师查收的第一个独立模型结果和后续同预测时点 LightGBM 的控制变量基线。

报告状态：**已完成并可复核**。

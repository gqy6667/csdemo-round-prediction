# 购买结束、交火前 LightGBM 独立报告

## 报告结论

本报告对应 M22–M27 已验收的“购买结束、交火前 LightGBM”模型。M27 最终状态为 **passed**，19/19 个阻断检查全部通过，模型、校准器和数据哈希在最终回放前后保持不变；最终验收只做回放与核验，LightGBM `fit` 调用次数为 0。

冻结测试结果为：Accuracy `0.650767`、AUC `0.727846`、Log Loss `0.591437`、Brier `0.205201`、ECE10 `0.018875`。五项点指标都略优于同合同 XGBoost，但五个系列赛级配对 95% 区间全部包含 0，因此不能宣称 LightGBM 显著优于 XGBoost。

正式证据见 [M27 最终验收报告](../esta_full_m27/m27_pre_round_lightgbm_final_acceptance_report.md)、[M27 summary](../esta_full_m27/m27_summary.json) 与 [M27 实验清单](../esta_full_m27/m27_experiment_manifest.json)。

## 1. 研究问题与预测时点

研究问题是：**在双方购买结束、尚未发生交火时，只使用当时已经可知的信息，预测本回合 CT 方获胜概率。**

严格预测时点为冻结时间结束后的第一个合格快照：

- 双方已完成购买；
- 仍处于首次伤害、首次交火和首次击杀之前；
- 输入只包含地图、回合序号、比分、经济、护甲与道具、武器；
- 输出 `P(CT win)` 与互补的 `P(T win)`。

LightGBM 与 XGBoost 共享这个预测时点，可以进行算法控制变量比较。首杀后模型已经使用了新的回合内信息，不能把两个预测时点的指标差解释为算法优劣。

## 2. 控制变量设计

本阶段的冻结实验政策是：**保持 M14 的数据、样本、标签、系列赛切分、36 个原始特征、43 个编码特征和五项评估口径不变，只替换模型算法，由 XGBoost 改为 LightGBM。**

因此，同预测时点 LightGBM 与 XGBoost 的差值不受以下因素混入：

- 没有新增或删除样本；
- 没有重新随机切分；
- 没有新增首杀、伤害、位置、队伍或选手身份特征；
- 没有改变标签方向或 0.5 分类阈值；
- 没有改变测试指标实现；
- 没有重新拟合 XGBoost，XGBoost 概率由冻结模型回放并按完整主键对齐。

M22 的正式策略和审计见 [M22 基线报告](../esta_full_m22/m22_pre_round_lightgbm_baseline_report.md)。

## 3. 数据来源、样本量与 70/20/10 切分

数据来自 ESTA LAN 680 个文件和 Online 878 个文件，共 1,558 场地图文件。清洗后得到 41,074 个购买结束样本、782 个系列赛和 1,558 个 `game_id`。

| split | 系列赛 | 回合样本 | 用途 |
|---|---:|---:|---|
| train | 547 | 28,522 | 拟合 LightGBM |
| validation | 156 | 8,380 | 早停、调参、校准器选择 |
| test | 79 | 4,172 | 冻结最终评估与配对比较 |
| 合计 | 782 | 41,074 | 100% |

切分单位为 `series_id`，比例按系列赛数量约为 70%/20%/10%。完整键 `series_id + game_id + round_id` 重复行 0；跨 split 的系列赛、比赛和回合均为 0；身份、标签与模型特征空值均为 0。

测试集不参与拟合、早停、调参或校准器选择。M23 的候选表和种子稳定性表明确禁止测试指标列；M27 再次确认测试集仅用于冻结后的最终评估和配对不确定性计算。

## 4. 标签与特征合同

标签仍为 `ct_win`：CT 获胜记为 1，T 获胜记为 0。模型使用与 XGBoost 完全相同的 36 个原始特征，地图独热编码后共 43 列：

| 特征组 | 内容 |
|---|---|
| context | 地图、回合序号 |
| score | 双方比分和 CT 视角比分差 |
| economy | 双方装备价值、现金及 CT 视角差值 |
| armor_utility | 护甲、头盔、拆弹器、手雷及差值 |
| weapons | AK、M4A4、M4A1-S、AWP、步枪、冲锋枪及差值 |

`series_id`、`game_id`、`round_id`、`split`、`ct_win` 及预测时点后的首伤、首杀、存活人数变化和炸弹结果均被禁止进入模型。训练集先确定地图类别和编码列，validation/test 只对齐到训练合同。

机器可读特征合同见 [M22 feature contract](../esta_full_m22/feature_contract.csv) 和 [项目 schema](../../src/csdemo/schema.py)。

## 5. 模型、训练与受控调参

正式模型为 LightGBM `LGBMClassifier`，版本 `4.6.0`，使用 CPU、确定性列式训练。M22 冻结的关键参数为：

| 参数 | 冻结值 |
|---|---:|
| `boosting_type` | `gbdt` |
| `learning_rate` | 0.03 |
| `num_leaves` | 15 |
| `min_child_samples` | 20 |
| `subsample` | 0.85 |
| `subsample_freq` | 1 |
| `colsample_bytree` | 0.85 |
| `reg_alpha` | 0.0 |
| `reg_lambda` | 1.0 |
| 最大 `n_estimators` | 3,000 |
| `early_stopping_rounds` | 100 |
| `random_state` | 42 |

模型只在 train 上拟合，用 validation `binary_logloss` 早停，得到 115 棵部署树。

M23 随后执行 validation-only 的贪心顺序调参：9 个阶段、36 个候选，每阶段只改变声明的一个参数，并要求 validation Log Loss 至少改善 `0.0001` 才接受。实际接受变更数为 0，因此最终保留 M22 参数，而不是为了得到更好测试结果继续搜索。5 个种子 42–46 的 validation Log Loss 极差为 `0.000331`，AUC 极差为 `0.000547`，均通过稳定性门槛。完整记录见 [M23 调参报告](../esta_full_m23/m23_pre_round_lightgbm_tuning_report.md)。

## 6. 校准规则

M24 仅在 validation 上，以 `series_id` 分组做 5 折 OOF，对比未校准、sigmoid、isotonic。选择规则为 OOF Log Loss 最低，其次 Brier，再按预先固定的方法顺序打破并列。

最终选择 `uncalibrated`，即保留 LightGBM 原始概率。尽管 sigmoid 在测试集上的点指标稍有改善，测试集不能推翻 validation 预先作出的选择，否则会构成测试集选择偏差。校准合同和结果见 [M24 评估报告](../esta_full_m24/m24_pre_round_lightgbm_evaluation_report.md)。

## 7. 冻结测试指标与全局不确定性

以下结果来自同一组 4,172 个测试回合。95% 区间以完整 `series_id` 为抽样单位执行 2,000 次 bootstrap，随机种子为 42。

| 指标 | 点估计 | 95% 系列赛 bootstrap CI | 方向 |
|---|---:|---:|---|
| Accuracy | `0.650767` | `[0.634450, 0.666320]` | 越高越好 |
| AUC | `0.727846` | `[0.714169, 0.741427]` | 越高越好 |
| Log Loss | `0.591437` | `[0.580635, 0.602921]` | 越低越好 |
| Brier | `0.205201` | `[0.200820, 0.209715]` | 越低越好 |
| ECE10 | `0.018875` | `[0.014073, 0.034253]` | 越低越好 |

完整精度来源是 [M24 global bootstrap CSV](../esta_full_m24/global_bootstrap_95ci.csv)。全局区间门槛也通过：AUC 下界 `0.714169` 高于阶段门槛 0.71，Log Loss 上界 `0.602921` 低于阶段门槛 0.605。

## 8. 与 XGBoost 的公平配对比较

下表中的“LightGBM 性能优势”已统一指标方向：Accuracy/AUC 用 LightGBM - XGBoost；Log Loss/Brier/ECE10 用 XGBoost - LightGBM，因此正值始终表示 LightGBM 点估计更好。区间以同一批系列赛成对重采样 2,000 次，而不是比较两组互相独立的区间。

| 指标 | LightGBM | XGBoost | LightGBM 性能优势 | 配对 95% CI | 结论 |
|---|---:|---:|---:|---:|---|
| Accuracy | 0.650767 | 0.647411 | `0.003356` | `[-0.001508, 0.008273]` | 包含 0 |
| AUC | 0.727846 | 0.727122 | `0.000724` | `[-0.001289, 0.002878]` | 包含 0 |
| Log Loss | 0.591437 | 0.591733 | `0.000296` | `[-0.000909, 0.001687]` | 包含 0 |
| Brier | 0.205201 | 0.205294 | `0.000094` | `[-0.000291, 0.000467]` | 包含 0 |
| ECE10 | 0.018875 | 0.023198 | `0.004323` | `[-0.003503, 0.013126]` | 包含 0 |

点估计方向是 5/5 略优，但统计结论是 0/5 指标达到显著优势。合理表述为“在该冻结测试集上两者性能非常接近，尚无足够证据判定 LightGBM 更优”，不能宣称 LightGBM 显著优于 XGBoost。完整配对结果见 [M24 paired bootstrap CSV](../esta_full_m24/paired_lightgbm_vs_xgboost_bootstrap.csv)。

## 9. 阶段目标

| 指标 | 当前 | 最低门槛 | 最低门槛 | 更高阶段目标 | 目标状态 |
|---|---:|---:|---|---:|---|
| Accuracy | 0.650767 | 0.640 | 通过 | 0.660 | 未通过，差 0.923 个百分点 |
| AUC | 0.727846 | 0.700 | 通过 | 0.730 | 未通过，差 0.215 个百分点 |
| Log Loss | 0.591437 | 0.610 | 通过 | 0.580 | 未通过，高 0.011437 |
| Brier | 0.205201 | 0.210 | 通过 | 0.195 | 未通过，高 0.010201 |
| ECE10 | 0.018875 | 0.050 | 通过 | 0.030 | 通过 |

五项最低门槛全部通过，更高阶段目标通过 1/5。M24 额外要求的全局 AUC 和 Log Loss 区间门槛也通过，但这不等价于战胜 XGBoost。

## 10. 稳健性与错误复核

LAN AUC 为 `0.732177`，Online AUC 为 `0.723467`，差值 LAN - Online 为 `0.008711`，95% CI 为 `[-0.017073, 0.034198]`，包含 0。七张大样本地图的最低 AUC 为 `0.693777`，其最低区间下界为 `0.661759`，地图级区间目标仍未通过。

模型产生 76 个预测方概率至少 0.80 但结果错误的高置信案例，结构化复核其中 30 个。54/76 个高置信错误中，模型看好的阵营随后输掉首杀；该首杀信息只用于事后诊断，从未作为购买结束模型特征。

分来源、地图、回合阶段、装备档位和错误案例的完整结果见 [M24 评估报告](../esta_full_m24/m24_pre_round_lightgbm_evaluation_report.md) 与 [Top 30 错误复核](../esta_full_m24/top30_error_review.md)。

## 11. 解释性

M25 对冻结的 115 棵树模型执行 Gain/Split、20 次 permutation 和 LightGBM 原生 TreeSHAP。36 个原始特征、43 个编码列、五个宏观组全部映射成功，全部特征和 Top 20 泄漏失败数均为 0；TreeSHAP 概率重构最大误差为 `7.771561e-16`。

五个宏观特征组的 permutation AUC 下降依次为：economy `0.133319`、armor/utility `0.010098`、weapons `0.003807`、score `0.002964`、context `0.002014`。多个方法都把 `eq_value_diff_ct` 排在首位；LightGBM 与 XGBoost 的 TreeSHAP Top 10 重合 9 个，说明两种树模型主要依赖相似的购买结束信号。

这些是非因果解释，只描述模型在当前样本中的依赖关系，不能解释为人为改变某个统计字段就会直接改变回合结果。完整证据见 [M25 解释报告](../esta_full_m25/m25_pre_round_lightgbm_explanation_report.md)。

## 12. 预测接口

M26 提供严格 JSON/CSV 单条预测接口：用户提供 27 个基础字段，程序自动生成 9 个 CT 视角差值，形成 36 个原始特征并严格对齐 43 个编码列。JSON 与 CSV 对同一样本的概率完全一致，非法字段、未来信息、未知地图和不一致差值会被拒绝。

接口验收见 [M26 接口报告](../esta_full_m26/m26_pre_round_lightgbm_interface_report.md)。示例概率仅验证接口，不用于重新计算或选择固定测试指标。

## 13. 外部参照与局限性

最接近预测时点的公开 DNN 报告 Accuracy 0.679220、Log Loss 0.567860；本 LightGBM 分别相差 -2.845 个百分点和 +0.023577。其他公开项目报告过约 0.88 Accuracy，但输入含回合内状态或任务定义不同，不能直接比较。来源与可比性分类见 [M26 外部对照](../esta_full_m26/external_benchmark_comparison.md)。

主要局限包括：

- 四项更高性能目标仍未达到，LightGBM 对 XGBoost 的配对优势也没有统计证据；
- 固定系列赛随机切分不能替代按比赛时间的未来外推测试；
- 部分地图的置信区间较宽，地图版本和比赛环境变化可能造成分布漂移；
- 尚未纳入战队和选手身份，加入前需要时间切分及未知身份回退设计；
- 当前特征没有位置、战术和实时交火信息，不能代替后续实时胜率模型；
- M0 的正式 20 回合人工快照记录和 snapshot tick 偏移持久化仍是数据审计遗留项。

## 14. 一键复现与关键哈希

环境为 Python `3.10.20`、LightGBM `4.6.0`、scikit-learn `1.7.2`，解释器为 `C:\Users\admin\11\envs\game\python.exe`。正式模型使用 CPU，CUDA 不是运行要求。

只核验当前冻结 M22–M27 产物：

```powershell
.\scripts\run_pre_round_lightgbm_pipeline.ps1
```

从已验收 M14 产物重建 LightGBM 阶段：

```powershell
.\scripts\run_pre_round_lightgbm_pipeline.ps1 -RebuildLightGBM
```

从 `C:\project1\data\esta` 原始数据完整重建：

```powershell
.\scripts\run_pre_round_lightgbm_pipeline.ps1 -FullRebuild
```

入口见 [run_pre_round_lightgbm_pipeline.ps1](../../scripts/run_pre_round_lightgbm_pipeline.ps1)，环境锁定见 [environment.yml](../../environment.yml) 和 [requirements-lock.txt](../../requirements-lock.txt)。M27 最终回放 4,172 条测试概率，最大概率误差 `1.110223e-16`，五项指标最大误差 0，且没有重新拟合模型。

| 关键产物 | SHA-256 |
|---|---|
| `pre_round.parquet` | `85f8154ce27a5b5d88da6b0abba29057ad9989e0ceed46c7543b089da36d5d72` |
| `pre_round_lightgbm_tuned.joblib` | `3a95983ed73cd99ae0178a16009036d48510e1ad091d33994cf296dfc69244fd` |
| `pre_round_lightgbm_calibrator.joblib` | `84e6b533e50bb9e169bb34cbbf748d6566482de716510e9f8dd733ec08147ff1` |
| `run_pre_round_lightgbm_pipeline.ps1` | `1e1559653994cd5e180a5916b87a1d63402be13684e9d75319d787eff7703089` |

M27 清单确认数据、模型和校准器在验收前后哈希完全一致。正式产物生成时记录的 Git commit 为 `b30c8c958c1c7b71b0b1a99aff939df7533b1eb9`，M27 文档与任务收尾提交为 `7bba496424e6d480c281347df7b46a65817ff6ac`。

## 15. 最终判断

购买结束、交火前 LightGBM 已完成从公平基线、validation-only 受控调参、固定测试评估、系列赛级配对比较、校准、稳健性、解释、接口到最终哈希回放的完整链路。它达到 M27 验收标准，可以与第一份 [购买结束 XGBoost 报告](01_pre_round_xgboost_report.md) 共同交付；科学结论应保持为“两种算法表现接近，LightGBM 点指标略好但未显示显著优势”。

报告状态：**已完成并可复核**。

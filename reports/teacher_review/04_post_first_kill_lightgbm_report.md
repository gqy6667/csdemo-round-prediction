# 首杀后 LightGBM 独立报告

## 报告结论

本报告对应 M28–M33 已验收的“首杀后 LightGBM”模型。M33 最终状态为
**passed**，19/19 个阻断检查全部通过；最终验收只回放冻结的 M29 模型与 M30
校准器，LightGBM `fit` 调用次数为 0，4,170 条测试概率最大回放误差为
`1.110223e-16`，五项指标最大漂移为 0。

冻结测试结果为：Accuracy `0.742926`、AUC `0.808255`、Log Loss `0.524063`、
Brier `0.176003`、ECE10 `0.014191`。与 M21 XGBoost 的同回合、同特征配对比较中，
五项 95% 系列赛 bootstrap 区间全部包含 0，因此不能宣称 LightGBM 或 XGBoost 稳定显著更优。

该结果回答的是“首杀刚发生后”的胜率问题，不是购买结束模型的直接替代；不能把两个预测时点的指标差解释为算法优劣。

正式证据见 [M33 最终验收报告](../esta_full_m33/m33_post_first_kill_lightgbm_final_acceptance_report.md)、
[M33 summary](../esta_full_m33/m33_summary.json) 和
[M33 实验清单](../esta_full_m33/m33_experiment_manifest.json)。

## 1. 研究问题与预测时点

研究问题是：**双方购买完成后最早一次有效敌对击杀刚刚发生，此时根据购买结束信息
和该首杀事件，预测 CT 最终赢得本回合的概率。**

严格预测时点与 M21 XGBoost 完全一致：

- 购买结束快照提供地图、比分、经济、护甲、道具和武器信息；
- 在当前 `series_id + game_id + round_id` 内按最小有效 `tick` 选择敌对首杀；
- 只增加 `first_kill_advantage_ct`、`first_kill_time`、
  `first_kill_headshot`、`first_kill_weapon` 四个事件字段；
- 不使用第二次击杀、后续伤害、血量、位置、下包、拆包或回合结果；
- 输出 `P(CT win)`，T 方概率为其补数。

这是首杀刚发生后的离散快照，不是持续更新的实时胜率。实时课题需要另行定义后续
事件或时间间隔快照，不能直接把本模型改名为实时模型。

## 2. 数据来源与样本形成

数据来自本地 ESTA 的 LAN 与 Online demo 解析结果。M15 已使用完整主键连接回合与
击杀，并按最小有效 tick 修复首杀事件。购买结束标准回合 41,074 条，其中 47 条没有
可识别的有效敌对首杀而被排除，最终首杀后样本为 **41,027** 条，覆盖 782 个系列赛
和 1,558 个 `game_id`。

| 数据质量项目 | 结果 |
|---|---:|
| 最终首杀后样本 | 41,027 |
| 排除的无有效首杀回合 | 47 |
| 重复完整主键 | 0 |
| 跨 split 系列赛/比赛/回合 | 0 / 0 / 0 |
| 事件字段链接不一致 | 0 |
| 标签不一致 | 0 |

修复前后有 14,357 条首杀事件发生变化，因此事件主键修复是本课题的必要数据工程，
不是模型调参。完整证据见 [M15 数据报告](../esta_full_m15/m15_first_kill_data_report.md)
和 [M33 split assignments](../esta_full_m33/split_assignments.csv)。

## 3. 70/20/10 系列赛级切分与泄漏控制

LightGBM 复用 M21 冻结的 `series_id -> split` 清单，不重新随机分配：

| split | 系列赛 | 样本 | 样本比例 | 用途 |
|---|---:|---:|---:|---|
| train | 547 | 28,489 | 69.440% | 拟合候选模型 |
| validation | 156 | 8,368 | 20.396% | 早停、调参、种子和校准选择 |
| test | 79 | 4,170 | 10.164% | 冻结最终评估与配对比较 |
| 合计 | 782 | 41,027 | 100% | |

跨 split 的 `series_id`、`game_id`、`round_id` 均为 0，完整键重复行为 0。测试集不参与拟合、早停、调参或校准器选择；M29 的候选表没有测试指标列，正式 test 只在
参数协议冻结后评估一次。M33 最终回放仍按完整三列主键连接，不依赖行顺序。

## 4. 标签与特征合同

标签为 `ct_win`：CT 赢得回合记为 1，T 赢得回合记为 0。

模型使用 40 个编码前特征和 82 个编码后特征：

- 36 个购买结束特征；
- 4 个首杀事件特征；
- 8 张训练内地图和 36 种训练内首杀武器形成独热列；
- 训练集确定类别词表，validation/test 只按冻结列重排。

M28 验证 LightGBM 与 M21 XGBoost 的 82 个编码列名称和顺序完全相同。LightGBM
Booster 只把 9 个含空格的武器列名内部规范化为下划线，列位置和含义没有改变。

身份字段、主键、split、标签、冗余首杀后存活列、第二次击杀及未来事件均被禁止进入
模型。机器可读合同见 [M28 feature contract](../esta_full_m28/feature_contract.csv) 和
[M31 leakage audit](../esta_full_m31/all_feature_leakage_audit.csv)。

## 5. M28 控制变量基线

M28 以 M21 XGBoost 为冻结对照，保持 41,027 条样本、系列赛 split、预测时点、标签、
40/82 特征和评估口径不变，**只替换模型算法**为 LightGBM。这是首杀后两种算法可以
公平比较的基础。

基线 LightGBM 使用 CPU、`learning_rate=0.03`、`num_leaves=15`、
`subsample=0.85`、`colsample_bytree=0.85`、`reg_lambda=1`、种子 42，并只用
train 拟合、validation 早停。部署 160 棵树。

| 指标 | M28 LightGBM 基线 | M21 XGBoost | 点差（LGBM-XGB） |
|---|---:|---:|---:|
| Accuracy | 0.746043 | 0.744125 | +0.001918 |
| AUC | 0.809070 | 0.809837 | -0.000767 |
| Log Loss | 0.523799 | 0.523146 | +0.000653 |
| Brier | 0.175894 | 0.175656 | +0.000238 |
| ECE10 | 0.013622 | 0.015450 | -0.001828 |

M28 的五项配对区间均包含 0，没有凭点指标宣布模型胜者。详见
[M28 基线报告](../esta_full_m28/m28_post_first_kill_lightgbm_controlled_baseline_report.md)。

## 6. M29 validation-only 受控调参

M29 以 validation Log Loss 为唯一选择指标，最小阶段改进门槛为 `0.0001`。搜索包含
**9** 个顺序阶段、**36** 个候选；每阶段只改变一个参数并保留 incumbent。最终只有
**1** 个阶段被接受：`max_depth` 从 `-1` 改为 `3`，validation Log Loss 改善
`0.000738`，正式模型早停于 211 棵树。

| 参数 | M29 冻结值 |
|---|---:|
| `n_estimators` 上限 | 3,000 |
| `learning_rate` | 0.03 |
| `num_leaves` | 15 |
| `max_depth` | 3 |
| `min_child_samples` | 20 |
| `subsample` / `subsample_freq` | 0.85 / 1 |
| `colsample_bytree` | 0.85 |
| `reg_alpha` / `reg_lambda` | 0 / 1 |
| `random_state` | 42 |

M29 调参后测试点指标反而比 M28 基线五项都略差。这不构成返回测试集重选 M28 的
理由，因为 M28/M29 的选择规则在看 test 前已经冻结；正式链按 validation 规则保留
M29。完整过程见 [M29 调参报告](../esta_full_m29/m29_post_first_kill_lightgbm_tuning_report.md)。

## 7. 校准规则

M30 只在 validation 上按 `series_id` 分组做 5 折 OOF，对比 uncalibrated、sigmoid、
isotonic，并按 OOF Log Loss、Brier 和固定方法顺序选择。最终方法为
`uncalibrated`，即 identity 校准器。

选择协议明确记录 `selection_data = validation only`，校准候选表没有测试指标列。
测试集上其他校准方法即使某个单项更好，也不能推翻 validation 选择。校准器绑定 M29
模型与同一数据哈希，见 [M30 评估报告](../esta_full_m30/m30_post_first_kill_lightgbm_evaluation_report.md)。

## 8. 冻结测试指标与全局不确定性

以下点估计来自 4,170 个测试回合。95% 区间以 79 个完整系列赛为重采样单位，执行
2,000 次 bootstrap，随机种子 42。

| 指标 | 点估计 | 95% 系列赛 bootstrap CI | 方向 |
|---|---:|---:|---|
| Accuracy | `0.742926` | `[0.731144, 0.755184]` | 越高越好 |
| AUC | `0.808255` | `[0.796194, 0.820185]` | 越高越好 |
| Log Loss | `0.524063` | `[0.510784, 0.537071]` | 越低越好 |
| Brier | `0.176003` | `[0.170342, 0.181468]` | 越低越好 |
| ECE10 | `0.014191` | `[0.012311, 0.031676]` | 越低越好 |

完整精度见 [M30 global bootstrap CSV](../esta_full_m30/global_bootstrap_95ci.csv)。
AUC 区间下界 `0.796194` 高于阶段门槛 0.79，Log Loss 区间上界 `0.537071` 低于
阶段门槛 0.54。

## 9. 与 M21 XGBoost 的公平配对比较

下表使用同一 4,170 条测试回合和完整 `series_id` 配对重采样。性能优势统一为正数
代表 LightGBM 更好；对 Log Loss/Brier 已转换方向。

| 指标 | LightGBM | XGBoost | LGBM 性能优势 | 95% 配对 CI | 包含 0 |
|---|---:|---:|---:|---:|---|
| Accuracy | 0.742926 | 0.744125 | `-0.001199` | `[-0.003426, 0.001305]` | 是 |
| AUC | 0.808255 | 0.809837 | `-0.001581` | `[-0.003239, 0.000002]` | 是 |
| Log Loss | 0.524063 | 0.523146 | `-0.000917` | `[-0.002210, 0.000407]` | 是 |
| Brier | 0.176003 | 0.175656 | `-0.000347` | `[-0.000795, 0.000084]` | 是 |
| ECE10 | 0.014191 | 0.015450 | `0.001259` | `[-0.006455, 0.005976]` | 是 |

五项区间均跨 0，显著领先指标数为 0。XGBoost 的四项点指标更好、LightGBM 的 ECE10
点指标更好，但差异都没有达到本次系列赛级统计证据下的稳定结论。公平表见
[M33 paired comparison](../esta_full_m33/paired_lightgbm_vs_xgboost_bootstrap.csv)。

## 10. 稳健性与高置信错误

M30 生成地图、来源、回合阶段、装备档位、首杀阵营、首杀时间、武器家族和爆头八类
分组。7 张主要地图最低 AUC 为 `0.782199`，其 CI 下界最低为 `0.748254`，均通过
冻结门槛。

LAN AUC 为 `0.801019`，Online AUC 为 `0.813580`；LAN - Online 差为
`-0.012561`，95% CI `[-0.036757, 0.011680]` 包含 0。模型有 86 个预测方概率至少
0.80 但结果错误的案例，结构化复核 30 个。高置信概率不等于确定结果。

## 11. 模型解释与泄漏审计

M31 对冻结模型计算 LightGBM Gain、20 次编码列与原始特征 grouped permutation、
两个时点宏观组和原生 TreeSHAP。40 个原始特征全部映射到 82 个编码列，完整特征与
TreeSHAP 前 20 泄漏失败数均为 0；TreeSHAP 概率重建最大误差为
`7.771561e-16`。

`first_kill_advantage_ct` 是三种方法的首位特征。宏观 grouped permutation 中，
购买结束组平均 AUC 下降 `0.144799`，首杀事件组下降 `0.132496`，说明冻结模型同时
依赖首杀和此前经济状态。

与 M19 XGBoost 的 82 列解释排名 Spearman 分别为 Gain `0.818113`、Permutation
`0.512173`、TreeSHAP `0.871116`、平均排名 `0.840365`；Top 10 交集为 8、8、8、9。
解释排名差异不是阻断门槛。这些重要性结果属于**非因果**解释，只描述两个模型如何
使用当前输入，不能证明某个首杀武器或经济状态产生普遍因果效应。详见
[M31 解释报告](../esta_full_m31/m31_post_first_kill_lightgbm_explanation_report.md)。

## 12. 单条预测接口

M32 的 JSON/CSV 接口要求 27 个购买结束基础字段和 4 个首杀字段，共 **31** 个用户
输入；程序派生 9 个差值，形成 40 个原始特征并严格对齐 82 个编码列。接口锁定 8 张
地图、36 种首杀武器、211 棵树和 identity 校准器，并拒绝 10 类非法输入。

JSON 与 CSV 对同一快照概率差为 0。示例基础/最终 CT 概率均为 `0.7052604307`，T
概率为 `0.2947395693`；示例只证明接口工作，不改变冻结测试指标。见
[M32 接口报告](../esta_full_m32/m32_post_first_kill_lightgbm_interface_report.md)。

## 13. 外部参照与比较边界

M33 原样保留 M30/M31 的 7 行外部对照。公开 action-value 或 DNN 工作使用不同数据、
状态表示、切分或预测时点，多数被标记为 `not_comparable`，不能据此形成算法排行榜。
完整来源、指标方向和差值见
[M33 外部对照](../esta_full_m33/external_benchmark_comparison.md)。

购买结束、交火前 XGBoost 与 LightGBM 可以在各自同合同内比较；首杀后 XGBoost 与
LightGBM 也可以在本报告的 M21/M33 同合同内比较。但是购买结束和首杀后具有不同
可用信息，不能把两个预测时点的指标差解释为算法优劣。

## 14. 局限性

- 47 个没有有效敌对首杀的回合被排除，适用总体不是所有回合。
- 首杀快照只保留四个 canonical 事件字段，未使用位置、剩余血量或空间控制。
- 系列赛级随机切分控制同场泄漏，但尚未做严格比赛时间未来外推。
- 未加入战队和选手身份；加入前需先设计时间切分、未知身份回退和阵容变更处理。
- M29 validation 选择的模型在 test 点指标上略差于 M28，这反映选择噪声，也说明不能
  用 test 返回调参。
- 高置信错误仍存在，输出概率不能解释为单回合确定判断或投注建议。
- ESTA 覆盖特定时期职业比赛，游戏版本、地图池和经济规则变化会造成分布漂移。
- 当前是单一首杀后时点，不是连续实时系统。

## 15. 一键复现与关键哈希

环境为 Python `3.10.20`、LightGBM `4.6.0`、scikit-learn `1.7.2`，解释器为
`C:\Users\admin\11\envs\game\python.exe`。正式模型使用 CPU，不要求 CUDA。

只核验冻结的 M28–M33 产物：

```powershell
.\scripts\run_post_first_kill_lightgbm_pipeline.ps1
```

从 M21 冻结工件重建 LightGBM 阶段：

```powershell
.\scripts\run_post_first_kill_lightgbm_pipeline.ps1 -RebuildLightGBM
```

从 `C:\project1\data\esta` 完整重建首杀后 XGBoost 与 LightGBM：

```powershell
.\scripts\run_post_first_kill_lightgbm_pipeline.ps1 -FullRebuild
```

入口见 [run_post_first_kill_lightgbm_pipeline.ps1](../../scripts/run_post_first_kill_lightgbm_pipeline.ps1)，
环境锁定见 [environment.yml](../../environment.yml) 和
[requirements-lock.txt](../../requirements-lock.txt)。

| 关键产物 | SHA-256 |
|---|---|
| `first_kill.parquet` | `06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492` |
| `post_first_kill_lightgbm_tuned.joblib` | `35ce17435a3716efcfdd49dd5ca13ff441e75c65512322627249e8920546a5b5` |
| `post_first_kill_lightgbm_calibrator.joblib` | `c5453403a25dfb03bbda131028fda7bdfde934840093de3e527ad2988c8043e5` |
| `run_post_first_kill_lightgbm_pipeline.ps1` | `cb8862ef1baee10bcef17ffaae806dda3ef468500c47af871fec8cef546cd2e5` |

M33 清单中 35 个输入/输出工件哈希均已重新计算并通过，模型、校准器和数据运行前后
未变化。

## 16. 最终判断

首杀后 LightGBM 已完成“同合同控制基线 -> validation-only 调参 -> 系列赛级配对
不确定性与校准 -> 解释与泄漏审计 -> JSON/CSV 接口 -> 最终哈希回放”的完整证据链。
它达到 M33 全部验收要求，可以与 M21 XGBoost 进行公平配对比较；当前证据结论是
两者点指标略有差别，但没有稳定显著胜者。

报告状态：**已完成并可复核**。

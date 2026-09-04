# 首杀后模型更换验证报告

## 1. 审计问题与结论

本报告只回答一个问题：首杀后实验是否真的把 **M21 XGBoost** 更换成了
**M28/M29 LightGBM**，还是误用了旧模型、旧预测或错误指标。

结论为：**模型已经更换成功**。这一结论不是根据指标高低作出的，而是由以下五条
相互独立的证据共同支持：

1. 冻结工件加载后的 Python 类分别是 `XGBClassifier` 和 `LGBMClassifier`；
2. 三个模型文件的 SHA256、文件大小、参数和有效树数不同；
3. XGBoost 与最终 LightGBM 的 4,170 条测试概率没有一条完全相同；
4. 两模型在 0.5 阈值下有 29 个回合给出不同分类；
5. 从冻结预测重新计算的五项指标与正式报告完全一致。

两模型结果很接近不代表没有更换。相反，本实验刻意固定数据、split、特征、标签、
预测时点和评估口径，只替换算法，接近的结果是一个允许出现的实验结果。

## 2. 审计对象

| 阶段 | 加载后的模型类型 | 内部 Booster | 字节数 | 有效树数 | SHA256 |
|---|---|---|---:|---:|---|
| M21 | `xgboost.sklearn.XGBClassifier` | `xgboost.core.Booster` | 425,536 | 409 | `ecfaaf93031e78207f81ab5ad9674020657018c0601953238ee6b68e367e8279` |
| M28 | `lightgbm.sklearn.LGBMClassifier` | `lightgbm.basic.Booster` | 298,762 | 160 | `da0259966d0e0b7f89644131b9cafe2f7d37886cf3fdad05b16582915e1013db` |
| M29/M33 | `lightgbm.sklearn.LGBMClassifier` | `lightgbm.basic.Booster` | 229,629 | 211 | `35ce17435a3716efcfdd49dd5ca13ff441e75c65512322627249e8920546a5b5` |

三份工件分别保存在 [M21 XGBoost 模型](../../models/esta_full_m17/first_kill_xgboost_tuned.joblib)、
[M28 LightGBM 基线模型](../../models/esta_full_m28/post_first_kill_lightgbm_baseline.joblib)和
[M29 LightGBM 调参模型](../../models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib)。

仅凭扩展名 `.joblib` 无法判断模型类型，因此本次审计实际加载了三个 bundle，并检查
`type(bundle["model"])`、`task`、`model_name`、参数、特征列和最佳迭代数。结果不是
把文件名改成 LightGBM，而是内部估计器已经从 XGBoost 类变成 LightGBM 类。

M33 不是第四个新训练模型。它只冻结回放 M29 LightGBM 与 M30 校准选择，验收过程的
LightGBM `fit` 调用次数为 0；真正的 LightGBM 训练发生在 M28 和 M29。这种“只回放”
是防止最终报告临时换模或重新拟合的证据。

## 3. 为什么三个模型使用相同数据

### 3.1 预测任务

三者回答同一个问题：在购买完成后，最早一次有效敌对首杀刚刚发生时，预测 CT 赢得
本回合的概率。购买结束模型是另一个预测时点，不能混入本比较。

旧 XGBoost bundle 的 legacy `task` 字符串是 `first_kill`，新 LightGBM bundle 使用
`post_first_kill`；字符串名称不同，但两者的时点定义、完整测试主键、标签、40/82 特征
和测试回合完全对齐。这里比较的是相同的首杀后任务语义，而不是声称旧元数据字符串也
完全相同。

### 3.2 数据身份

三个模型都绑定同一份数据：

- 数据文件：`data/processed/esta_full/first_kill.parquet`
- SHA256：`06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492`
- 总回合数：41,027
- 系列赛数：782
- 游戏数：1,558
- 完整主键：`series_id + game_id + round_id`

数据相同是控制变量，不是模型没有更换的迹象。如果两个算法改用不同回合或不同特征，
最终点差就会同时混入“数据变化”和“算法变化”，反而无法公平回答模型差异。

### 3.3 split 与防泄漏

| split | `series_id` 数 | 回合数 | 用途 |
|---|---:|---:|---|
| train | 547 | 28,489 | 拟合候选模型 |
| validation | 156 | 8,368 | 早停、调参和校准选择 |
| test | 79 | 4,170 | 模型冻结后的最终评估 |
| 合计 | 782 | 41,027 | — |

跨 split 的 `series_id`、`game_id`、`round_id` 重叠均为 0，完整主键重复行为 0。
test 没有参与拟合、早停、参数选择或校准器选择。正式 split 证据见
[M33 split 清单](../esta_full_m33/split_assignments.csv)和
[M33 summary](../esta_full_m33/m33_summary.json)。

### 3.4 特征合同

三者共享 40 个原始特征和 82 个编码列，列名称与顺序完全一致。特征包含地图、回合与
比分、双方经济和装备，以及在预测截止时刻已经发生的首杀信息。M28 的控制变量协议
明确规定：保留 M21 的数据、split、预测时点、标签、特征和指标，只替换算法为
LightGBM。详见 [M28 控制变量报告](../esta_full_m28/m28_post_first_kill_lightgbm_controlled_baseline_report.md)。

## 4. 模型内部合同确实不同

| 合同项 | M21 XGBoost | M28 LightGBM 基线 | M29 LightGBM 调参模型 |
|---|---:|---:|---:|
| 模型库 | XGBoost | LightGBM | LightGBM |
| 核心模型类 | `XGBClassifier` | `LGBMClassifier` | `LGBMClassifier` |
| 学习率 | 0.03 | 0.03 | 0.03 |
| 深度限制 | `max_depth=2` | `max_depth=-1` | `max_depth=3` |
| 叶子数限制 | 由深度控制 | `num_leaves=15` | `num_leaves=15` |
| 行采样 | 0.90 | 0.85 | 0.85 |
| 列采样 | 0.85 | 0.85 | 0.85 |
| L2 正则 | 1 | 1 | 1 |
| 随机种子 | 42 | 42 | 42 |
| 有效树数 | 409 | 160 | 211 |

相同的学习率、列采样和 L2 正则使比较更容易解释，但两个库的建树过程并不相同。
LightGBM 采用 leaf-wise 生长并重点用 `num_leaves` 控制复杂度；XGBoost 当前工件用
`max_depth` 控制较浅的树。因此树数也不能一对一解释成模型复杂度或性能。

M29 只使用 train 拟合、validation 早停和选择参数，共评估 36 个候选；test 没有出现在
搜索表中。最终参数与 211 棵部署树见 [M29 summary](../esta_full_m29/m29_summary.json)。

## 5. 独立指标复算

本次审计从冻结的 4,170 条测试预测重新计算 Accuracy、AUC、Log Loss、Brier 和
ECE10，得到：

| 模型 | Accuracy ↑ | AUC ↑ | Log Loss ↓ | Brier ↓ | ECE10 ↓ |
|---|---:|---:|---:|---:|---:|
| M21 XGBoost | 0.744125 | 0.809837 | 0.523146 | 0.175656 | 0.015450 |
| M28 LightGBM 基线 | 0.746043 | 0.809070 | 0.523799 | 0.175894 | 0.013622 |
| M29/M33 LightGBM 最终 | 0.742926 | 0.808255 | 0.524063 | 0.176003 | 0.014191 |

复算值与 [M28 模型比较表](../esta_full_m28/m28_model_comparison.csv)及
[M33 固定测试指标](../esta_full_m33/fixed_test_metrics.csv)一致。指标实现统一使用 0.5
分类阈值；Accuracy/AUC 越高越好，Log Loss/Brier/ECE10 越低越好。

M28 基线在 Accuracy 和 ECE10 上略好，M21 XGBoost 在 AUC、Log Loss 和 Brier 上
略好；M29 调参模型在冻结 test 上没有超过 M28 基线。这不构成错误：调参只能根据
validation 选择，有限 test 上出现小幅波动是正常的，也不能回看 test 后重新选择模型。

## 6. 逐回合预测不是复制结果

使用完整主键一对一连接 M21 XGBoost 与最终 LightGBM 的测试预测，标签不一致数为 0。
预测差异如下：

| 检查项 | 结果 |
|---|---:|
| 测试回合 | 4,170 |
| 概率 Pearson 相关系数 | 0.998928 |
| 平均绝对概率差 | 0.009181 |
| 中位绝对概率差 | 0.006750 |
| 95% 分位绝对概率差 | 0.025424 |
| 最大绝对概率差 | 0.084621 |
| 完全相同概率 | 0 |
| 0.5 阈值分类一致率 | 0.993046 |
| 0.5 阈值分类不一致回合 | 29 |

高相关说明两个模型学到了相似的主要信号；“完全相同概率为 0”和 29 个分类分歧则直接
排除了把同一列预测重复命名为两个模型的情况。逐回合来源见
[M29 测试预测](../esta_full_m29/test_predictions.csv)。

## 7. 接近的指标是否异常

最终 LightGBM 相对 XGBoost 的“性能优势”统一转换为正数代表 LightGBM 更好：

| 指标 | LightGBM 性能优势 | 95% `series_id` 配对 CI | 包含 0 |
|---|---:|---:|---|
| Accuracy | -0.001199 | [-0.003426, 0.001305] | 是 |
| AUC | -0.001581 | [-0.003239, 0.000002] | 是 |
| Log Loss | -0.000917 | [-0.002210, 0.000407] | 是 |
| Brier | -0.000347 | [-0.000795, 0.000084] | 是 |
| ECE10 | 0.001259 | [-0.006455, 0.005976] | 是 |

区间以 79 个完整测试系列赛为配对重采样单位，执行 2,000 次 bootstrap。五项区间均
包含 0，所以正确结论是“当前数据下没有稳定显著胜者”，而不是“模型没有更换”。原始
结果见 [M33 配对 bootstrap](../esta_full_m33/paired_lightgbm_vs_xgboost_bootstrap.csv)。

结果接近有四个合理原因：

1. XGBoost 和 LightGBM 都属于梯度提升决策树；
2. 两者看到完全相同的训练样本和特征；
3. 学习率、采样比例、正则化方向和有效复杂度都较保守；
4. 当前数据中的主要胜负信号可能足够明确，使不同 GBDT 实现学习到相似排序。

因此，不应为了让两个模型“看起来不同”而强行调参。算法实现不同，但在同一任务上
得到接近预测，是可接受且有研究价值的结果。

## 8. 自动验收与回放证据

- M28 控制变量基线：16/16 阻断检查通过；
- M29 validation-only 调参：16/16 阻断检查通过；
- M33 最终验收：19/19 阻断检查通过；
- M33 最终回放 LightGBM `fit` 调用次数为 0；
- 4,170 条回放概率最大绝对误差为 `1.110223e-16`；
- 五项指标与冻结来源最大漂移为 0；
- 原正式验收记录 274 项自动化测试通过；
- 2026-08-29 对 M28–M33 六个模块重新执行 60 项相关测试，结果为 `OK`。

最终验收证据见 [M33 最终报告](../esta_full_m33/m33_post_first_kill_lightgbm_final_acceptance_report.md)和
[老师版首杀后 LightGBM 报告](04_post_first_kill_lightgbm_report.md)。

## 9. 最终判定：为什么可以说“已经换成功”

“更换模型成功”与“新模型显著更好”是两个不同命题：

- **更换成功：已证明。** 模型类、工件哈希、参数、有效树数和逐回合概率均不同，且
  LightGBM 工件可以独立回放出保存结果。
- **显著更好：未证明。** 五项配对区间全部包含 0，当前不能宣称 LightGBM 或
  XGBoost 稳定胜出。

最关键的逻辑是：相同数据和相同评估口径证明比较公平；不同模型类和不同逐回合预测
证明算法已经更换；配对置信区间则限定了性能结论。三者不能互相替代。

所以本报告的最终结论是：**首杀后 LightGBM 已经真实、可复现地替换了 XGBoost 进行
训练和预测；两者结果很接近是真实实验结果，不是模型未更换或指标复用。**

## 10. 结论边界与下一步

本报告不证明 LightGBM 在其他赛事、版本或未来数据上一定与 XGBoost 相同，也不把
首杀后结果外推到购买结束或实时胜率模型。如果下一步必须判断哪种算法更适合部署，
建议预先冻结相同搜索预算、多个随机种子和 out-of-time 测试方案，再做新一轮独立比较。

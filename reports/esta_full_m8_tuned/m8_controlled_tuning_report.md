# M8 XGBoost 控制变量调参报告

完成日期：2026-08-17  
任务：购买完毕、冻结结束、交火前 CT 回合胜率

## 1. 控制规则

调参期间固定以下内容：

- 同一份 41,074 回合数据和相同的 70/20/10 `series_id` 切分。
- 同一组 43 个编码后特征和相同预处理。
- 随机种子 42、XGBoost 3.2.0、CPU 训练。
- 只用 validation 的 Log Loss 选择参数，AUC 作为辅助。
- test 在最终参数确定后只查看一次。
- 每个阶段只改变一个参数，其余参数固定为上一阶段的选择。

## 2. 外部指标调查

### 接近同一预测时点

Aakerholt 等人的课程项目在 `RoundFreezetimeEnd` 收集双方装备、装备总价值和地图，
与本项目时间点最接近。他们使用 2,500 多个 demo、70,000 多个回合训练 DNN，
21,747 个测试回合上的 Accuracy 为 0.6792、loss 为 0.5679。

来源：[Predicting the outcome of a round in CS:GO using a deep neural network](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf)

该报告没有提供 AUC，也没有说明与本项目相同的 `series_id` 分组切分，因此只能作为
数量级参考，不能把 0.6792 当成我们的直接达标线。

### 不同预测时点，不应直接比较

常见 Kaggle 数据包含 122,411 个快照，每个回合每 20 秒记录一次，并包含剩余时间、
血量、存活人数和下包状态。这些中后期信息会显著降低预测难度。

来源：[Kaggle CS:GO Round Winner Classification](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification)

使用该数据的 2023 年研究报告 XGBoost Accuracy 0.75、AUC 0.89，Random Forest
Accuracy 0.88、AUC 0.95。另一个公开项目的 Random Forest 最佳 Accuracy 为 0.8841。

来源：[Prediction of CS:GO Round Results with Machine Learning Techniques](https://dergipark.org.tr/en/pub/jista/article/1235031)，
[anantoj/csgo-round-winner-classification](https://github.com/anantoj/csgo-round-winner-classification)

这些 0.85-0.95 量级结果混合了回合中后期快照，不能与本项目严格首杀前 AUC 比较。

### 其他购买/经济模型

Rubin 的回合/比赛预测使用余额、装备价值、购买花费和比分，采用 70/30 切分。
其回合 XGBoost Accuracy 为 0.5970、F1 为 0.6210；论文报告各模型最高 AUC 约 0.71。

来源：[Predicting Round and Game Winners in CSGO](https://it.scribd.com/document/612353843/Predicting-Round-and-Game-Winners-in-CSGO)

ESTA 官方仓库也提供 XGBoost、LightGBM、Deep Sets 和 Set Transformer 基准，
说明树模型使用默认参数和 10 轮 early stopping，但 README 没有直接列出数值结果。

来源：[pnxenopoulos/esta](https://github.com/pnxenopoulos/esta)

## 3. 单变量实验结论

完整逐行结果保存在 `controlled_tuning_results.csv`。

| 阶段 | 只改变的参数 | 选择 | 主要观察 |
|---|---|---:|---|
| 1 | `max_depth` | 2 | 深度 2 的 val Log Loss 0.596435；深度 6 恶化到 0.603308 |
| 2 | `min_child_weight` | 3 | 1 和 3 基本并列，选择 3 作为轻度约束 |
| 3 | `reg_lambda` | 1 | 更强 L2 没有稳定改善 Log Loss |
| 4 | `reg_alpha` | 0 | L1 没有改善，保持关闭 |
| 5 | `subsample` | 0.85 | 当前值最佳 |
| 6 | `colsample_bytree` | 0.85 | 当前值最佳 |
| 7 | `learning_rate` + early stopping | 0.03 | 213 棵树时 val Log Loss 0.596038、AUC 0.718596 |

## 4. 参数前后对比

| 参数 | M6 | M8 tuned |
|---|---:|---:|
| `n_estimators` | 固定 500 | 上限 3000 |
| `max_depth` | 4 | 2 |
| `min_child_weight` | 1 | 3 |
| `learning_rate` | 0.03 | 0.03 |
| `subsample` | 0.85 | 0.85 |
| `colsample_bytree` | 0.85 | 0.85 |
| `reg_alpha` | 0 | 0 |
| `reg_lambda` | 1 | 1 |
| early stopping | 无 | 100 轮 |
| 实际树数 | 500 | 213 |

## 5. 正式指标

| split | Accuracy | Log Loss | AUC |
|---|---:|---:|---:|
| train | 0.6609 | 0.5864 | 0.7297 |
| validation | 0.6504 | 0.5960 | 0.7186 |
| test | 0.6474 | 0.5917 | 0.7271 |

与 M6 正式模型比较：

| Test 指标 | M6 | M8 tuned | 变化 |
|---|---:|---:|---:|
| Accuracy | 0.6462 | 0.6474 | +0.0012 |
| Log Loss | 0.5938 | 0.5917 | -0.0020 |
| AUC | 0.7220 | 0.7271 | +0.0051 |
| Train-Val AUC 差 | 0.0580 | 0.0111 | -0.0469 |

调参的主要收益不是 Accuracy，而是 AUC、Log Loss 和过拟合差距同时改善。

## 6. 随机种子稳定性

固定全部参数，仅把种子改为 42-46：

| 指标 | 均值 | 标准差 | 最大差 |
|---|---:|---:|---:|
| validation AUC | 0.718284 | 0.000209 | 0.000660 |
| test AUC | 0.727150 | 0.000276 | 0.000748 |
| validation Log Loss | 0.596161 | 0.000086 | 0.000219 |
| test Log Loss | 0.591770 | 0.000172 | 0.000495 |

所有关键指标波动均小于 0.002，达到 M8 稳定性目标。

## 7. 阶段结论

M8 的 early stopping、参数保存、最佳迭代、训练历史、过拟合控制和随机种子稳定性
均已达到验收要求。Test AUC 0.7271 尚未达到 0.73 阶段目标，但距离只差 0.0029；
下一步应完成 M7 Dummy/逻辑回归统一对照，而不是继续根据 test 反复调参。

# M29 post-first-kill LightGBM 外部模型指标对照

差值统一定义为 `我们的指标 - 外部报告指标`。Accuracy/AUC 的差值同时换算为
百分点；`performance_advantage_ours` 已按指标方向换算，正数才表示我们的模型更好。
不同数据集、年代、特征和切分会影响结果，因此这些数字不是同一排行榜。

## 当前模型

| 指标 | 当前值 |
|---|---:|
| accuracy | 0.742926 |
| auc | 0.808255 |
| log_loss | 0.524063 |
| brier_score | 0.176003 |
| ece10 | 0.014191 |

## 任务时点最接近

| 外部工作 | 模型 | 时点 | 指标 | 我们 | 外部 | 原始差值 | 方向修正后 |
|---|---|---|---|---:|---:|---:|---|
| [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | Deep neural network | RoundFreezetimeEnd | accuracy | 0.742926 | 0.679220 | +6.37 个百分点 | 我们的模型较好 0.063706 |
| [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | Deep neural network | RoundFreezetimeEnd | log_loss | 0.524063 | 0.567860 | -0.043797 | 我们的模型较好 0.043797 |

## 不可直接比较

| 外部工作 | 模型 | 时点 | 指标 | 我们 | 外部 | 原始差值 | 方向修正后 |
|---|---|---|---|---:|---:|---:|---|
| [CS:GO Round Winner Classification](https://github.com/anantoj/csgo-round-winner-classification) | Random forest | In-round snapshots | accuracy | 0.742926 | 0.884100 | -14.12 个百分点 | 仅数值差，不判断优劣 |
| [Prediction of CS:GO Round Results with ML Techniques](https://doi.org/10.38016/jista.1235031) | Random forest | In-round snapshots | accuracy | 0.742926 | 0.880000 | -13.71 个百分点 | 仅数值差，不判断优劣 |

## 可比性说明

- `aakerholt_dnn_accuracy`：Same prediction event and similar equipment features; different dataset and random row-level 80/20 validation split with no reported match-group isolation.
- `aakerholt_dnn_log_loss`：Same prediction event and similar equipment features; different dataset and random row-level 80/20 validation split with no reported match-group isolation.
- `anantoj_rf_accuracy`：Includes time remaining alive-player counts health and bomb status from in-round snapshots; this is an easier task than pre-combat prediction.
- `sinap_rf_accuracy`：Uses the Kaggle snapshot dataset and reports the best random-forest accuracy; prediction states can include information observed after combat begins.

## 使用规则

后续阶段报告继续使用同一张结构化基准表并重新生成本报告。只有在预测时点、
数据集、划分单位和评价代码都一致时，才允许把差值解释为模型本身的优劣。

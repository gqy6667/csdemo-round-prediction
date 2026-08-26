# M30 post-first-kill LightGBM 外部模型指标对照

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
| [CS156 - Round-Win Probability in CS2 via Economic Asymmetry](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | Logistic regression | Post-first-kill using equipment-value gap and first-kill side | accuracy | 0.742926 | 0.682400 | +6.05 个百分点 | 我们的模型较好 0.060526 |
| [CS156 - Round-Win Probability in CS2 via Economic Asymmetry](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | Logistic regression | Post-first-kill using equipment-value gap and first-kill side | auc | 0.808255 | 0.760000 | +4.83 个百分点 | 我们的模型较好 0.048255 |

## 部分可比

| 外部工作 | 模型 | 时点 | 指标 | 我们 | 外部 | 原始差值 | 方向修正后 |
|---|---|---|---|---:|---:|---:|---|
| [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | Deep neural network | RoundFreezetimeEnd before combat | accuracy | 0.742926 | 0.679220 | +6.37 个百分点 | 我们的模型较好 0.063706 |
| [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | Deep neural network | RoundFreezetimeEnd before combat | log_loss | 0.524063 | 0.567860 | -0.043797 | 我们的模型较好 0.043797 |

## 不可直接比较

| 外部工作 | 模型 | 时点 | 指标 | 我们 | 外部 | 原始差值 | 方向修正后 |
|---|---|---|---|---:|---:|---:|---|
| [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | Tuned XGBoost | Event-driven in-round game states | auc | 0.808255 | 0.791300 | +1.70 个百分点 | 仅数值差，不判断优劣 |
| [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | Tuned XGBoost | Event-driven in-round game states | log_loss | 0.524063 | 0.535300 | -0.011237 | 仅数值差，不判断优劣 |
| [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | Tuned XGBoost | Event-driven in-round game states | brier_score | 0.176003 | 0.184200 | -0.008197 | 仅数值差，不判断优劣 |

## 可比性说明

- `cs156_fk_logistic_accuracy`：Closest known prediction point but only 424 personal rounds with random row-level 80/20 split and no match-group isolation.
- `cs156_fk_logistic_auc`：Closest known prediction point but only 424 personal rounds with random row-level 80/20 split and no match-group isolation.
- `xenopoulos_wpa_xgb_auc`：Uses all in-round timestamps plus players remaining HP bomb and spatial features; much richer and later states than a single first-kill snapshot.
- `xenopoulos_wpa_xgb_logloss`：Uses all in-round timestamps plus players remaining HP bomb and spatial features; much richer and later states than a single first-kill snapshot.
- `xenopoulos_wpa_xgb_brier`：Uses all in-round timestamps plus players remaining HP bomb and spatial features; much richer and later states than a single first-kill snapshot.
- `aakerholt_freezetime_dnn_accuracy`：Prediction point is earlier and therefore harder; dataset and random row-level split also differ.
- `aakerholt_freezetime_dnn_logloss`：Prediction point is earlier and therefore harder; dataset and random row-level split also differ.

## 使用规则

后续阶段报告继续使用同一张结构化基准表并重新生成本报告。只有在预测时点、
数据集、划分单位和评价代码都一致时，才允许把差值解释为模型本身的优劣。

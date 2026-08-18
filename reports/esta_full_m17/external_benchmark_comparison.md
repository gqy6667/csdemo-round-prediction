# M17 首杀后调优模型外部指标对照

差值统一为“我们的指标 - 外部指标”。Accuracy/AUC 同时换算为百分点。
这些工作使用不同数据、切分和预测时点，因此表格只比较数值，不构成模型排名。

| 可比性 | 本地模型 | 外部工作 | 指标 | 我们 | 外部 | 差值 |
|---|---|---|---|---:|---:|---:|
| closest_task | `logistic_regression` | [CS156 - Round-Win Probability in CS2 via Economic Asymmetry](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | accuracy | 0.743405 | 0.682400 | +6.10 个百分点 |
| closest_task | `logistic_regression` | [CS156 - Round-Win Probability in CS2 via Economic Asymmetry](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | auc | 0.809059 | 0.760000 | +4.91 个百分点 |
| not_comparable | `xgboost_tuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | auc | 0.809837 | 0.791300 | +1.85 个百分点 |
| not_comparable | `xgboost_tuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | log_loss | 0.523146 | 0.535300 | -0.012154 |
| not_comparable | `xgboost_tuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | brier_score | 0.175656 | 0.184200 | -0.008544 |
| partial | `xgboost_tuned` | [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | accuracy | 0.744125 | 0.679220 | +6.49 个百分点 |
| partial | `xgboost_tuned` | [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | log_loss | 0.523146 | 0.567860 | -0.044714 |

`closest_task` 只表示预测时点接近；`partial` 和 `not_comparable` 不用于
判断模型优劣。完整限制记录在 benchmark registry 的 `notes` 字段。

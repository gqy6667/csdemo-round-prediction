# M20 外部模型指标差距

差值统一为“当前项目指标 - 外部报告指标”。Accuracy/AUC 同时换算成百分点。
M20 没有训练或改变概率，因此数值与 M19 相同。不同数据、切分、特征和时点
不能解释为纯算法排名。

| 可比性 | 当前模型 | 外部来源 | 指标 | 当前 | 外部 | 差值 |
|---|---|---|---|---:|---:|---:|
| closest_task | `logistic_regression` | [CS156 - Round-Win Probability in CS2 via Economic Asymmetry](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | accuracy | 0.743405 | 0.682400 | +6.10 百分点 |
| closest_task | `logistic_regression` | [CS156 - Round-Win Probability in CS2 via Economic Asymmetry](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | auc | 0.809059 | 0.760000 | +4.91 百分点 |
| not_comparable | `xgboost_tuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | auc | 0.809837 | 0.791300 | +1.85 百分点 |
| not_comparable | `xgboost_tuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | log_loss | 0.523146 | 0.535300 | -0.012154 |
| not_comparable | `xgboost_tuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | brier_score | 0.175656 | 0.184200 | -0.008544 |
| partial | `xgboost_tuned` | [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | accuracy | 0.744125 | 0.679220 | +6.49 百分点 |
| partial | `xgboost_tuned` | [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | log_loss | 0.523146 | 0.567860 | -0.044714 |

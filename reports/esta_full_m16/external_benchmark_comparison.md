# M16 首杀后外部模型指标对照

差值统一为“我们的指标 - 外部指标”。Accuracy/AUC 同时换算为百分点。
`current_model` 指明应使用本阶段逻辑回归还是 XGBoost；不同数据、切分和预测
时点仍使这些结果无法成为受控模型排行榜。

| 可比性 | 本地模型 | 外部工作 | 指标 | 我们 | 外部 | 差值 |
|---|---|---|---|---:|---:|---:|
| closest_task | `logistic_regression` | [CS156 - Round-Win Probability in CS2 via Economic Asymmetry](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | accuracy | 0.743405 | 0.682400 | +6.10 个百分点 |
| closest_task | `logistic_regression` | [CS156 - Round-Win Probability in CS2 via Economic Asymmetry](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | auc | 0.809059 | 0.760000 | +4.91 个百分点 |
| not_comparable | `xgboost_untuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | auc | 0.808896 | 0.791300 | +1.76 个百分点 |
| not_comparable | `xgboost_untuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | log_loss | 0.524753 | 0.535300 | -0.010547 |
| not_comparable | `xgboost_untuned` | [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324) | brier_score | 0.176265 | 0.184200 | -0.007935 |
| partial | `xgboost_untuned` | [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | accuracy | 0.745324 | 0.679220 | +6.61 个百分点 |
| partial | `xgboost_untuned` | [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | log_loss | 0.524753 | 0.567860 | -0.043106 |

## 解释边界

- `closest_task` 只表示预测时点最接近。424 回合个人数据和随机行切分的方差、
  难度都不同于本项目的 782 个系列赛分组切分。
- 实时 WPA 工作混合整回合时点，并使用 HP、人数、炸弹和空间信息；只报告
  数值差，不判断模型优劣。
- freeze-time DNN 的预测时点早于首杀，因此任务更难；差值不能解释为 XGBoost
  优于 DNN。

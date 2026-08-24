# M25 外部模型指标差距

M25 不改变 M24 概率，因此本表逐行复用 M24 的外部比较。差值统一为本项目减
外部报告；不同数据、预测时点和 split 不能用于证明算法排名。

| 来源 | 指标 | 本项目 | 外部报告 | 性能优势 | 可比性 |
|---|---|---:|---:|---:|---|
| [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | accuracy | 0.650767 | 0.679220 | -0.028453 | closest_task |
| [Predicting the outcome of a round in CS:GO using a DNN](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | log_loss | 0.591437 | 0.567860 | -0.023577 | closest_task |
| [CS:GO Round Winner Classification](https://github.com/anantoj/csgo-round-winner-classification) | accuracy | 0.650767 | 0.884100 | -0.233333 | not_comparable |
| [Prediction of CS:GO Round Results with ML Techniques](https://doi.org/10.38016/jista.1235031) | accuracy | 0.650767 | 0.880000 | -0.229233 | not_comparable |

`closest_task` 只表示预测时点最接近；`not_comparable` 表示输入包含回合内
信息或任务明显更容易。性能优势为正表示按该指标方向本项目更好。

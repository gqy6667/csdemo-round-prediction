# 外部模型指标对照规则

## 目的

每个模型阶段报告都回答四个问题：我们的指标是多少、外部公开指标是多少、
数值相差多少、两项实验是否可以直接比较。外部结果不是统一排行榜，数据集、
预测时点、特征和切分任一不同，都可能造成明显差异。

## 固定口径

- 原始差值固定为 `我们的指标 - 外部报告指标`。
- Accuracy、AUC 等比例指标同时报告百分点差。
- `performance_advantage_ours` 会处理指标方向：正数表示我们的模型更好。
- Log Loss、Brier 和 ECE 越低越好，不能只看原始差值的正负号。
- `closest_task` 表示预测时点接近，但不代表数据和实验设计完全一致。
- `not_comparable` 只报告数值差，禁止写成模型优劣结论。

## 当前外部基准

最接近本项目任务的是 Aakerholt 等人的 DNN：同样在
`RoundFreezetimeEnd` 提取双方装备和装备价值，报告 Accuracy `0.679220`、
Log Loss `0.567860`。本项目目前分别为 `0.647411` 和 `0.591733`，因此
Accuracy 低 `3.18` 个百分点，Log Loss 高 `0.023873`。但外部研究使用不同
HLTV 数据，并随机保留 20% 行作为验证集，没有报告按比赛分组隔离，所以差值
不能只归因于 DNN 和 XGBoost 的算法区别。

另外两项使用 Kaggle 回合中快照的工作报告 Random Forest Accuracy `0.8841`
和 `0.88`。本项目数值分别低 `23.67` 和 `23.26` 个百分点，但回合中快照包含
剩余时间、存活人数、血量和下包状态等交火后信息，任务明显更容易，不允许据此
判断本项目模型更差。

结构化来源和备注保存在：

```text
benchmarks\external_round_model_metrics.csv
```

## 每阶段报告命令

当前 M11：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.benchmark_comparison --metrics reports\esta_full_m9\m9_summary.json --benchmarks benchmarks\external_round_model_metrics.csv --report-dir reports\esta_full_m11 --stage-label M11
```

后续阶段替换 `--metrics`、`--report-dir` 和 `--stage-label`。每次必须生成：

```text
external_benchmark_comparison.csv
external_benchmark_comparison.md
```

如果阶段没有改变最终测试指标，可以沿用 M9 指标，但报告中要明确写明模型未重训。
如果增加新的外部研究，必须先核对原始来源、预测时点、数据集、切分方法和指标定义，
再向基准 CSV 增加记录。

## 来源

- Aakerholt 等，DNN 冻结时间结束预测：
  https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf
- Anantoj，Kaggle 回合中快照 Random Forest：
  https://github.com/anantoj/csgo-round-winner-classification
- Sinap，Kaggle 回合中快照模型比较：
  https://doi.org/10.38016/jista.1235031

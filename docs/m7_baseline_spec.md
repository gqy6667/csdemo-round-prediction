# M7 简单基线模型验收

## 目的

验证调优 XGBoost 是否优于常数模型和简单线性模型。预测时点保持为
`freezeTimeEndTick` 附近，即购买完毕、交火前。

## 固定实验条件

- 数据：41,074 条购买完毕快照。
- 切分：固定的 70/20/10，按 `series_id` 分组，无系列、比赛或回合穿越。
- 样本数：训练 28,522，验证 8,380，测试 4,172。
- 特征：三个模型使用相同的 43 列编码特征。
- 阈值：分类 Accuracy 统一使用 0.5。
- 指标：Accuracy、AUC、Log Loss、Brier Score、10 箱 ECE。

## 三个模型

1. 常数基线：始终输出训练集 CT 胜率 `0.542949`。
2. 逻辑回归：先标准化全部输入，再训练线性分类器。
3. XGBoost：复用 M8 调优参数及验证集 early stopping，最佳迭代为 212。

## 测试集结果

| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|
| 常数基线 | 0.524449 | 0.500000 | 0.692640 | 0.249745 | 0.018501 |
| 逻辑回归 | 0.658437 | 0.727229 | 0.592538 | 0.205508 | 0.008624 |
| 调优 XGBoost | 0.647411 | 0.727122 | 0.591733 | 0.205294 | 0.023198 |

## 验收结论

- 三个模型使用同一份切分、测试行、特征列和指标函数：通过。
- 常数模型、逻辑回归和 XGBoost 模型均已保存：通过。
- XGBoost 测试 AUC 比逻辑回归至少高 0.01：未达到。
- 实际 AUC 差值为 `-0.000107`，已按原规格明确记录，因此 M7 流程验收完成。

当前结果说明这些购买完毕特征的大部分排序信号接近线性。XGBoost 的测试
Log Loss 和 Brier 略好，但 ECE 和 Accuracy 不如逻辑回归。后续不应只继续
调树参数，而应在 M9 统一评估和 M10 概率校准中检查差异，再决定是否增加
战队、选手、地图交互或更高质量的经济与装备特征。

## 复现命令

```powershell
cd C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m7_baselines --data data\processed\esta_full\pre_round.parquet --model-dir models\esta_full_m7 --report-dir reports\esta_full_m7
```

机器可读结果位于：

```text
reports\esta_full_m7\m7_model_comparison.csv
reports\esta_full_m7\m7_summary.json
reports\esta_full_m7\m7_baseline_report.md
```

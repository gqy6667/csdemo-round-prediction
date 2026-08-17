# M9 统一评估验收

## 目的

从排序、概率误差和分类结果三个角度，对 M8 已定稿的开局前 XGBoost 做一次
固定测试集评估。M9 不重新训练、不调参数，也不根据测试集选择分类阈值。

## 评估条件

- 预测时点：购买完毕、冻结时间结束、交火前。
- 模型：`models/esta_full_m8_tuned/pre_round_xgb.joblib`。
- 测试集：4,172 个唯一回合，来自 79 个系列赛。
- 特征：训练时保存的 43 列特征，测试列按模型 bundle 对齐。
- 分类阈值：固定为 0.5。
- 置信区间：按 `series_id` 整组重采样 2,000 次，随机种子 42。

按系列赛 bootstrap 的原因是同一系列赛中的地图和回合并不完全独立。直接按
回合随机重采样通常会低估不确定性。

## 核心结果

| 指标 | 点估计 | 系列赛 bootstrap 95% CI | 最低门槛 | 阶段目标 | 结论 |
|---|---:|---:|---:|---:|---|
| AUC | 0.727122 | [0.713125, 0.740907] | >= 0.70 | >= 0.73 | 通过最低门槛 |
| Log Loss | 0.591733 | [0.580192, 0.603874] | <= 0.61 | <= 0.58 | 通过最低门槛 |
| Accuracy | 0.647411 | [0.632426, 0.662448] | >= 0.64 | >= 0.66 | 通过最低门槛 |
| Brier Score | 0.205294 | [0.200853, 0.209890] | <= 0.21 | <= 0.195 | 通过最低门槛 |
| ECE10 | 0.023198 | [0.016944, 0.039134] | M10 检查 | <= 0.03 | 点估计达到目标 |

四个 M9 核心指标全部通过最低门槛，但没有任何一项达到阶段目标。AUC 的区间
跨过 0.73，说明真实泛化水平可能在阶段目标上下；当前不能声称已经稳定达到
0.73。

## 分类结果

固定阈值 0.5 的混淆矩阵：

| 真实结果 | 预测 T | 预测 CT |
|---|---:|---:|
| T 获胜 | 879 | 1,105 |
| CT 获胜 | 366 | 1,822 |

CT recall 为 0.8327，T specificity 为 0.4430。大量概率集中在 0.5 到 0.6，
所以阈值 0.5 会更频繁地预测 CT。测试表中阈值 0.4 的 Accuracy 略高，但不能
据此修改阈值；需要改变分类阈值时，应只在验证集确定，再锁定后评估测试集。

## 初步分组结果

- LAN：1,855 回合，AUC 0.731693，Log Loss 0.587747。
- Online：2,317 回合，AUC 0.722690，Log Loss 0.594925。
- LAN/online AUC 差为 0.009003，当前没有明显来源偏差。
- 样本数至少 300 的地图中，最低 AUC 为 `de_ancient` 的 0.695993。
- `de_train` 只有 139 回合和 6 个系列赛，其 AUC 0.773839 不能直接解释为最好。

这些分组数值是 M11 的输入，不代替 M11 的分组置信区间和错误案例分析。

## 输出文件

```text
reports\esta_full_m9\test_predictions.csv
reports\esta_full_m9\m9_summary.json
reports\esta_full_m9\bootstrap_95ci.csv
reports\esta_full_m9\roc_curve.csv
reports\esta_full_m9\confusion_matrix.csv
reports\esta_full_m9\calibration_table.csv
reports\esta_full_m9\probability_distribution.csv
reports\esta_full_m9\threshold_metrics.csv
reports\esta_full_m9\metrics_by_map.csv
reports\esta_full_m9\metrics_by_source.csv
reports\esta_full_m9\roc_curve.png
reports\esta_full_m9\confusion_matrix.png
reports\esta_full_m9\probability_distribution.png
reports\esta_full_m9\reliability_curve.png
reports\esta_full_m9\m9_evaluation_report.md
```

## 复现命令

```powershell
cd C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m9_evaluation --data data\processed\esta_full\pre_round.parquet --model models\esta_full_m8_tuned\pre_round_xgb.joblib --report-dir reports\esta_full_m9 --bootstrap-samples 2000 --seed 42
```

## 验收结论

M9 流程完成：预测明细、点估计、系列赛级 95% 置信区间、ROC、混淆矩阵、
概率分布、可靠性表和初步分组指标均已固化。下一阶段是 M10 概率校准，只能
使用验证集拟合校准器，测试集继续只用于最终比较。

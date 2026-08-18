# M12 模型解释与泄漏检查验收

## 目的

解释固定开局前 XGBoost 为什么输出某个 CT 胜率，并检查模型是否依赖 ID、首杀、
伤害、下包或回合结果等不允许在购买结束时使用的信息。M12 只解释 M8 保存的模型
和 M9 固定测试集，不重新训练、调参或改变预测概率，所以正式指标保持不变。

## 三种解释方法

### Gain

统计特征被用于树分裂时带来的平均损失下降。本项目只统计 early stopping 选择的
前 213 棵部署树，不统计模型文件中额外保留的 100 棵树。Gain 适合观察树喜欢在哪些
变量上分裂，但不能单独证明该变量对测试集最重要。

### Permutation Importance

在 4,172 条固定测试样本中，每次随机打乱一个编码后的特征，重复 20 次，记录测试
AUC 平均下降。下降越大，说明模型越依赖该列。原始值和差值特征彼此相关，例如
`ct_eq_value`、`t_eq_value` 与 `eq_value_diff_ct`，所以重要性会在相关列之间分摊；
这不是因果实验。

### TreeSHAP

使用 XGBoost 原生 TreeSHAP，不需要额外安装 `shap`。每个 SHAP 值表示该特征把
该回合的 CT 获胜 log-odds 向上或向下推动多少：正数推向 CT，负数推向 T。
SHAP 值不是概率百分点，必须和基准 log-odds 相加后再经过 sigmoid 才得到概率。

## 解释完整性

| 检查 | 结果 |
|---|---:|
| 模型文件中的树 | 313 |
| 部署实际使用的树 | 213 |
| 固定测试回合 | 4,172 |
| 编码后特征 | 43 |
| SHAP 重建概率最大绝对误差 | 0.0000002512 |
| Permutation 重复次数 | 20 |

如果直接对 313 棵树计算 SHAP，解释概率会与正式预测相差约 0.011；当前实现显式
限制到 `best_iteration + 1`，测试已锁定这个行为。

## 全局结果

| 特征 | Gain 排名 | Permutation 排名 | SHAP 排名 | Permutation AUC 下降 | Mean abs SHAP |
|---|---:|---:|---:|---:|---:|
| `eq_value_diff_ct` | 2 | 1 | 1 | 0.076410 | 0.425467 |
| `ct_eq_value` | 7 | 2 | 2 | 0.011400 | 0.146753 |
| `helmet_diff_ct` | 3 | 4 | 4 | 0.002921 | 0.069526 |
| `t_eq_value` | 11 | 3 | 3 | 0.004556 | 0.095839 |
| `grenade_diff_ct` | 1 | 12 | 5 | 0.000392 | 0.046907 |
| `ct_m4a1_s` | 12 | 6 | 6 | 0.001906 | 0.036575 |
| `score_diff_ct` | 14 | 5 | 9 | 0.002032 | 0.032513 |

主要结论：

- `eq_value_diff_ct` 是最稳定的第一信号，三种方法排名分别为 2、1、1。
- 双方装备价值本身也靠前，说明模型主要学习购买阶段的经济不对称。
- `helmet_diff_ct`、护甲、步枪差和 M4A1-S 数量提供次级信息。
- `score_diff_ct` 的 Permutation 排名为 5，当前比分仍提供独立上下文。
- `grenade_diff_ct` 的 Gain 排名为 1，但 Permutation 只有 12，证明不能只看 Gain。
- 三种排名的 Spearman 相关系数为 Gain-Permutation `0.644`、Gain-SHAP
  `0.873`、Permutation-SHAP `0.714`，方向一致但并不完全相同。

## 泄漏检查

全部 43 个模型输入都通过开局前 schema 检查；TreeSHAP 前 20 个特征中：

- ID 特征为 0。
- 首杀、死亡、伤害和交火后人数特征为 0。
- 下包状态、回合结果和结束原因特征为 0。
- `ct_defuse_kits` 表示冻结时间结束时购买的拆弹器数量，是合法开局特征，不能因
  名字含 `defuse` 被误判为拆包事件。

地图独热列属于合法开局上下文。它们对模型有影响，但 M11 已另外按地图报告 AUC
和置信区间，不能把“地图重要”理解为某张地图必然由某一方获胜。

## 三个回合案例

### CT 高胜率且预测正确

Ancient 第 5 回合，模型输出 CT `0.982434`。T 装备价值只有 1,000，CT-T 装备
价值差为 25,050，CT 多 5 个头盔和 4 支步枪；这些特征共同把概率推向 CT。

### T 高胜率且预测正确

Inferno 第 3 回合，模型输出 CT `0.034359`，即 T 约 `0.965641`。CT 装备价值
只有 1,500，CT-T 装备价值差为 -19,800，护甲、手雷和步枪也处于劣势，主要
SHAP 贡献都推向 T。

### 高置信错误

Vertigo 第 2 回合，模型给 CT `0.977234`，但最终 T 获胜。CT 装备价值 19,950，
T 只有 1,100，差值为 18,850，CT 还有 5 个头盔，因此模型根据合法购买快照
强烈看好 CT。M11 的事后记录显示 CT 甚至拿到首杀后仍输掉回合。这说明解释回答
的是“模型为什么看好 CT”，不是“T 为什么最终翻盘”；枪法、位置、道具执行和
后续决策不在开局前特征中。

## 外部模型差值

M12 没有改变模型，所以与外部工作的差值和 M11 相同：与最接近的冻结时间 DNN
相比，Accuracy 低 `3.18` 个百分点，Log Loss 高 `0.023873`。完整可比性说明见
`reports\esta_full_m12\external_benchmark_comparison.md`。

## 验收结论

- Gain、Permutation 和 TreeSHAP 三种方法：通过。
- CT 高胜率、T 高胜率和高置信错误案例：通过。
- TreeSHAP 概率加和校验：通过。
- 前 20 和全部 43 个特征无 ID 或未来信息：通过。
- 解释结果、CSV 和图表写入正式报告：通过。
- 本阶段不根据测试解释结果调参：通过。

## 输出文件

```text
reports\esta_full_m12\gain_importance.csv
reports\esta_full_m12\permutation_importance_auc.csv
reports\esta_full_m12\shap_importance.csv
reports\esta_full_m12\importance_comparison.csv
reports\esta_full_m12\all_feature_leakage_audit.csv
reports\esta_full_m12\top20_feature_audit.csv
reports\esta_full_m12\selected_cases.csv
reports\esta_full_m12\case_explanations.csv
reports\esta_full_m12\gain_importance.png
reports\esta_full_m12\permutation_importance_auc.png
reports\esta_full_m12\shap_importance.png
reports\esta_full_m12\shap_summary.png
reports\esta_full_m12\case_explanations.png
reports\esta_full_m12\m12_summary.json
reports\esta_full_m12\m12_explanation_report.md
reports\esta_full_m12\external_benchmark_comparison.csv
reports\esta_full_m12\external_benchmark_comparison.md
```

## 复现命令

```powershell
cd C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m12_explanation --data data\processed\esta_full\pre_round.parquet --model models\esta_full_m8_tuned\pre_round_xgb.joblib --report-dir reports\esta_full_m12 --permutation-repeats 20 --seed 42 --case-features 10 --shap-plot-rows 1500
```

下一阶段是 M13：建立独立预测接口、输入校验和可直接运行的示例。

# LightGBM、XGBoost 与公开模型指标总表

## 1. 阅读规则

本报告把目前最重要的指标集中在一个文件中，但不把不同预测时点混成排行榜：

- **开局前**：购买完毕、冻结时间结束、正式交火前；
- **首杀后**：在购买快照上加入最早有效敌对击杀；
- **回合中实时状态**：可能已经包含人数、血量、炸弹和位置，通常比开局前任务容易。

只有本项目开局前逻辑回归、XGBoost、LightGBM 使用完全相同的 41,074 条样本、系列赛级
70/20/10 切分、36 个原始/43 个编码特征和 4,172 条测试行，属于严格控制变量比较。

## 2. 本项目最终指标

| 预测时点 | 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |
|---|---|---:|---:|---:|---:|---:|
| 开局前 | 逻辑回归 | 0.658437 | 0.727229 | 0.592538 | 0.205508 | 0.008624 |
| 开局前 | XGBoost 最终 M14 | 0.647411 | 0.727122 | 0.591733 | 0.205294 | 0.023198 |
| 开局前 | LightGBM 基线 M22 | 0.650767 | 0.727846 | 0.591437 | 0.205201 | 0.018875 |
| 开局前 | LightGBM 调参 M23 | 0.650767 | 0.727846 | 0.591437 | 0.205201 | 0.018875 |
| 首杀后 | 逻辑回归 M16 | 0.743405 | 0.809059 | 0.526642 | 0.176070 | 0.015017 |
| 首杀后 | XGBoost 最终 M21 | 0.744125 | 0.809837 | 0.523146 | 0.175656 | 0.015450 |

首杀后指标显著高于开局前，主要是因为模型已经知道首杀阵营、时间、武器和爆头信息，
不是单纯因为 XGBoost 调参。因此不能用 M21 对 M23 来判断 XGBoost 和 LightGBM 谁更好。

## 3. LightGBM 与最终开局前 XGBoost

差值为 `LightGBM - XGBoost`，两者是目前最公平的算法比较。

| 指标 | LightGBM | XGBoost | 原始差值 | 谁更好 |
|---|---:|---:|---:|---|
| Accuracy | 0.650767 | 0.647411 | +0.003356 | LightGBM 高 0.34 个百分点 |
| AUC | 0.727846 | 0.727122 | +0.000724 | LightGBM 略高 |
| Log Loss | 0.591437 | 0.591733 | -0.000296 | LightGBM 略低 |
| Brier | 0.205201 | 0.205294 | -0.000094 | LightGBM 略低 |
| ECE10 | 0.018875 | 0.023198 | -0.004323 | LightGBM 较低 |

五项方向均有利于 LightGBM，但除 ECE10 外差距很小，尚未做配对 bootstrap 显著性
检验。因此准确结论是“LightGBM 第一版略优”，不是“LightGBM 明显胜出”。

## 4. M23 调参结果

M23 固定 9 个阶段、36 个候选，以 validation Log Loss 为唯一选择指标；改善至少达到
`0.0001` 才接受。结果没有候选达到门槛：

| 最接近的候选 | Validation Log Loss 改善 | 是否接受 |
|---|---:|---|
| `min_child_samples=10` | 0.000025 | 否 |
| `max_depth=4` | 0.000002 | 否 |

其余阶段的当前参数就是最优候选。M23 因此保留 M22 参数，test 指标完全不变。五种子
validation Log Loss 范围为 `0.000331`，AUC 范围为 `0.000547`，稳定性门槛通过。

## 5. 公开模型指标

| 预测时点 | 公开工作 | 模型 | 指标 | 报告值 | 可比性 |
|---|---|---|---|---:|---|
| 开局前 | [Aakerholt: Predicting the outcome of a round in CS:GO](https://nikolaiaakerholt.com/posts/projects/academic/csgoroundwinningpredictor/Deep_Learning_CourseProject_Report.pdf) | DNN | Accuracy | 0.679220 | 时点最接近，数据与随机行切分不同 |
| 开局前 | 同上 | DNN | Log Loss | 0.567860 | 时点最接近，数据与随机行切分不同 |
| 首杀后 | [CS156 Round-Win Probability in CS2](https://madiyarzm.github.io/ML-CS2-Round-Forecaster/cs156_report.html) | 逻辑回归 | Accuracy | 0.682400 | 时点接近，仅 424 个个人回合 |
| 首杀后 | 同上 | 逻辑回归 | AUC | 0.760000 | 时点接近，随机行级 80/20 切分 |
| 回合中 | [Valuing Player Actions in CS:GO](https://arxiv.org/abs/2011.01324) | XGBoost | AUC | 0.791300 | 有实时人数、HP、炸弹和位置，不可直接比较 |
| 回合中 | 同上 | XGBoost | Log Loss | 0.535300 | 不可直接比较 |
| 回合中 | 同上 | XGBoost | Brier | 0.184200 | 不可直接比较 |
| 回合中 | [CS:GO Round Winner Classification](https://github.com/anantoj/csgo-round-winner-classification) | Random Forest | Accuracy | 0.884100 | 回合中 Kaggle 快照，不可直接比较 |
| 回合中 | [Prediction of CS:GO Round Results with ML Techniques](https://doi.org/10.38016/jista.1235031) | Random Forest | Accuracy | 0.880000 | 回合中 Kaggle 快照，不可直接比较 |

## 6. 与时点最接近的公开结果相差多少

### 开局前 LightGBM 对公开 DNN

| 指标 | 本项目 LightGBM | 公开 DNN | 本项目减公开值 |
|---|---:|---:|---:|
| Accuracy | 0.650767 | 0.679220 | -2.85 个百分点 |
| Log Loss | 0.591437 | 0.567860 | +0.023577 |

本项目数值较差，但使用 ESTA LAN+online、782 个系列赛分组切分；公开 DNN 使用不同
数据和随机行切分。差距不能全部归因于算法。

### 首杀后同模型族逻辑回归

| 指标 | 本项目逻辑回归 | 公开逻辑回归 | 本项目减公开值 |
|---|---:|---:|---:|
| Accuracy | 0.743405 | 0.682400 | +6.10 个百分点 |
| AUC | 0.809059 | 0.760000 | +4.91 个百分点 |

这组比较保持了逻辑回归模型族，但样本量、比赛来源和切分方式仍不同，也不是严格公平
对照。

## 7. 当前结论

1. 开局前 LightGBM 五项指标略优于开局前最终 XGBoost，但优势很小；
2. M23 的 36 个候选没有可靠改善，保留 M22 是正确实验结果；
3. 开局前逻辑回归 Accuracy 最高，说明复杂树模型尚未形成明显排序优势；
4. 首杀后最终 XGBoost 是当前项目数值最强的模型，但任务本身拥有更多信息；
5. 下一步 M24 应做配对系列赛 bootstrap、分地图/LAN-online 稳健性和校准，而不是继续
   根据 test 结果扩大参数搜索。

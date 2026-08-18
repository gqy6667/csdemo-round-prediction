# M16 首杀后固定切分基线报告

## 阶段决定

验收状态：**passed**。
可以进入 M17：**True**。
本阶段没有调参；三个正式模型使用完全相同的样本、特征、编码列和指标代码。

## 数据与特征

- M15 样本：41,027；编码后特征：82。
- train/val/test：28,489 / 8,368 / 4,170。
- 正式首杀特征：CT 优势、首杀时间、爆头、武器；五个确定性重复字段全部排除。
- ID、split 和 label 不进入模型。

## 三模型测试结果

| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|
| `constant_train_prior` | 0.524700 | 0.500000 | 0.692643 | 0.249746 | 0.018878 |
| `logistic_regression` | 0.743405 | 0.809059 | 0.526642 | 0.176070 | 0.015017 |
| `xgboost_untuned` | 0.745324 | 0.808896 | 0.524753 | 0.176265 | 0.010908 |

## 预先目标验收

| 指标 | 当前 XGBoost | 最低门槛 | 最低通过 | 阶段目标 | 目标通过 |
|---|---:|---:|---|---:|---|
| accuracy | 0.745324 | 0.680 | True | 0.700 | True |
| auc | 0.808896 | 0.750 | True | 0.780 | True |
| log_loss | 0.524753 | 0.580 | True | 0.550 | True |
| brier_score | 0.176265 | 0.200 | True | 0.185 | True |

## XGBoost 与逻辑回归

差值为 XGBoost 减逻辑回归；Log Loss/Brier/ECE 的负值代表 XGBoost 更低。

| 指标 | 原始差值 | XGBoost 是否更好 |
|---|---:|---|
| accuracy | +0.001918 | True |
| auc | -0.000163 | False |
| log_loss | -0.001889 | True |
| brier_score | +0.000194 | False |
| ece10 | -0.004108 | True |

逻辑回归和 XGBoost 的 AUC 几乎相同。树模型不能仅凭模型复杂度宣称胜出；
M17 调参应优先改善概率损失与泛化，而不是追逐极小的 AUC 波动。

## 首杀信息控制组

两个 XGBoost 使用相同参数和相同回合，唯一变量是是否加入四个首杀特征。

| profile | split | Accuracy | AUC | Log Loss | Brier |
|---|---|---:|---:|---:|---:|
| `canonical_event` | val | 0.741874 | 0.802069 | 0.529835 | 0.178334 |
| `canonical_event` | test | 0.745324 | 0.808896 | 0.524753 | 0.176265 |
| `pre_round_control` | val | 0.650454 | 0.714053 | 0.598461 | 0.208554 |
| `pre_round_control` | test | 0.649400 | 0.722889 | 0.593823 | 0.206026 |

Validation AUC 增益：**+0.088015**；测试 AUC 增益：**+0.086007**。
这项控制说明性能提高来自首杀事件信息，而不是样本集合或切分变化。

## 与外部模型相差多少

| 本地模型 | 外部工作 | 指标 | 我们 | 外部 | 差值 |
|---|---|---|---:|---:|---:|
| `logistic_regression` | CS156 - Round-Win Probability in CS2 via Economic Asymmetry | accuracy | 0.743405 | 0.682400 | +6.10 个百分点 |
| `logistic_regression` | CS156 - Round-Win Probability in CS2 via Economic Asymmetry | auc | 0.809059 | 0.760000 | +4.91 个百分点 |
| `xgboost_untuned` | Valuing Player Actions in Counter-Strike: Global Offensive | auc | 0.808896 | 0.791300 | +1.76 个百分点 |
| `xgboost_untuned` | Valuing Player Actions in Counter-Strike: Global Offensive | log_loss | 0.524753 | 0.535300 | -0.010547 |
| `xgboost_untuned` | Valuing Player Actions in Counter-Strike: Global Offensive | brier_score | 0.176265 | 0.184200 | -0.007935 |

最近的同任务公开项目样本仅 424 回合且按行随机切分；实时 WPA 工作使用
更丰富的整回合状态。上表只回答数值差，不能把差值归因于模型本身。
完整来源和 freeze-time 参考见 `external_benchmark_comparison.md`。

## 历史结果关系

旧主键/旧事件选择的首杀 XGBoost 测试 AUC 为 0.774750；当前为 0.808896，原始差值 +0.034146。
旧值无效，因此这不是受控提升，只用于说明为什么必须先完成 M15。

## 下一阶段

M17 只看 train/validation 做控制变量调参。由于逻辑回归 AUC 已与 XGBoost
相当，M17 必须同时观察 Log Loss、Brier、过拟合差距和首杀特征消融，不能只
以测试 AUC 反复选择参数。

复现命令：

```powershell
.\scripts\run_first_kill_baselines.ps1
```

# M29 首杀后 LightGBM 控制变量调参报告

## 阶段结论

验收状态：**passed**（16/16）。
可进入 M30 固定模型评估：**True**。
36 个候选和五种子稳定性实验只读取 train/validation；test 在参数、正式种子和选择规则冻结后评估一次。

## 冻结合同

- 样本：41,027；train/val/test 行数：{'train': 28489, 'val': 8368, 'test': 4170}。
- 系列赛切分：{'train': 547, 'val': 156, 'test': 79}。
- 特征：40 个原始、82 个编码列。
- 选择指标：validation Log Loss；每阶段最小接受改善 0.0001。
- 正式模型：seed 42，最佳迭代 211。

## 九阶段选择

| 阶段 | 选择值 | 改变 | Validation Log Loss | 接受改善 |
|---|---:|---|---:|---:|
| `num_leaves` | 15 | False | 0.528706 | 0.000000 |
| `max_depth` | 3 | True | 0.527968 | 0.000738 |
| `min_child_samples` | 20 | False | 0.527968 | 0.000000 |
| `reg_lambda` | 1.0 | False | 0.527968 | 0.000000 |
| `reg_alpha` | 0.0 | False | 0.527968 | 0.000000 |
| `subsample` | 0.85 | False | 0.527968 | 0.000000 |
| `colsample_bytree` | 0.85 | False | 0.527968 | 0.000000 |
| `min_split_gain` | 0.0 | False | 0.527968 | 0.000000 |
| `learning_rate` | 0.03 | False | 0.527968 | 0.000000 |

## Validation 目标

| 目标 | 当前 | 门槛 | 通过 |
|---|---:|---:|---|
| validation_log_loss_improvement | 0.000738 | 0.000500 | True |
| validation_auc | 0.803162 | 0.800863 | True |
| train_validation_auc_gap | 0.013546 | 0.030000 | True |

## 五种子稳定性

| seed | 最佳迭代 | Validation Log Loss | Validation AUC |
|---:|---:|---:|---:|
| 42 | 211 | 0.527968 | 0.803162 |
| 43 | 218 | 0.528010 | 0.803218 |
| 44 | 210 | 0.527783 | 0.803330 |
| 45 | 269 | 0.528145 | 0.802836 |
| 46 | 240 | 0.528076 | 0.802764 |

Log Loss 范围 `0.000362`，AUC 范围 `0.000565`；稳定性通过：**True**。

## 冻结后测试结果

| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|
| `xgboost_frozen` | 0.744125 | 0.809837 | 0.523146 | 0.175656 | 0.015450 |
| `lightgbm_baseline` | 0.746043 | 0.809070 | 0.523799 | 0.175894 | 0.013622 |
| `lightgbm_tuned` | 0.742926 | 0.808255 | 0.524063 | 0.176003 | 0.014191 |

## M29 与 M28 LightGBM 基线相差多少

方向修正后正值表示 M29 更好；这里只是点指标，显著性留给 M30。

| 指标 | 原始差值 | 方向修正后 | M29 更好 |
|---|---:|---:|---|
| accuracy | -0.003118 | -0.003118 | False |
| auc | -0.000814 | -0.000814 | False |
| log_loss | +0.000264 | -0.000264 | False |
| brier_score | +0.000109 | -0.000109 | False |
| ece10 | +0.000569 | -0.000569 | False |

## M29 与 M21 XGBoost 相差多少

方向修正后正值表示 M29 更好；这里只是点指标，显著性留给 M30。

| 指标 | 原始差值 | 方向修正后 | M29 更好 |
|---|---:|---:|---|
| accuracy | -0.001199 | -0.001199 | False |
| auc | -0.001581 | -0.001581 | False |
| log_loss | +0.000917 | -0.000917 | False |
| brier_score | +0.000347 | -0.000347 | False |
| ece10 | -0.001259 | +0.001259 | True |

## 最低门槛与阶段目标

| 指标 | 当前 | 最低门槛 | 最低通过 | 阶段目标 | 目标通过 |
|---|---:|---:|---|---:|---|
| accuracy | 0.742926 | 0.680 | True | 0.700 | True |
| auc | 0.808255 | 0.750 | True | 0.780 | True |
| log_loss | 0.524063 | 0.580 | True | 0.550 | True |
| brier_score | 0.176003 | 0.200 | True | 0.185 | True |
| ece10 | 0.014191 | 0.050 | True | 0.030 | True |

## 与公开结果的数值距离

| 外部工作 | 指标 | M29 | 外部 | 原始差值 |
|---|---|---:|---:|---:|
| Predicting the outcome of a round in CS:GO using a DNN | accuracy | 0.742926 | 0.679220 | +0.063706 |
| Predicting the outcome of a round in CS:GO using a DNN | log_loss | 0.524063 | 0.567860 | -0.043797 |

## 解释边界

- 这是固定顺序的 greedy search，不是全部参数组合的穷举。
- test 不参与参数选择，测试集上的正负变化都保留。
- M29 尚未做配对置信区间，不能仅凭点指标宣布模型显著更优。
- 不同数据、预测时点或随机行切分的公开结果不能作为公平模型排名。

## 下一阶段

M30 冻结本阶段模型，执行系列赛 bootstrap、与 M21 XGBoost 的配对比较、分地图和 LAN/online 稳健性、校准选择及错误分析。

复现命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_post_first_kill_lightgbm_tuning.ps1
```

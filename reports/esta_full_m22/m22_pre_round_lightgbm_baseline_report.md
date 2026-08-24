# M22 开局前 LightGBM 受控基线报告

## 阶段决定

验收状态：**passed**（13/13）。
可以进入 M23 validation-only 调参：**True**。
LightGBM 是否胜过 XGBoost 不是阻断条件；本阶段首先验证公平实验闭环。

## 固定条件

- 输入：41,074 条购买完毕、交火前快照。
- train/val/test：28,522 / 8,380 / 4,172。
- 特征：36 个原始字段、43 个训练集编码列。
- XGBoost 不重训；LightGBM 只用 train 拟合、validation Log Loss 早停。
- LightGBM `4.6.0`，CPU；最佳迭代 115。

## 测试集结果

| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|
| `logistic_regression` | 0.658437 | 0.727229 | 0.592538 | 0.205508 | 0.008624 |
| `xgboost_frozen` | 0.647411 | 0.727122 | 0.591733 | 0.205294 | 0.023198 |
| `lightgbm_baseline` | 0.650767 | 0.727846 | 0.591437 | 0.205201 | 0.018875 |

## LightGBM 与 XGBoost 相差多少

原始差值为 `LightGBM - XGBoost`。方向修正后大于 0 才代表 LightGBM 更好。

| 指标 | LightGBM | XGBoost | 原始差值 | 方向修正后 |
|---|---:|---:|---:|---:|
| accuracy | 0.650767 | 0.647411 | +0.003356 | +0.003356 |
| auc | 0.727846 | 0.727122 | +0.000724 | +0.000724 |
| log_loss | 0.591437 | 0.591733 | -0.000296 | +0.000296 |
| brier_score | 0.205201 | 0.205294 | -0.000094 | +0.000094 |
| ece10 | 0.018875 | 0.023198 | -0.004323 | +0.004323 |

## 预先门槛

| 指标 | 当前 | 最低门槛 | 最低通过 | 更高目标 | 目标通过 | 尚差 |
|---|---:|---:|---|---:|---|---:|
| accuracy | 0.650767 | 0.640 | True | 0.660 | False | 0.009233 |
| auc | 0.727846 | 0.700 | True | 0.730 | False | 0.002154 |
| log_loss | 0.591437 | 0.610 | True | 0.580 | False | 0.011437 |
| brier_score | 0.205201 | 0.210 | True | 0.195 | False | 0.010201 |
| ece10 | 0.018875 | 0.050 | True | 0.030 | True | 0.000000 |

## 与公开结果相差多少

差值为 LightGBM 减外部报告。数据和切分不同，只作数值参考。

| 外部工作 | 指标 | LightGBM | 外部 | 差值 |
|---|---|---:|---:|---:|
| Predicting the outcome of a round in CS:GO using a DNN | accuracy | 0.650767 | 0.679220 | -2.85 个百分点 |
| Predicting the outcome of a round in CS:GO using a DNN | log_loss | 0.591437 | 0.567860 | +0.023577 |

## 为什么本阶段可信

- 4,172 条冻结 XGBoost 概率回放最大误差：`2.978e-08`。
- XGBoost 五项指标回放最大误差：`0.000e+00`。
- 两个树模型使用同一 test 键、标签和编码列；M22 的 XGBoost fit 次数为 0。
- 测试集没有出现在 LightGBM `eval_set`，最佳树数由 validation 决定。

## 下一阶段

M23 保持数据、切分、特征和 test 不变，只按 validation Log Loss 逐项调整 `num_leaves`、`min_child_samples`、采样和正则化。候选阶段不得输出 test 指标；冻结最终参数后再做一次正式测试。首杀后 LightGBM 和实时胜率仍在之后。

复现命令：

```powershell
.\scripts\run_pre_round_lightgbm_baseline.ps1
```

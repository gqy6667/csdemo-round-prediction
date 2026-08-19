# M18 首杀后固定模型评估报告

## 阶段结论

阻断验收状态：**passed**；可进入 M19：**True**。
本阶段没有重训或调参 XGBoost；只回放 M17 冻结模型并做统计评估。
完整主键为 `series_id+game_id+round_id`，测试概率回放最大绝对误差为 `1.110e-16`。

## 70/20/10 与主键

总样本 41,027，系列赛 782；train/validation/test 行数为 28,489/8,368/4,170。
实际行比例为 69.44% / 20.40% / 10.16%；同一 series_id 没有跨 split。

## 整体置信区间

| 指标 | 点估计 | 95% CI | 成功次数 |
|---|---:|---:|---:|
| accuracy | 0.744125 | [0.731877, 0.756081] | 2000 |
| auc | 0.809837 | [0.797731, 0.822081] | 2000 |
| log_loss | 0.523146 | [0.509747, 0.536146] | 2000 |
| brier_score | 0.175656 | [0.170040, 0.181184] | 2000 |
| ece10 | 0.015450 | [0.012111, 0.031797] | 2000 |

AUC CI 下界最低线 0.780：0.797731，通过 `True`。
Log Loss CI 上界最低线 0.550：0.536146，通过 `True`。

## 地图与来源

| 地图 | 回合 | 系列 | AUC | AUC 95% CI | Log Loss |
|---|---:|---:|---:|---:|---:|
| de_inferno | 1015 | 36 | 0.787809 | [0.759465, 0.815621] | 0.548231 |
| de_nuke | 717 | 27 | 0.816437 | [0.785175, 0.845581] | 0.511672 |
| de_mirage | 706 | 28 | 0.804975 | [0.769598, 0.838659] | 0.527532 |
| de_overpass | 491 | 19 | 0.834305 | [0.801426, 0.863609] | 0.487657 |
| de_dust2 | 400 | 16 | 0.802708 | [0.765131, 0.840309] | 0.536629 |
| de_vertigo | 364 | 14 | 0.833756 | [0.800979, 0.867957] | 0.505070 |
| de_ancient | 338 | 13 | 0.783901 | [0.750719, 0.816182] | 0.533522 |
| de_train | 139 | 6 | 0.843284 | [0.785381, 0.890490] | 0.485547 |

| 来源 | 回合 | 系列 | AUC | AUC 95% CI | Log Loss |
|---|---:|---:|---:|---:|---:|
| online | 2316 | 42 | 0.814074 | [0.798384, 0.829878] | 0.519136 |
| lan | 1854 | 37 | 0.803798 | [0.786143, 0.822848] | 0.528156 |

LAN-online AUC 点差为 `-0.010276`，绝对差 `0.010276`，95% CI [-0.034586, 0.014829]。
主要地图最低 AUC 为 `0.783901`；最低 CI 下界为 `0.750719`。

## 校准

只根据 validation 的 5 折 GroupKFold OOF 结果选择 `uncalibrated`。

| 数据 | 方法 | Log Loss | Brier | ECE10 | AUC |
|---|---|---:|---:|---:|---:|
| validation_oof | uncalibrated | 0.527796 | 0.177468 | 0.011933 | 0.803324 |
| validation_oof | sigmoid | 0.527920 | 0.177539 | 0.008516 | 0.802901 |
| validation_oof | isotonic | 0.539496 | 0.177848 | 0.006228 | 0.800661 |
| test | uncalibrated | 0.523146 | 0.175656 | 0.015450 | 0.809837 |
| test | sigmoid | 0.523235 | 0.175705 | 0.012426 | 0.809837 |
| test | isotonic | 0.539508 | 0.175958 | 0.011581 | 0.808888 |

所选方法相对原始概率：Log Loss `+0.000000`，Brier `+0.000000`，ECE10 `+0.000000`。
概率指标无明显伤害：`True`。

## 高置信度错误

共有 90 个概率至少 0.80 的错误回合，已复核前 30 个。
错误模式表和完整案例保存在本阶段 CSV；这些模式不代表因果关系。

## 与外部模型差多少

| 可比性 | 当前模型 | 外部来源 | 指标 | 当前 | 外部 | 差值 |
|---|---|---|---|---:|---:|---:|
| closest_task | `logistic_regression` | CS156 - Round-Win Probability in CS2 via Economic Asymmetry | accuracy | 0.743405 | 0.682400 | +6.10 百分点 |
| closest_task | `logistic_regression` | CS156 - Round-Win Probability in CS2 via Economic Asymmetry | auc | 0.809059 | 0.760000 | +4.91 百分点 |
| not_comparable | `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | auc | 0.809837 | 0.791300 | +1.85 百分点 |
| not_comparable | `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | log_loss | 0.523146 | 0.535300 | -0.012154 |
| not_comparable | `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | brier_score | 0.175656 | 0.184200 | -0.008544 |
| partial | `xgboost_tuned` | Predicting the outcome of a round in CS:GO using a DNN | accuracy | 0.744125 | 0.679220 | +6.49 百分点 |
| partial | `xgboost_tuned` | Predicting the outcome of a round in CS:GO using a DNN | log_loss | 0.523146 | 0.567860 | -0.044714 |

外部结果使用不同数据、切分和预测时点，只能作为参照，不能作为同场排名。
M18 模型与 M17 相同，因此点指标差值不变；本阶段新增的是统计不确定性和分组证据。

## 下一阶段

M19 将对首杀后模型执行 gain、Permutation Importance、SHAP 和特征泄漏审计，重点解释首杀方、首杀时间、武器与购买状态怎样影响预测。

运行命令：

```powershell
.\scripts\run_first_kill_evaluation.ps1
```

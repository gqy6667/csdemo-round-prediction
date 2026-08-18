# M17 首杀后 XGBoost 控制变量调参报告

## 阶段决定

验收状态：**passed**。
可以进入 M18：**True**。
39 个候选只使用 validation Log Loss 选择；最终参数冻结后才评价 test。

## 八阶段选择

| 阶段 | 入选候选 | 是否改变 | 接受改善 | Validation Log Loss | Validation AUC | 树数 |
|---|---|---|---:|---:|---:|---:|
| tree_policy | `tree_cap_1500_es50` | True | +0.001584 | 0.528251 | 0.803135 | 200 |
| max_depth | `max_depth_2` | True | +0.000316 | 0.527935 | 0.802976 | 429 |
| min_child_weight | `min_child_weight_1` | False | +0.000000 | 0.527935 | 0.802976 | 429 |
| reg_lambda | `reg_lambda_1` | False | +0.000000 | 0.527935 | 0.802976 | 429 |
| reg_alpha | `reg_alpha_0` | False | +0.000000 | 0.527935 | 0.802976 | 429 |
| subsample | `subsample_0.9` | True | +0.000139 | 0.527796 | 0.803324 | 409 |
| colsample_bytree | `colsample_bytree_0.85` | False | +0.000000 | 0.527796 | 0.803324 | 409 |
| learning_rate | `learning_rate_0.03` | False | +0.000000 | 0.527796 | 0.803324 | 409 |

## 预先目标

| 目标 | 当前 | 门槛 | 通过 |
|---|---:|---:|---|
| validation_log_loss_improvement | 0.002038 | 0.001000 | True |
| validation_auc | 0.803324 | 0.800069 | True |
| train_validation_auc_gap | 0.012846 | 0.030000 | True |

## 最终测试结果

| 模型 | Accuracy | AUC | Log Loss | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|
| `constant_train_prior` | 0.524700 | 0.500000 | 0.692643 | 0.249746 | 0.018878 |
| `logistic_regression` | 0.743405 | 0.809059 | 0.526642 | 0.176070 | 0.015017 |
| `xgboost_untuned` | 0.745324 | 0.808896 | 0.524753 | 0.176265 | 0.010908 |
| `xgboost_tuned` | 0.744125 | 0.809837 | 0.523146 | 0.175656 | 0.015450 |

## 与 M16 相差多少

性能优势已按指标方向换算，正数表示 M17 更好。

| 对照 | Accuracy | AUC | Log Loss | Brier | ECE10 |
|---|---:|---:|---:|---:|---:|
| M17 vs M16 XGBoost | -0.001199 | +0.000941 | +0.001607 | +0.000609 | -0.004541 |
| M17 vs Logistic | +0.000719 | +0.000778 | +0.003496 | +0.000414 | -0.000433 |

## 随机种子稳定性

Validation Log Loss 最大差：`0.000130`；AUC 最大差：`0.000318`。
种子实验没有读取 test 指标。

| Seed | Validation Log Loss | Validation AUC | 树数 |
|---:|---:|---:|---:|
| 42 | 0.527796 | 0.803324 | 409 |
| 43 | 0.527887 | 0.803356 | 303 |
| 44 | 0.527906 | 0.803038 | 450 |
| 45 | 0.527855 | 0.803304 | 361 |
| 46 | 0.527926 | 0.803204 | 373 |

## 与外部模型相差多少

| 本地模型 | 外部工作 | 指标 | 我们 | 外部 | 差值 |
|---|---|---|---:|---:|---:|
| `logistic_regression` | CS156 - Round-Win Probability in CS2 via Economic Asymmetry | accuracy | 0.743405 | 0.682400 | +6.10 个百分点 |
| `logistic_regression` | CS156 - Round-Win Probability in CS2 via Economic Asymmetry | auc | 0.809059 | 0.760000 | +4.91 个百分点 |
| `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | auc | 0.809837 | 0.791300 | +1.85 个百分点 |
| `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | log_loss | 0.523146 | 0.535300 | -0.012154 |
| `xgboost_tuned` | Valuing Player Actions in Counter-Strike: Global Offensive | brier_score | 0.175656 | 0.184200 | -0.008544 |

外部数据、特征、预测时点和切分均不同，上表不能解释为模型本身更优。
完整来源和 freeze-time 参考见 `external_benchmark_comparison.md`。

## 结论与下一阶段

部署记录建议：`M17 tuned XGBoost`。
M18 在不再调参的前提下，对冻结模型执行系列赛 bootstrap、分地图、
LAN/online 稳健性和概率校准诊断。

复现命令：

```powershell
.\scripts\run_first_kill_tuning.ps1
```

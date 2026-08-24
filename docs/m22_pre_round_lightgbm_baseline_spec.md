# M22 开局前 LightGBM 受控基线规格

## 1. 阶段目标

M22 启动 LightGBM 研究线。第一步只回答一个问题：在开局前 XGBoost 已冻结的
数据、70/20/10 系列赛切分、特征和评估合同上，仅把模型替换为 LightGBM，结果如何。

本阶段不是参数搜索，也不要求 LightGBM 必须胜过 XGBoost。验收重点是比较公平、
概率有效、结果可重复，并形成下一阶段是否值得调参的证据。

## 2. 为什么现在可以开始

XGBoost 两个预测时点已经形成闭环：

- M14 完成“购买完毕、交火前”最终验收，15/15 阻断项通过；
- M21 完成“首杀后”最终验收，17/17 阻断项通过；
- 主键已修复为 `series_id + game_id + round_id`，跨 split 系列赛、地图和回合均为 0；
- 数据、模型、校准器、测试概率、环境版本和复现入口均有指纹或机器可读记录；
- 固定五项指标为 Accuracy、AUC、Log Loss、Brier 和 ECE10；
- 测试集只用于冻结模型的阶段验收，不用于候选选择。

因此模型算法现在是可以单独改变的实验变量，不会与主键修复、样本变化或特征变化混在
一起。XGBoost 的完整结论见 `reports/xgboost_final_summary.md`。

## 3. 固定实验合同

| 项目 | M22 固定值 |
|---|---|
| 任务 | 购买完毕、冻结时间结束、正式交火前预测 CT 回合胜率 |
| 数据 | `data/processed/esta_full/pre_round.parquet` |
| 样本 | 41,074 回合 |
| 切分单位 | `series_id` |
| train/val/test | 28,522 / 8,380 / 4,172 |
| 原始特征 | `PRE_ROUND_FEATURES` 的 36 列 |
| 编码 | 类别只从 train 学习，val/test 对齐到 train 的 43 列 |
| 标签 | `ct_win`，1 表示 CT 获胜 |
| 阈值 | 0.5，仅用于 Accuracy；概率指标使用原概率 |
| 正式选择集 | validation |
| test 用途 | 固定配置训练完成后只评估一次 |
| 计算设备 | CPU；不要求 CUDA/NVCC |

ID、`split`、label、未来事件、战队名和选手身份不得进入模型。

## 4. 唯一实验变量

对照模型为 M14 冻结的 `models/esta_full_m8_tuned/pre_round_xgb.joblib`。M22 必须加载
它并在同一批 4,172 条测试行上回放，不得重新训练 XGBoost。

LightGBM 第一版参数预先固定：

| 参数 | 值 |
|---|---:|
| boosting_type | gbdt |
| n_estimators | 3000 |
| learning_rate | 0.03 |
| num_leaves | 15 |
| min_child_samples | 20 |
| subsample | 0.85 |
| subsample_freq | 1 |
| colsample_bytree | 0.85 |
| reg_alpha | 0 |
| reg_lambda | 1 |
| random_state | 42 |
| early_stopping_rounds | 100（validation Log Loss） |

这组参数是固定起点，不是声称最优。M23 才能按 validation Log Loss 做控制变量调参。

## 5. 指标与目标

M22 沿用 M14 门槛，避免模型更换时偷偷修改成功标准。

| 指标 | 最低门槛 | 更高目标 | 方向 |
|---|---:|---:|---|
| Accuracy | 0.640 | 0.660 | 越高越好 |
| AUC | 0.700 | 0.730 | 越高越好 |
| Log Loss | 0.610 | 0.580 | 越低越好 |
| Brier | 0.210 | 0.195 | 越低越好 |
| ECE10 | 0.050 | 0.030 | 越低越好 |

阶段可以通过的条件是五项最低门槛全部达到。LightGBM 是否胜过 XGBoost不是阻断项；
所有差值必须报告，负结果同样保留。

## 6. 公平比较规则

1. 两个树模型必须使用相同 test `round_id` 顺序、标签和 43 个编码列；
2. M22 不得调用 XGBoost `fit()`；
3. LightGBM 拟合只能看到 train，早停只能看到 validation；
4. test 不得出现在 LightGBM 的 `eval_set`，也不得用于参数或树数选择；
5. 每项内部差值统一记录为“LightGBM - XGBoost”；
6. 对越低越好的指标，另记 `performance_advantage_lightgbm`，正值才表示 LightGBM 更好；
7. 与公开工作的差值统一为“我们的 LightGBM - 外部报告”；不同数据和切分只作参考。

## 7. 阻断检查

M22 必须全部通过：

1. `m14_prerequisite`：M14 已通过且冻结 XGBoost 存在；
2. `data_identity`：输入文件指纹、行数和 split 数与 M14 一致；
3. `split_contract`：系列赛、游戏和回合跨 split 重叠均为 0；
4. `feature_contract`：36 个原始特征、43 个编码列，无 ID/label/split；
5. `lightgbm_environment`：运行版本与锁文件中的 `4.6.0` 一致；
6. `validation_only_training`：早停只使用 validation，test 未参与拟合；
7. `frozen_xgboost_replay`：XGBoost 测试概率与 M9 保存值一致；
8. `probability_contract`：4,172 条概率均有限且位于 [0, 1]；
9. `minimum_metrics`：LightGBM 五项最低门槛全部通过；
10. `controlled_comparison`：两模型测试键、标签和特征合同完全一致；
11. `external_report`：生成外部差值表并标记可比性；
12. `automated_tests`：完整自动化测试通过；
13. `reproduction_entrypoint`：一键脚本存在且参数固定。

## 8. 阶段产物

```text
src/csdemo/m22_pre_round_lightgbm_baseline.py  M22 训练、审计和报告
tests/test_m22_pre_round_lightgbm_baseline.py   单元与契约测试
scripts/run_pre_round_lightgbm_baseline.ps1    一键入口
models/esta_full_m22/                           本地模型（Git 忽略）
reports/esta_full_m22/                          指标、概率、合同和报告
reports/xgboost_final_summary.md                 XGBoost 研究线最终总结
```

## 9. 验收后的下一步

M22 通过后进入 M23：保持数据、切分、特征和测试集不变，只在 train/validation 上对
LightGBM 做逐项控制变量调参。M23 冻结参数后，才能再次查看正式 test 指标。之后按
XGBoost 的同一思路依次完成评估/校准、稳健性、解释、接口和最终验收，再把同样合同
迁移到首杀后 LightGBM。实时胜率仍是独立的后续数据模块。

# M14 开局前 XGBoost 最终验收报告

## 最终决定

验收状态：**passed**。
开局前 XGBoost 阶段按已确认的最低门槛完成，可以进入首杀后 XGBoost。
这不表示所有研究目标都已达到；未达项在本报告中作为后续改进保留。

## 阻塞项检查

| 检查 | 结果 |
|---|---|
| `required_artifacts` | PASS |
| `raw_source` | PASS |
| `data_identity` | PASS |
| `quality_gate` | PASS |
| `split_contract` | PASS |
| `baseline_models` | PASS |
| `minimum_metrics` | PASS |
| `generalization_gap` | PASS |
| `calibration` | PASS |
| `robustness` | PASS |
| `explanation` | PASS |
| `prediction_interface` | PASS |
| `automated_tests` | PASS |
| `environment_lock` | PASS |
| `reproduction_entrypoint` | PASS |

## 数据与切分

- 原始 ESTA：1,558 个 `.json.xz`，LAN 680、Online 878。
- 标准回合：41,074；击杀：268,640；开局前样本：41,074。
- 重复回合键：0；孤立击杀：0。
- 质量闸门：error=0，warning=0，info=47。

| split | 系列赛 | 回合 |
|---|---:|---:|
| train | 547 | 28,522 |
| val | 156 | 8,380 |
| test | 79 | 4,172 |

跨 split 系列赛、地图和回合均为 0；总系列赛 782。

## 指标验收

最低门槛用于决定阶段能否完成；阶段目标用于记录还需提高多少。

| 指标 | 当前 | 最低门槛 | 最低通过 | 阶段目标 | 目标通过 | 尚差 |
|---|---:|---:|---|---:|---|---:|
| accuracy | 0.647411 | 0.640 | True | 0.660 | False | 1.259 个百分点 |
| auc | 0.727122 | 0.700 | True | 0.730 | False | 0.288 个百分点 |
| log_loss | 0.591733 | 0.610 | True | 0.580 | False | 0.011733 |
| brier_score | 0.205294 | 0.210 | True | 0.195 | False | 0.010294 |

四项最低门槛全部通过，四项更高阶段目标均未达到。

## 与外部模型相差多少

差值为“我们的指标 - 外部报告指标”。数据和切分不同，只能作为参考。

| 外部工作 | 指标 | 我们 | 外部 | 差值 |
|---|---|---:|---:|---:|
| Predicting the outcome of a round in CS:GO using a DNN | accuracy | 0.647411 | 0.679220 | -3.18 个百分点 |
| Predicting the outcome of a round in CS:GO using a DNN | log_loss | 0.591733 | 0.567860 | +0.023873 |

## 可复现记录

- Git commit：`40cc2424e82bc8aab06e4cb4da881e12435e89c3`。
- Python：`3.10.20`，解释器：`C:\Users\admin\11\envs\game\python.exe`。
- XGBoost：`3.2.0`；pandas：`2.3.3`；scikit-learn：`1.7.2`。
- NVCC 可用：`True`；当前模型需要 GPU：`False`。
- 自动化测试：70 项，返回码 0，耗时 1.973 秒。
- 模型 SHA-256：`bf958cd64fd5a398894c286f2db77db4bed7c762054cc04bae9477b82f8d003d`。
- 数据 SHA-256 记录在 `m14_experiment_manifest.json`。

精确核心环境在 `environment.yml` 和 `requirements-lock.txt`。默认验收命令：

```powershell
.\scripts\run_pre_round_pipeline.ps1
```

从原始 ESTA 完整重建：

```powershell
.\scripts\run_pre_round_pipeline.ps1 -FullRebuild
```

## 未达目标与剩余风险

- M0 尚缺正式的 20 回合人工快照核验记录；现有自动测试与 M4.1 原始帧核验不能完全替代人工抽查。
- M3 当前 Parquet 未保留 freezeTimeEndTick 与 snapshot tick，所以下次全量重建应新增字段并输出完整 tick 偏移分布。
- 四个核心指标通过最低门槛，但 Accuracy、AUC、Log Loss 和 Brier 均未达到更高阶段目标。
- XGBoost 测试 AUC 比逻辑回归低约 0.000107，未达到领先 0.01 的研究目标。
- 部分大地图的 AUC 置信区间下界仍低于 0.67。
- 当前是固定系列赛级随机切分，尚未完成按比赛时间的外推测试。
- 战队和选手身份特征仍未加入；应在时间切分设计完成后再评估。

这些项目不阻塞已约定的阶段最低验收，但必须保留在后续研究记录中。
下一阶段首先用修复后的 `game_id + round_num` 重建首杀后 XGBoost；当前历史首杀模型指标不能直接作为正式结果。

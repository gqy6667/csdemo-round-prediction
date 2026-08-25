# M27 购买结束 LightGBM 最终验收规格

## 1. 目标

M27 将 M22-M26 的购买结束、交火前 LightGBM 数据、基线、控制变量调参、固定评估、
解释和单条预测接口组合成一次最终验收。通过后关闭购买结束 LightGBM 研究线，并进入
首杀后 LightGBM 同合同控制变量对照。

本阶段默认只读取冻结产物，不训练、不调参、不删特征、不更换校准器、不改变阈值，
也不允许使用 test 选择模型。

## 2. 冻结合同

- 数据：`data/processed/esta_full/pre_round.parquet`，41,074 行；
- 切分：按系列赛固定为 train/val/test 28,522 / 8,380 / 4,172；
- 特征：36 个原始特征、43 个编码列；
- 模型：`models/esta_full_m23/pre_round_lightgbm_tuned.joblib`；
- 校准器：`models/esta_full_m24/pre_round_lightgbm_calibrator.joblib`，identity；
- 部署树数：115；
- 数据、模型和校准器 SHA-256 必须与 M24-M26 一致；
- 冻结指标：Accuracy `0.6507670182`、AUC `0.7278463079`、Log Loss
  `0.5914369800`、Brier `0.2052005929`、ECE10 `0.0188745440`。

M22-M26 的状态必须依次为 passed，且每个 `ready_for_mXX` 交接均为 true。M27 不把
LightGBM 的点指标小幅领先解释为显著优势；M24 的五项系列赛级配对 bootstrap 95%
区间均包含 0，此结论必须原样保留。

## 3. 命令

聚焦测试：

```powershell
C:\Users\admin\11\envs\game\python.exe -m unittest tests.test_m27_pre_round_lightgbm_acceptance -v
```

默认最终验收：

```powershell
.\scripts\run_pre_round_lightgbm_pipeline.ps1
```

从 M14 冻结产物重建 LightGBM 阶段：

```powershell
.\scripts\run_pre_round_lightgbm_pipeline.ps1 -RebuildLightGBM
```

从原始 ESTA 重建购买结束 XGBoost 和 LightGBM：

```powershell
.\scripts\run_pre_round_lightgbm_pipeline.ps1 -FullRebuild
```

完整测试与编译：

```powershell
C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests -v
C:\Users\admin\11\envs\game\python.exe -m compileall src tests
```

## 4. 项目结构

```text
src/csdemo/m27_pre_round_lightgbm_acceptance.py
tests/test_m27_pre_round_lightgbm_acceptance.py
scripts/run_pre_round_lightgbm_pipeline.ps1
reports/esta_full_m27/
docs/m27_pre_round_lightgbm_final_acceptance_spec.md
```

正式输出至少包括 `m27_summary.json`、`m27_checks.csv`、
`m27_experiment_manifest.json`、`runtime_environment.json`、`split_assignments.csv`、
`replayed_test_predictions.csv`、`fixed_test_metrics.csv`、
`paired_lightgbm_vs_xgboost_bootstrap.csv`、测试日志、编译日志和中文最终报告。

## 5. 测试策略

1. 单元测试 M22-M26 阶段链和交接状态；
2. 测试完整主键对齐、概率回放和严格 `1e-12` 容差；
3. 测试数据、模型、校准器、特征数和 115 棵树合同；
4. 测试五项指标完全冻结，配对表恰有五项且区间解释一致；
5. 测试解释、接口、外部比较、环境锁和复现脚本证据；
6. 使用真实 4,172 条 test 回放概率，但 M27 内 LightGBM `fit()` 调用必须为 0；
7. 最后运行完整 unittest 和 compileall。

## 6. 阻断验收

以下 19 项全部通过：

```text
m14_prerequisite, stage_chain, required_artifacts, data_identity,
split_contract, model_contract, calibrator_contract, prediction_replay,
fixed_metrics, paired_uncertainty, robustness_calibration, explanation,
prediction_interface, external_comparison, environment_lock,
automated_tests, source_compile, reproduction_entrypoint, artifact_manifest
```

关键条件：

- 跨 split 的 series/game/round 为 0，重复完整主键为 0；
- 4,172 条概率最大回放误差不超过 `1e-12`；
- 五项指标最大误差不超过 `1e-12`；
- 2,000 次系列赛级配对 bootstrap 五项齐全，显著领先数量仍为 0；
- M25 泄漏失败为 0，M26 15/15 和 10/10 非法输入继续通过；
- 模型和校准器运行前后哈希不变；
- 自动化测试、源码编译、三模式复现脚本和实验清单完整。

更高阶段目标全部达到、LightGBM 显著优于 XGBoost，不是 M27 阻断条件。

## 7. 边界

始终执行：严格主键连接、系列赛级切分、validation-only 选择、哈希和环境记录。

另立阶段：首杀后 LightGBM、实时胜率、身份特征、HTTP API 和 GUI。

禁止执行：M27 训练或调参、按 test 修改参数、改变已有指标、提交 ESTA 原始数据或模型
二进制、把不同公开数据集写成算法排行榜。

## 8. 成功标准

1. 19/19 阻断项通过，`pre_round_lightgbm_complete=true`；
2. 4,172 条概率、五项指标、统计结论和 M26 接口均无漂移；
3. 默认验收及两个可选重建模式均被代码与测试覆盖；
4. 报告、清单、环境、切分、测试和源码编译证据可复核；
5. 完成提交与推送后进入 M28 首杀后 LightGBM 受控基线。

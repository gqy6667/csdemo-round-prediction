# M33 首杀后 LightGBM 最终验收规格

## 1. 目标

M33 将 M28–M32 的首杀后 LightGBM 受控基线、validation-only 调参、冻结评估、
解释审计和单条预测接口组合成一次最终验收。通过后关闭首杀后 LightGBM 研究线，
并允许生成第四份老师正式报告和老师查收总索引。

本阶段默认只读取冻结产物，不训练、不调参、不删特征、不更换校准器、不改变阈值，
也不允许使用 test 选择模型。M33 内 LightGBM `fit()` 调用必须为 0。

## 2. 冻结合同

- 数据：`data/processed/esta_full/first_kill.parquet`，41,027 行；
- 切分：系列赛级 train/val/test 28,489 / 8,368 / 4,170；
- 系列赛：547 / 156 / 79，跨 split 泄漏和重复完整主键均为 0；
- 特征：40 个原始特征、82 个编码列；
- 模型：`models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib`；
- 校准器：`models/esta_full_m30/post_first_kill_lightgbm_calibrator.joblib`；
- 部署树数：211；地图 8 张，首杀武器 36 种；
- 数据、模型和校准器 SHA-256 必须与 M29–M32 一致；
- 冻结指标：Accuracy `0.7429256595`、AUC `0.8082554462`、Log Loss
  `0.5240626574`、Brier `0.1760026226`、ECE10 `0.0141908441`。

M28–M32 必须依次为 passed 且交接状态为 true。M33 必须保留 M30 的统计结论：
相对同样本 M21 XGBoost，五项系列赛级配对 bootstrap 95% 区间全部包含 0，
不能宣称 LightGBM 或 XGBoost 稳定显著领先。

## 3. 一键复现

默认只重放最终验收：

```powershell
.\scripts\run_post_first_kill_lightgbm_pipeline.ps1
```

从已验收 M21 工件重建 M28–M32：

```powershell
.\scripts\run_post_first_kill_lightgbm_pipeline.ps1 -RebuildLightGBM
```

从原始 ESTA 开始重建首杀后 XGBoost 与 LightGBM：

```powershell
.\scripts\run_post_first_kill_lightgbm_pipeline.ps1 -FullRebuild
```

## 4. 输出

```text
src/csdemo/m33_post_first_kill_lightgbm_acceptance.py
tests/test_m33_post_first_kill_lightgbm_acceptance.py
scripts/run_post_first_kill_lightgbm_pipeline.ps1
reports/esta_full_m33/m33_summary.json
reports/esta_full_m33/m33_checks.csv
reports/esta_full_m33/m33_experiment_manifest.json
reports/esta_full_m33/runtime_environment.json
reports/esta_full_m33/split_assignments.csv
reports/esta_full_m33/replayed_test_predictions.csv
reports/esta_full_m33/fixed_test_metrics.csv
reports/esta_full_m33/paired_lightgbm_vs_xgboost_bootstrap.csv
reports/esta_full_m33/m33_post_first_kill_lightgbm_final_acceptance_report.md
```

## 5. 测试策略

1. 单元测试 M28–M32 阶段链和交接状态；
2. 使用完整主键而非行顺序回放 4,170 条测试概率，容差 `1e-12`；
3. 核对 41,027 行数据、系列赛切分、40/82 特征、211 棵树和三类哈希；
4. 核对五项冻结指标、2,000 次系列赛级配对区间和“不确定胜者”结论；
5. 核对 M30 稳健性/校准、M31 解释/泄漏和 M32 接口证据；
6. 运行全量 unittest、compileall 和三模式复现脚本合同；
7. 对实验清单列出的全部输入输出重新计算 SHA-256。

## 6. 阻断验收

以下 19 项全部通过：

```text
m21_prerequisite, stage_chain, required_artifacts, data_identity,
split_contract, model_contract, calibrator_contract, prediction_replay,
fixed_metrics, paired_uncertainty, robustness_calibration, explanation,
prediction_interface, external_comparison, environment_lock,
automated_tests, source_compile, reproduction_entrypoint, artifact_manifest
```

关键门槛：

- M21 XGBoost 已最终验收，M28–M32 五阶段交接完整；
- 数据 41,027 行，split 行数和系列赛数固定，泄漏和重复主键为 0；
- 4,170 条测试概率最大回放误差 <= `1e-12`；
- 五项指标最大误差 <= `1e-12`；
- 五项配对区间均为 2,000 次系列赛 bootstrap、全部包含 0；
- M31 完整与前 20 泄漏失败均为 0，TreeSHAP 重建 <= `1e-10`；
- M32 15/15、10/10 非法输入、JSON/CSV 概率一致；
- 模型和校准器运行前后 SHA-256 不变；
- 全量测试、编译、环境锁、复现脚本和实验清单全部通过。

LightGBM 显著优于 XGBoost、解释排名完全一致，不是 M33 阻断条件。

## 7. 边界

始终执行：系列赛级切分、完整主键、validation-only 选择、冻结 test、哈希和环境记录。

另立阶段：实时胜率数据与模型、战队/选手身份、批量服务、HTTP API 和 GUI。

禁止执行：M33 训练或调参、按 test 修改参数、改写已有指标、提交原始 ESTA 或模型
二进制、将不同数据和预测时点写成算法排行榜。

## 8. 成功标准

1. 19/19 阻断项通过；
2. `post_first_kill_lightgbm_complete=true`；
3. `ready_for_teacher_report=true`；
4. 概率、指标、统计结论、解释和接口均无漂移；
5. 默认验收和两个可选重建模式有代码与测试证据；
6. 清单、环境、切分、测试、编译和最终中文报告可独立复核。

## 9. 实际结果

M33 正式运行通过 19/19 个阻断检查、274 项自动化测试和源码编译。41,027 行数据仍
按系列赛划分为 28,489 / 8,368 / 4,170，系列赛为 547 / 156 / 79；跨 split 的
series/game/round 和重复完整主键均为 0。4,170 条测试概率最大回放误差为
`1.11e-16`，五项指标最大误差为 0，LightGBM `fit()` 调用为 0。

相对 M21 XGBoost 的五项系列赛级配对 bootstrap 均完成 2,000 次，95% 区间全部
包含 0，显著领先指标数为 0。数据、模型和校准器运行前后哈希不变，35 个实验清单
输入/输出哈希全部复核通过。

正式状态为 `passed`、`post_first_kill_lightgbm_complete=true`、
`ready_for_teacher_report=true`。下一步生成第四份独立报告和老师查收总索引。

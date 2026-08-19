# M21 首杀后 XGBoost 最终验收规格

## 1. 目标

M21 将 M15-M20 已完成的数据、基线、调参、评估、解释和预测接口组合成一次可重复的
最终验收。验收通过后，首杀后 XGBoost 任务正式完成，可以进入 LightGBM 同数据对照。

M21 不训练、不调参、不做特征选择，也不改变固定测试集概率。它只回答以下问题：

1. 当前数据、模型、校准器和阶段报告是否仍是同一组已验收产物；
2. 70/20/10 系列级切分是否没有混入相同系列、比赛或回合；
3. 固定模型能否重新产生 M18 的测试概率和指标；
4. M15-M20 的质量、稳健性、解释和接口证据是否完整；
5. 新环境能否使用一个入口重跑验收或从上游产物重建整个首杀后流水线；
6. 从 M6 到 M21 的实际进步、剩余限制和外部模型差距是否被清楚记录。

## 2. 已确认假设

1. 正式数据仍是 `data/processed/esta_full/first_kill.parquet`，共 41,027 行；
2. 正式模型仍是 `models/esta_full_m17/first_kill_xgboost_tuned.joblib`；
3. 正式校准器仍是 `models/esta_full_m18/first_kill_calibrator.joblib`，方法为 identity；
4. 正式测试集仍包含 4,170 行，使用 M14 固定的系列级 70/20/10 切分；
5. M21 默认只验证已有产物；`-RebuildFirstKill` 重建 M15-M20，`-FullRebuild` 从原始 ESTA 重建 M1-M21；
6. 完整重建入口会被代码和测试检查，但本阶段正式运行默认模式，避免无必要地重新训练；
7. 外部结果来自现有已记录基准，数据集和切分不同，不能解释为算法排行榜。

## 3. 技术栈与命令

正式 Python：

```text
C:\Users\admin\11\envs\game\python.exe
```

聚焦测试：

```powershell
C:\Users\admin\11\envs\game\python.exe -m unittest tests.test_m21_first_kill_acceptance -v
```

完整测试与编译：

```powershell
C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests -v
C:\Users\admin\11\envs\game\python.exe -m compileall src tests
```

默认最终验收：

```powershell
.\scripts\run_first_kill_pipeline.ps1
```

从 M14 产物重建首杀后任务：

```powershell
.\scripts\run_first_kill_pipeline.ps1 -RebuildFirstKill
```

从 1,558 个 ESTA 文件完整重建：

```powershell
.\scripts\run_first_kill_pipeline.ps1 -FullRebuild
```

## 4. 阻断验收项

以下 17 项全部通过才可关闭首杀后 XGBoost：

```text
required_artifacts, raw_source, stage_chain, data_identity,
split_contract, model_contract, calibrator_contract, prediction_replay,
formal_targets, robustness, explanation, prediction_interface,
external_comparison, environment_lock, automated_tests,
reproduction_entrypoint, progress_report
```

关键阈值：

- 数据 SHA-256、模型 SHA-256、校准器 SHA-256 必须与 M18-M20 一致；
- train/validation/test 行数必须为 28,489 / 8,368 / 4,170；
- 跨 split 的 `series_id`、`game_id`、`round_id` 均为 0；
- 测试概率相对 M17 保存结果的最大绝对误差不超过 `1e-12`；
- 五项测试指标相对 M18 的最大绝对误差不超过 `1e-12`；
- M19 十项正式目标必须保持 10/10 通过且 remaining 为 0；
- M20 JSON/CSV 示例必须继续得到完全相同的互补概率；
- M21 内不得调用 XGBoost `fit()`；
- 完整测试必须通过，测试数量不得少于 M20 的 131 项。

## 5. 项目结构

```text
src/csdemo/m21_first_kill_acceptance.py       最终验收与报告生成
tests/test_m21_first_kill_acceptance.py        M21 单元和集成契约测试
scripts/run_first_kill_pipeline.ps1            默认验收及可选完整重建入口
reports/esta_full_m21/                          M21 机器可读和人工报告
reports/m6_to_m21_progress_report.md            M6 到 M21 中文进度报告
docs/m21_first_kill_final_acceptance_spec.md    本规格
```

## 6. 测试策略

先写失败测试，再实现最小功能：

1. 单元测试阶段链、切分隔离、目标冻结、指标回放比较和最终决策；
2. 测试一键脚本是否覆盖 M15-M21，并区分默认、首杀后重建和完整重建；
3. 测试 M6-M21 进度表和报告必须标注可比性；
4. 正式运行使用真实本地产物重放 4,170 条测试概率；
5. 最后运行整个 `unittest` 套件和 `compileall`。

## 7. M6-M21 进度报告合同

报告至少包含：

- M6、M14、M16、M17/M21 的关键指标和变化；
- M6 到 M14 的同任务公平比较；
- M16 到 M17 的首杀后同任务公平比较；
- 购买结束到首杀后的受控时点增益，并明确它不是纯调参增益；
- 70/20/10 切分、数据质量、测试数量、接口和可复现性进展；
- 当前模型与逻辑回归及外部公开指标的差距和可比性限制；
- 尚未完成的 LightGBM、实时胜率、时间外推和身份特征工作。

## 8. 边界

始终执行：校验输入、记录哈希、固定测试集、运行测试、保留外部可比性标签。

需要另立阶段：LightGBM、实时预测、HTTP 服务、GUI、战队或选手身份特征。

禁止执行：在 M21 重新调参、用测试集选模型、修改目标阈值、提交 ESTA 原始数据或模型二进制、
把不同数据集的外部差值写成直接排名。

## 9. 成功标准

1. 17/17 阻断项通过，`first_kill_xgboost_complete=true`；
2. 测试概率、五项指标和十项目标与 M18-M20 一致；
3. 生成完整实验清单、运行环境、哈希、切分、测试日志和外部比较；
4. `run_first_kill_pipeline.ps1` 默认验收可成功执行；
5. M6-M21 进度报告内容与机器产物一致；
6. 完整测试与源码编译通过；
7. 工作成果提交并推送，原始数据和模型不进入 Git。


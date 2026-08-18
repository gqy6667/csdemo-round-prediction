# M14 开局前 XGBoost 最终验收

## 1. M14 的作用

M14 不再调参，也不重新选择特征。它回答三个问题：

1. 当前结果是否来自正确的数据、固定切分和可加载模型？
2. 隔一段时间后，能否用固定环境和一条入口命令重新检查或完整重建？
3. 开局前 XGBoost 是否达到已确认的最低门槛，可以进入首杀后课题？

最终结论是：**最低验收通过，可以进入首杀后 XGBoost；更高阶段目标仍未达到。**

## 2. 一条命令怎么用

在 VSCode 中打开项目根目录并打开 PowerShell 终端：

```powershell
cd C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
```

快速验收现有数据、模型和报告：

```powershell
.\scripts\run_pre_round_pipeline.ps1
```

它会读取现有正式产物、运行全部测试、加载模型预测并重新生成 M14 报告，不会重新
训练模型。

从本地 1,558 个 ESTA 文件完整重建：

```powershell
.\scripts\run_pre_round_pipeline.ps1 -FullRebuild
```

完整重建会依次运行数据转换、质量检查、特征分析、三类基线、XGBoost 训练、统一评估、
概率校准、稳健性、模型解释、预测接口和最终验收。它耗时更长，并会更新正式数据、
模型和阶段报告，因此只在需要验证完整复现时使用。

## 3. 环境如何复现

当前核心实验环境：

| 软件 | 版本 |
|---|---:|
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| XGBoost | 3.2.0 |
| PyArrow | 24.0.0 |
| joblib | 1.5.3 |
| matplotlib | 3.10.9 |

创建新的 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate csdemo-game
```

`requirements-lock.txt` 记录同一组精确核心版本；原来的 `requirements.txt` 继续作为
宽松的最低依赖说明。

当前 `game` 环境里的 NVCC 为 CUDA 12.8、V12.8.93，但本阶段保存的 XGBoost 模型
使用 CPU 默认配置，预测和复现不要求 NVIDIA GPU。CUDA 可用于后续明确设计的 GPU
实验，不能因为安装了 CUDA 就假定模型自动使用显卡。

## 4. 最终数据证据

| 项目 | M14 结果 |
|---|---:|
| 原始 `.json.xz` | 1,558 |
| LAN / Online | 680 / 878 |
| 系列赛 | 782 |
| 标准回合 | 41,074 |
| 击杀事件 | 268,640 |
| 开局前样本 | 41,074 |
| 首杀后候选样本 | 41,027 |
| 重复 `round_id` | 0 |
| 孤立击杀 | 0 |
| 质量 error / warning | 0 / 0 |

系列赛级固定切分：

| split | 系列赛 | 回合 |
|---|---:|---:|
| train | 547 | 28,522 |
| val | 156 | 8,380 |
| test | 79 | 4,172 |

三个集合的系列赛、地图 demo 和回合交集均为 0。M14 新增
`reports/esta_full_m14/split_assignments.csv`，为每个 `series_id` 固定保存 split、
地图数、回合数和 CT 胜率。

## 5. 指标如何判定

最低门槛决定是否完成第一课题；更高阶段目标表示后续还可以提高多少。

| 指标 | 当前 | 最低门槛 | 最低通过 | 阶段目标 | 尚差 |
|---|---:|---:|---|---:|---:|
| Accuracy | 0.647411 | 0.640 | 是 | 0.660 | 1.259 个百分点 |
| AUC | 0.727122 | 0.700 | 是 | 0.730 | 0.288 个百分点 |
| Log Loss | 0.591733 | 0.610 | 是 | 0.580 | 0.011733 |
| Brier Score | 0.205294 | 0.210 | 是 | 0.195 | 0.010294 |

四个最低门槛全部通过，四个更高阶段目标全部未达到。M7 中 XGBoost 的测试 AUC
还比逻辑回归低 `0.000107`，所以不能宣称 XGBoost 已经优于简单线性模型；当前价值
主要是建立了一套完整、可解释、可复现的树模型基线。

## 6. 与别人模型相差多少

和预测时点最接近的公开 DNN 报告相比：

- Accuracy：我们的 `0.647411`，外部 `0.679220`，低 `3.18` 个百分点。
- Log Loss：我们的 `0.591733`，外部 `0.567860`，高 `0.023873`，因此较差。

两者的数据集、年代和切分协议不同，这只是参考差距，不是受控的 XGBoost 与 DNN
排行榜。包含交火后人数、生命值、时间和炸弹状态的 mid-round 工作单独标为不可直接
比较。完整表见 `reports/esta_full_m14/external_benchmark_comparison.md`。

## 7. 15 个阻塞检查

以下检查全部通过：

```text
required_artifacts   raw_source          data_identity
quality_gate         split_contract      baseline_models
minimum_metrics      generalization_gap  calibration
robustness           explanation         prediction_interface
automated_tests      environment_lock    reproduction_entrypoint
```

自动化测试为 `70/70`。测试覆盖主键、快照、M4 规范化、差值特征、列对齐、三类基线、
调参选择、指标、校准、稳健性、SHAP、预测接口以及 M14 验收规则。

## 8. 实验清单

`m14_experiment_manifest.json` 保存：

- Git commit `40cc2424e82bc8aab06e4cb4da881e12435e89c3`。
- Python 路径、平台、核心包版本和 NVCC 输出。
- 原始 ESTA 文件数、总字节数和文件清单哈希。
- rounds、kills、训练表、模型、校准器和运行脚本的 SHA-256。
- XGBoost 参数、最佳迭代、指标目标、所有验收项和测试结果。

当前模型 SHA-256：

```text
bf958cd64fd5a398894c286f2db77db4bed7c762054cc04bae9477b82f8d003d
```

这个哈希用于确认加载的是同一个模型文件，不说明模型质量高低。

## 9. 仍然保留的证据缺口

以下项目不阻塞已约定的最低门槛，但没有被写成“已完成”：

1. M0 尚缺正式的 20 回合人工快照核验记录；现有自动测试和 M4.1 三个原始帧案例
   不能完全替代人工抽查。
2. M3 当前 Parquet 没有保存 `freezeTimeEndTick`、实际快照 tick 和二者偏移；下次从
   原始 ESTA 完整重建时应新增字段并输出全量偏移分布。
3. 当前只做固定随机种子 42 的系列赛级切分，尚未做按比赛日期的时间外推测试。
4. 部分地图 AUC 置信区间下界仍低于 0.67。
5. 战队和选手身份尚未加入，是否加入应结合时间切分和未知身份处理再决定。

## 10. 下一阶段

下一阶段进入“首杀后 XGBoost”，但必须从修复后的 41,027 条首杀后候选样本重新开始：

1. 重新确认首杀时点、标签和允许特征。
2. 审计每个首杀事件是否按 `game_id + round_num` 关联到正确地图回合。
3. 复用同一份 782 系列赛 split 清单，不能重新随机划分。
4. 先建立 Dummy/逻辑回归/XGBoost 基线，再做调参和外部对照。

旧的首杀后 AUC `0.7747` 来自主键修复前的历史关联，只能作为历史记录，不能作为
新阶段正式结果。

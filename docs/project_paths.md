# CSDemo 项目路径说明

## 项目根目录

```text
C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
```

在 VSCode 中应打开上面这个文件夹，而不是只打开某个 Python 文件。

## 外部路径

原始 ESTA 数据：

```text
C:\project1\data\esta
```

Python/Conda 环境：

```text
C:\Users\admin\11\envs\game
```

项目使用的 Python：

```text
C:\Users\admin\11\envs\game\python.exe
```

## 项目目录

```text
csdemo_round_prediction\
├─ data\
│  ├─ sample\                 手工编写的最小示例数据
│  ├─ interim\                从 ESTA 转换出的 rounds/kills 标准表
│  └─ processed\              加好特征和 train/val/test 的训练表
├─ benchmarks\                 外部公开模型指标、来源和可比性标签
├─ docs\                       项目规格、路径说明和指标说明
├─ models\                     训练后保存的模型
├─ reports\                    Accuracy、AUC、Log Loss 等实验结果
├─ src\csdemo\                 Python 源代码
├─ tests\                      自动化测试
├─ README.md                   项目首页和快速命令
└─ requirements.txt            Python 依赖
```

## 数据路径

M2 修复后的完整数据：

```text
data\interim\esta_full\rounds.parquet
data\interim\esta_full\kills.parquet
data\processed\esta_full\pre_round.parquet
data\processed\esta_full\first_kill.parquet
```

M2 修复前的历史数据和模型：

```text
data\interim\esta_full_legacy_m1
data\processed\esta_full_legacy_m1
models\esta_full_legacy_m1
reports\esta_full_legacy_m1
```

注意：`esta_full` 现在是 M2 修复后的正式数据；`*_legacy_m1` 只用于对比旧结果。

20-demo 验证样本仍保存在：

```text
data\interim\esta_sample
data\processed\esta_sample
```

## 代码路径

```text
src\csdemo\esta_to_tables.py   读取 ESTA，建立 series/game/round 主键并提取快照
src\csdemo\features.py         构造开局前和首杀后特征
src\csdemo\split.py            按 series_id 划分 train/val/test
src\csdemo\make_dataset.py     生成模型训练表
src\csdemo\train_xgb.py        训练与评估 XGBoost
src\csdemo\m6_analysis.py      M6 特征体检、重要性、分地图指标和消融
src\csdemo\metrics.py          M7-M10 共用概率指标
src\csdemo\m7_baselines.py     M7 三模型统一对照
src\csdemo\m9_evaluation.py    M9 统一评估、bootstrap 和图表
src\csdemo\calibration.py      可持久化的 Identity/Sigmoid/Isotonic 校准器
src\csdemo\m10_calibration.py  M10 验证集校准选择和测试比较
src\csdemo\m11_robustness.py   M11 分组置信区间和高置信错误分析
src\csdemo\benchmark_comparison.py  各阶段外部模型差值报告
src\csdemo\train_lgbm.py       后续 LightGBM 对比
src\csdemo\schema.py           特征列和 ID 列定义
src\csdemo\config.py           路径、随机种子和 70/20/10 比例
```

## 模型和报告路径

M2 修复前的完整基线模型：

```text
models\esta_full_legacy_m1\pre_round_xgb.joblib
reports\esta_full_legacy_m1\pre_round_xgb_metrics.csv
```

M2 修复后的 20-demo 冒烟测试模型：

```text
models\esta_sample\pre_round_xgb.joblib
reports\esta_sample\pre_round_xgb_metrics.csv
```

20-demo 模型只用于检查代码链路，样本太少，不能用它的指标评价项目效果。

M6 正式开局前模型和报告：

```text
models\esta_full_m6\pre_round_xgb.joblib
reports\esta_full_m6\pre_round_xgb_metrics.csv
reports\esta_full_m6\m6_feature_report.md
reports\esta_full_m6\feature_profile.csv
reports\esta_full_m6\feature_importance.csv
reports\esta_full_m6\ablation_metrics.csv
reports\esta_full_m6\test_metrics_by_map.csv
reports\pre_round_xgb_initial_to_current_report.md
```

M8 控制变量调优模型和报告：

```text
models\esta_full_m8_tuned\pre_round_xgb.joblib
reports\esta_full_m8_tuned\pre_round_xgb_metrics.csv
reports\esta_full_m8_tuned\pre_round_xgb_training_summary.json
reports\esta_full_m8_tuned\pre_round_xgb_training_history.csv
reports\esta_full_m8_tuned\controlled_tuning_results.csv
reports\esta_full_m8_tuned\seed_stability.csv
reports\esta_full_m8_tuned\m8_controlled_tuning_report.md
```

M9 统一评估报告：

```text
reports\esta_full_m9\test_predictions.csv
reports\esta_full_m9\bootstrap_95ci.csv
reports\esta_full_m9\m9_summary.json
reports\esta_full_m9\m9_evaluation_report.md
reports\esta_full_m9\roc_curve.png
reports\esta_full_m9\confusion_matrix.png
reports\esta_full_m9\probability_distribution.png
reports\esta_full_m9\reliability_curve.png
```

M10 校准模型和报告：

```text
models\esta_full_m10\pre_round_calibrator.joblib
reports\esta_full_m10\validation_oof_comparison.csv
reports\esta_full_m10\test_calibration_comparison.csv
reports\esta_full_m10\m10_summary.json
reports\esta_full_m10\m10_calibration_report.md
reports\esta_full_m10\reliability_comparison.png
```

M11 稳健性和错误分析：

```text
reports\esta_full_m11\metrics_by_map_with_ci.csv
reports\esta_full_m11\metrics_by_source_with_ci.csv
reports\esta_full_m11\metrics_by_round_stage_with_ci.csv
reports\esta_full_m11\metrics_by_equipment_band_with_ci.csv
reports\esta_full_m11\reviewed_top30_errors.csv
reports\esta_full_m11\top30_error_review.md
reports\esta_full_m11\map_auc_with_ci.png
reports\esta_full_m11\error_pattern_counts.png
reports\esta_full_m11\m11_summary.json
reports\esta_full_m11\m11_robustness_report.md
reports\esta_full_m11\external_benchmark_comparison.csv
reports\esta_full_m11\external_benchmark_comparison.md
```

## 文档路径

```text
docs\pre_round_xgb_module_spec.md   模块、目标和当前效果
docs\project_paths.md               本路径说明
docs\metrics_guide.md               模型指标概念
docs\m6_feature_dictionary.md       M6 开局前特征定义和取值范围
docs\m7_baseline_spec.md            M7 简单基线验收
docs\m9_evaluation_spec.md          M9 统一评估验收
docs\m10_calibration_spec.md        M10 概率校准验收
docs\m11_robustness_spec.md         M11 稳健性和错误分析验收
docs\external_benchmark_policy.md   每阶段外部模型差值和可比性规则
reports\data_quality\esta_full\   M4 质量检查 CSV 和结论
reports\pre_round_xgb_initial_to_current_report.md   初始到当前 XGBoost 总结报告
reports\m5_split_leakage_audit.md                    70/20/10 泄漏审计
reports\esta_full_m8_tuned\m8_controlled_tuning_report.md   M8 控制变量调参
```

## 常用命令

进入项目：

```powershell
cd C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
```

运行测试：

```powershell
C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests -v
```

检查代码语法：

```powershell
C:\Users\admin\11\envs\game\python.exe -m compileall src tests
```

训练 M6 开局前 XGBoost：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.train_xgb --task pre_round --data data\processed\esta_full\pre_round.parquet --model-dir models\esta_full_m6 --report-dir reports\esta_full_m6
```

运行 M6 特征分析和消融：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m6_analysis --data data\processed\esta_full\pre_round.parquet --report-dir reports\esta_full_m6
```

运行 M4 质量检查：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.check_quality --input data\interim\esta_full --report-dir reports\data_quality\esta_full
```

运行 M9 统一评估：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m9_evaluation --data data\processed\esta_full\pre_round.parquet --model models\esta_full_m8_tuned\pre_round_xgb.joblib --report-dir reports\esta_full_m9 --bootstrap-samples 2000 --seed 42
```

运行 M10 概率校准：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m10_calibration --data data\processed\esta_full\pre_round.parquet --base-model models\esta_full_m8_tuned\pre_round_xgb.joblib --model-dir models\esta_full_m10 --report-dir reports\esta_full_m10 --folds 5
```

运行 M11 稳健性和错误分析：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m11_robustness --predictions reports\esta_full_m9\test_predictions.csv --data data\processed\esta_full\pre_round.parquet --kills data\interim\esta_full\kills.parquet --report-dir reports\esta_full_m11 --bootstrap-samples 2000 --seed 42 --review-cases 30
```

生成 M11 外部模型差值表：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.benchmark_comparison --metrics reports\esta_full_m9\m9_summary.json --benchmarks benchmarks\external_round_model_metrics.csv --report-dir reports\esta_full_m11 --stage-label M11
```

M2 至 M11 已完成。当前标准数据为 41,074 个开局前样本，质量报告没有
error 或 warning；M9 正式测试 AUC 为 0.7271，系列赛级 95% CI 为
[0.7131, 0.7409]；M10 验证集选择保留原始概率；M11 已完成分组 CI 和 30 个
高置信错误审查。下一阶段是 M12 模型解释。

从 M11 开始，每个阶段报告还要生成 `external_benchmark_comparison.csv` 和
`external_benchmark_comparison.md`，统一说明与公开模型的数值差和可比性。

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
├─ scripts\                    可重复运行的阶段入口
├─ src\csdemo\                 Python 源代码
├─ tasks\                      当前阶段的实施计划和检查表
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
data\processed\esta_full\first_kill.parquet   M15 按最小 tick 修复后的首杀样本
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
src\csdemo\m12_explanation.py  M12 Gain、Permutation、TreeSHAP 和泄漏检查
src\csdemo\predict_pre_round.py M13 单条 JSON/CSV 校验与胜率预测
src\csdemo\m13_interface.py    M13 接口验收和阶段报告
src\csdemo\m14_acceptance.py   M14 最终验收、实验清单和报告
src\csdemo\m15_first_kill_data.py M15 首杀样本重建、主键/split 审计和报告
src\csdemo\m16_first_kill_baselines.py M16 三模型基线、开局控制组和验收报告
src\csdemo\m17_first_kill_tuning.py M17 validation-only 控制变量调参和报告
src\csdemo\m18_first_kill_evaluation.py M18 固定模型 bootstrap、稳健性、错误和校准评估
src\csdemo\m19_first_kill_explanation.py M19 三种解释、分组置换、泄漏与目标距离审计
src\csdemo\predict_first_kill.py M20 首杀后单条 JSON/CSV 校验与胜率预测
src\csdemo\m20_first_kill_interface.py M20 接口、模型/校准器哈希和阶段验收
src\csdemo\m21_first_kill_acceptance.py M21 最终验收、概率回放、实验清单和进度报告
src\csdemo\benchmark_comparison.py  各阶段外部模型差值报告
src\csdemo\train_lgbm.py       LightGBM 固定训练器、五项指标和 validation 早停
src\csdemo\m22_pre_round_lightgbm_baseline.py M22 开局前公平对照、回放和验收报告
src\csdemo\m23_pre_round_lightgbm_tuning.py M23 validation-only 调参、稳定性和冻结评估
src\csdemo\m24_pre_round_lightgbm_evaluation.py M24 固定评估、配对 bootstrap、稳健性和校准
src\csdemo\m25_pre_round_lightgbm_explanation.py M25 三种解释、分组置换、泄漏和 XGBoost 对照
src\csdemo\predict_pre_round_lightgbm.py M26 单条 JSON/CSV 校验、特征对齐和概率预测
src\csdemo\m26_pre_round_lightgbm_interface.py M26 工件合同、CLI、验收和报告生成
src\csdemo\m27_pre_round_lightgbm_acceptance.py M27 冻结回放、阶段链和最终验收
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

M12 模型解释：

```text
reports\esta_full_m12\gain_importance.csv
reports\esta_full_m12\permutation_importance_auc.csv
reports\esta_full_m12\shap_importance.csv
reports\esta_full_m12\importance_comparison.csv
reports\esta_full_m12\all_feature_leakage_audit.csv
reports\esta_full_m12\top20_feature_audit.csv
reports\esta_full_m12\selected_cases.csv
reports\esta_full_m12\case_explanations.csv
reports\esta_full_m12\gain_importance.png
reports\esta_full_m12\permutation_importance_auc.png
reports\esta_full_m12\shap_importance.png
reports\esta_full_m12\shap_summary.png
reports\esta_full_m12\case_explanations.png
reports\esta_full_m12\m12_summary.json
reports\esta_full_m12\m12_explanation_report.md
reports\esta_full_m12\external_benchmark_comparison.csv
reports\esta_full_m12\external_benchmark_comparison.md
```

M13 独立预测接口：

```text
examples\pre_round_snapshot.json
examples\pre_round_snapshot.csv
examples\pre_round_prediction_output.json
reports\esta_full_m13\m13_summary.json
reports\esta_full_m13\example_prediction.json
reports\esta_full_m13\validation_error_examples.json
reports\esta_full_m13\m13_interface_report.md
reports\esta_full_m13\external_benchmark_comparison.csv
reports\esta_full_m13\external_benchmark_comparison.md
```

M14 最终验收：

```text
reports\esta_full_m14\m14_summary.json
reports\esta_full_m14\m14_experiment_manifest.json
reports\esta_full_m14\runtime_environment.json
reports\esta_full_m14\automated_test_output.txt
reports\esta_full_m14\split_assignments.csv
reports\esta_full_m14\m14_final_acceptance_report.md
reports\esta_full_m14\external_benchmark_comparison.csv
reports\esta_full_m14\external_benchmark_comparison.md
```

M15 首杀样本修复与验收：

```text
reports\esta_full_m15\m15_summary.json
reports\esta_full_m15\m15_checks.csv
reports\esta_full_m15\excluded_rounds.csv
reports\esta_full_m15\split_summary.csv
reports\esta_full_m15\automated_test_output.txt
reports\esta_full_m15\m15_first_kill_data_report.md
reports\esta_full_m15\external_benchmark_comparison.csv
reports\esta_full_m15\external_benchmark_comparison.md
```

M16 首杀后固定切分基线：

```text
models\esta_full_m16\first_kill_constant_train_prior.joblib   本地，不提交
models\esta_full_m16\first_kill_logistic_regression.joblib    本地，不提交
models\esta_full_m16\first_kill_xgboost_untuned.joblib        本地，不提交
models\esta_full_m16\first_kill_xgboost_pre_round_control.joblib 本地，不提交
reports\esta_full_m16\m16_model_comparison.csv
reports\esta_full_m16\m16_feature_control.csv
reports\esta_full_m16\test_predictions.csv
reports\esta_full_m16\feature_contract.csv
reports\esta_full_m16\m16_summary.json
reports\esta_full_m16\automated_test_output.txt
reports\esta_full_m16\m16_first_kill_baseline_report.md
reports\esta_full_m16\external_benchmark_comparison.csv
reports\esta_full_m16\external_benchmark_comparison.md
```

M17 首杀后控制变量调参：

```text
models\esta_full_m17\first_kill_xgboost_tuned.joblib   本地，不提交
reports\esta_full_m17\controlled_tuning_results.csv
reports\esta_full_m17\phase_selections.csv
reports\esta_full_m17\seed_stability.csv
reports\esta_full_m17\model_comparison.csv
reports\esta_full_m17\test_predictions.csv
reports\esta_full_m17\final_training_history.csv
reports\esta_full_m17\m17_checks.csv
reports\esta_full_m17\m17_summary.json
reports\esta_full_m17\automated_test_output.txt
reports\esta_full_m17\m17_first_kill_tuning_report.md
reports\esta_full_m17\external_benchmark_comparison.csv
reports\esta_full_m17\external_benchmark_comparison.md
```

M18 首杀后固定模型评估：

```text
models\esta_full_m18\first_kill_calibrator.joblib   本地，不提交
reports\esta_full_m18\global_bootstrap_95ci.csv
reports\esta_full_m18\metrics_by_map_with_ci.csv
reports\esta_full_m18\metrics_by_source_with_ci.csv
reports\esta_full_m18\source_auc_gap.csv
reports\esta_full_m18\validation_oof_calibration.csv
reports\esta_full_m18\test_calibration_comparison.csv
reports\esta_full_m18\all_high_confidence_errors.csv
reports\esta_full_m18\reviewed_top30_errors.csv
reports\esta_full_m18\m18_checks.csv
reports\esta_full_m18\m18_summary.json
reports\esta_full_m18\m18_first_kill_evaluation_report.md
reports\esta_full_m18\external_benchmark_comparison.csv
reports\esta_full_m18\external_benchmark_comparison.md
```

M19 首杀后模型解释与泄漏审计：

```text
reports\esta_full_m19\source_feature_importance.csv
reports\esta_full_m19\grouped_permutation_importance_auc.csv
reports\esta_full_m19\all_feature_leakage_audit.csv
reports\esta_full_m19\target_gap.csv
reports\esta_full_m19\internal_model_gap.csv
reports\esta_full_m19\selected_cases.csv
reports\esta_full_m19\case_explanations.csv
reports\esta_full_m19\m19_checks.csv
reports\esta_full_m19\m19_summary.json
reports\esta_full_m19\m19_first_kill_explanation_report.md
reports\esta_full_m19\external_benchmark_comparison.csv
reports\esta_full_m19\external_benchmark_comparison.md
```

M20 首杀后单条预测接口：

```text
examples\first_kill_snapshot.json
examples\first_kill_snapshot.csv
examples\first_kill_prediction_output.json
reports\esta_full_m20\m20_summary.json
reports\esta_full_m20\m20_checks.csv
reports\esta_full_m20\example_prediction.json
reports\esta_full_m20\validation_error_examples.json
reports\esta_full_m20\model_contract_audit.json
reports\esta_full_m20\m20_first_kill_interface_report.md
reports\esta_full_m20\external_benchmark_comparison.csv
reports\esta_full_m20\external_benchmark_comparison.md
```

M21 首杀后最终验收：

```text
reports\esta_full_m21\m21_summary.json
reports\esta_full_m21\m21_checks.csv
reports\esta_full_m21\m21_experiment_manifest.json
reports\esta_full_m21\runtime_environment.json
reports\esta_full_m21\split_assignments.csv
reports\esta_full_m21\m6_to_m21_stage_metrics.csv
reports\esta_full_m21\m6_to_m21_metric_changes.csv
reports\esta_full_m21\external_benchmark_comparison.csv
reports\esta_full_m21\external_benchmark_comparison.md
reports\esta_full_m21\m21_first_kill_final_acceptance_report.md
reports\m6_to_m21_progress_report.md
```

M22 开局前 LightGBM 受控基线：

```text
models\esta_full_m22\pre_round_lightgbm_baseline.joblib   本地模型，Git 忽略
reports\esta_full_m22\m22_summary.json
reports\esta_full_m22\m22_checks.csv
reports\esta_full_m22\m22_experiment_manifest.json
reports\esta_full_m22\m22_model_comparison.csv
reports\esta_full_m22\m22_test_predictions.csv
reports\esta_full_m22\feature_contract.csv
reports\esta_full_m22\encoded_feature_columns.csv
reports\esta_full_m22\lightgbm_training_history.csv
reports\esta_full_m22\external_benchmark_comparison.csv
reports\esta_full_m22\external_benchmark_comparison.md
reports\esta_full_m22\m22_pre_round_lightgbm_baseline_report.md
reports\xgboost_final_summary.md
reports\m6_to_m22_progress_report.md
```

M23 开局前 LightGBM 调参：

```text
models\esta_full_m23\pre_round_lightgbm_tuned.joblib   本地模型，Git 忽略
reports\esta_full_m23\tuning_candidates.csv
reports\esta_full_m23\phase_selections.csv
reports\esta_full_m23\seed_stability.csv
reports\esta_full_m23\test_predictions.csv
reports\esta_full_m23\m23_model_comparison.csv
reports\esta_full_m23\m23_summary.json
reports\esta_full_m23\m23_experiment_manifest.json
reports\esta_full_m23\m23_pre_round_lightgbm_tuning_report.md
reports\lightgbm_xgboost_external_metrics.md
```

M24 开局前 LightGBM 固定模型评估：

```text
models\esta_full_m24\pre_round_lightgbm_calibrator.joblib   本地校准器，Git 忽略
reports\esta_full_m24\m24_summary.json
reports\esta_full_m24\m24_checks.csv
reports\esta_full_m24\m24_experiment_manifest.json
reports\esta_full_m24\global_bootstrap_95ci.csv
reports\esta_full_m24\paired_lightgbm_vs_xgboost_bootstrap.csv
reports\esta_full_m24\metrics_by_map_with_ci.csv
reports\esta_full_m24\metrics_by_source_with_ci.csv
reports\esta_full_m24\test_calibration_comparison.csv
reports\esta_full_m24\reviewed_top30_errors.csv
reports\esta_full_m24\m24_pre_round_lightgbm_evaluation_report.md
```

M25 开局前 LightGBM 模型解释与泄漏审计：

```text
reports\esta_full_m25\m25_summary.json
reports\esta_full_m25\m25_checks.csv
reports\esta_full_m25\m25_experiment_manifest.json
reports\esta_full_m25\gain_importance.csv
reports\esta_full_m25\permutation_importance_auc.csv
reports\esta_full_m25\grouped_permutation_importance_auc.csv
reports\esta_full_m25\macro_group_permutation_auc.csv
reports\esta_full_m25\shap_importance.csv
reports\esta_full_m25\source_feature_importance.csv
reports\esta_full_m25\all_feature_leakage_audit.csv
reports\esta_full_m25\xgboost_lightgbm_importance_comparison.csv
reports\esta_full_m25\model_importance_agreement.csv
reports\esta_full_m25\selected_cases.csv
reports\esta_full_m25\case_explanations.csv
reports\esta_full_m25\external_benchmark_comparison.csv
reports\esta_full_m25\m25_pre_round_lightgbm_explanation_report.md
```

M26 开局前 LightGBM 单条预测接口：

```text
examples\pre_round_lightgbm_prediction_output.json
reports\esta_full_m26\m26_summary.json
reports\esta_full_m26\m26_checks.csv
reports\esta_full_m26\m26_experiment_manifest.json
reports\esta_full_m26\model_contract_audit.json
reports\esta_full_m26\validation_error_examples.json
reports\esta_full_m26\fixed_test_metrics.csv
reports\esta_full_m26\external_benchmark_comparison.csv
reports\esta_full_m26\m26_pre_round_lightgbm_interface_report.md
```

M27 开局前 LightGBM 最终验收：

```text
reports\esta_full_m27\m27_summary.json
reports\esta_full_m27\m27_checks.csv
reports\esta_full_m27\m27_experiment_manifest.json
reports\esta_full_m27\runtime_environment.json
reports\esta_full_m27\split_assignments.csv
reports\esta_full_m27\replayed_test_predictions.csv
reports\esta_full_m27\fixed_test_metrics.csv
reports\esta_full_m27\paired_lightgbm_vs_xgboost_bootstrap.csv
reports\esta_full_m27\m27_pre_round_lightgbm_final_acceptance_report.md
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
docs\m12_explanation_spec.md        M12 模型解释与泄漏检查验收
docs\m13_prediction_interface_spec.md M13 独立预测接口与使用教程
docs\m14_final_acceptance_spec.md    M14 最终验收与复现教程
docs\m15_first_kill_data_spec.md     M15 首杀定义、主键关联和数据验收
docs\m16_first_kill_baseline_spec.md M16 首杀后特征、基线和验收目标
docs\m17_first_kill_tuning_spec.md M17 调参网格、validation-only 规则和验收目标
docs\m18_first_kill_evaluation_spec.md M18 固定模型评估、分组和校准验收目标
docs\m19_first_kill_explanation_spec.md M19 解释方法、泄漏合同和目标距离定义
docs\m20_first_kill_prediction_interface_spec.md M20 单条输入、模型/校准器和错误合同
docs\m21_first_kill_final_acceptance_spec.md M21 最终验收、一键复现和进度报告合同
docs\m22_pre_round_lightgbm_baseline_spec.md M22 固定数据、特征、训练和公平对照合同
docs\m23_pre_round_lightgbm_tuning_spec.md M23 九阶段网格、validation-only 和稳定性合同
docs\m24_pre_round_lightgbm_evaluation_spec.md M24 固定模型区间、分组、校准和错误审计合同
docs\m25_pre_round_lightgbm_explanation_spec.md M25 解释、泄漏和 XGBoost 排名对照合同
docs\m26_pre_round_lightgbm_prediction_interface_spec.md M26 单条输入、工件绑定和 CLI 合同
docs\m27_pre_round_lightgbm_final_acceptance_spec.md M27 最终回放、哈希和三模式复现合同
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

运行 M12 模型解释（同时生成 M12 外部模型差值表）：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m12_explanation --data data\processed\esta_full\pre_round.parquet --model models\esta_full_m8_tuned\pre_round_xgb.joblib --report-dir reports\esta_full_m12 --permutation-repeats 20 --seed 42 --case-features 10 --shap-plot-rows 1500
```

使用 M13 接口预测一条开局快照：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round --input examples\pre_round_snapshot.json --model models\esta_full_m8_tuned\pre_round_xgb.joblib --calibrator models\esta_full_m10\pre_round_calibrator.joblib
```

重新生成 M13 接口验收和外部模型差值报告：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m13_interface --model models\esta_full_m8_tuned\pre_round_xgb.joblib --calibrator models\esta_full_m10\pre_round_calibrator.joblib --json-example examples\pre_round_snapshot.json --csv-example examples\pre_round_snapshot.csv --metrics reports\esta_full_m9\m9_summary.json --benchmarks benchmarks\external_round_model_metrics.csv --report-dir reports\esta_full_m13
```

运行 M14 最终验收：

```powershell
.\scripts\run_pre_round_pipeline.ps1
```

从本地 ESTA 完整重建到 M14：

```powershell
.\scripts\run_pre_round_pipeline.ps1 -FullRebuild
```

运行 M15 首杀样本修复和验收：

```powershell
.\scripts\run_first_kill_data_stage.ps1
```

运行 M16 首杀后固定切分基线：

```powershell
.\scripts\run_first_kill_baselines.ps1
```

运行 M17 首杀后控制变量调参：

```powershell
.\scripts\run_first_kill_tuning.ps1
```

运行 M18 首杀后固定模型评估：

```powershell
.\scripts\run_first_kill_evaluation.ps1
```

运行 M19 首杀后模型解释与泄漏审计：

```powershell
.\scripts\run_first_kill_explanation.ps1
```

使用 M20 接口预测一条首杀后快照：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_first_kill --input examples\first_kill_snapshot.json --model models\esta_full_m17\first_kill_xgboost_tuned.joblib --calibrator models\esta_full_m18\first_kill_calibrator.joblib
```

运行 M20 首杀后接口验收：

```powershell
.\scripts\run_first_kill_interface.ps1
```

运行 M21 首杀后最终验收：

```powershell
.\scripts\run_first_kill_pipeline.ps1
```

运行 M22 开局前 LightGBM 受控基线：

```powershell
.\scripts\run_pre_round_lightgbm_baseline.ps1
```

运行 M23 开局前 LightGBM 控制变量调参：

```powershell
.\scripts\run_pre_round_lightgbm_tuning.ps1
```

运行 M24 开局前 LightGBM 固定模型评估：

```powershell
.\scripts\run_pre_round_lightgbm_evaluation.ps1
```

运行 M25 开局前 LightGBM 模型解释与泄漏审计：

```powershell
.\scripts\run_pre_round_lightgbm_explanation.ps1
```

运行 M26 开局前 LightGBM 单条 JSON/CSV 接口验收：

```powershell
.\scripts\run_pre_round_lightgbm_interface.ps1
```

直接预测一条 JSON 快照：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round_lightgbm `
  --input examples\pre_round_snapshot.json `
  --model models\esta_full_m23\pre_round_lightgbm_tuned.joblib `
  --calibrator models\esta_full_m24\pre_round_lightgbm_calibrator.joblib
```

运行 M27 开局前 LightGBM 最终验收：

```powershell
.\scripts\run_pre_round_lightgbm_pipeline.ps1
```

从 M14 冻结产物重建 M22-M27 时增加 `-RebuildLightGBM`；从原始 ESTA 完整重建时
增加 `-FullRebuild`。

只从 M14 产物重建首杀后流水线：

```powershell
.\scripts\run_first_kill_pipeline.ps1 -RebuildFirstKill
```

从原始 ESTA 完整重建：

```powershell
.\scripts\run_first_kill_pipeline.ps1 -FullRebuild
```

M14 最低门槛最终验收已通过。当前标准数据为 41,074 个开局前样本，质量报告没有
error 或 warning；M9 正式测试 AUC 为 0.7271，系列赛级 95% CI 为
[0.7131, 0.7409]；M10 验证集选择保留原始概率；M11 已完成分组 CI 和 30 个
高置信错误审查；M12 已完成三种重要性、43 个特征泄漏检查和三个回合案例解释；
M13 已提供单条 JSON/CSV 预测；M14 已锁定环境、运行 70 项测试、保存哈希和 782 条
系列赛 split 清单。M15 已使用完整三列主键和最小 tick 重建 41,027 条首杀后样本，
排除 47 个无有效击杀回合，12 个阻塞检查与 80 项测试全部通过。旧样本有 14,357 条
首杀事件发生变化；历史首杀模型指标因此继续作废。M16 已在同一 split 和同一特征合同
下完成 Dummy、逻辑回归和未经调参 XGBoost：测试 AUC 分别为 0.5000、0.8091、
0.8089；XGBoost 的 Accuracy/Log Loss/Brier 为 0.7453/0.5248/0.1763。首杀特征相对
同样本开局控制组增加 0.0880 validation AUC。8 个阻塞检查与 90 项测试通过。M17 在
只使用 train/validation 的 39 个候选中选择 1,500 树上限、early stopping 50、depth 2、
subsample 0.9，正式 seed 42 使用 409 棵树。测试 AUC/Log Loss/Brier 为
0.8098/0.5231/0.1757，相对 M16 分别改善 0.0009/0.0016/0.0006；Accuracy 下降 0.0012，
ECE10 恶化 0.0045。12 个阻塞检查与 100 项测试通过。M18 没有重新训练 XGBoost，
模型概率回放最大误差为 1.11e-16；测试 AUC 为 0.8098，系列赛级 95% CI 为
[0.7977, 0.8221]，Log Loss 为 0.5231，95% CI 为 [0.5097, 0.5361]。LAN-online
AUC 差为 -0.0103，95% CI 为 [-0.0346, 0.0148]；主要地图最低 AUC 为 0.7839。
validation OOF 选择保留原始概率。13 个阻塞检查与 108 项测试通过，下一阶段是 M19
模型解释与特征泄漏审计。M19 已完成 Gain、20 次编码/原始特征分组 Permutation 和
TreeSHAP；82 个编码列全部映射到 40 个允许原始特征，泄漏失败数为 0，SHAP 概率
重建最大误差为 4.03e-7。十项正式目标全部通过，首要特征为首杀阵营优势，其次为
购买结束装备价值差。9 个阻塞检查与 118 项测试通过。M20 已提供严格校验的单条 JSON/CSV 命令：31 个必填输入
生成 40 个原始特征并对齐 82 个编码列；模型和校准器哈希一致，JSON/CSV 概率完全
相同，10 个错误案例全部拒绝。示例输出 CT/T 为 0.718764/0.281236，但这只是接口演示，
不改变固定测试指标。10 个阻断检查和 131 项测试通过。M21 已完成最终验收：17/17
阻断项和 145 项测试通过，4,170 条测试概率的最大回放误差为 1.11e-16，五项指标误差
为 0，十项目标 Remaining 全部为 0，series/game/round 跨集合交叉均为 0。首杀后
XGBoost 已完成；M22 随后在相同数据合同下开始 LightGBM 控制变量对照。

M22 已完成第一版开局前 LightGBM 公平基线：13/13 阻断项和 155 项测试通过，最佳
迭代为 115。测试 Accuracy/AUC/Log Loss/Brier/ECE10 为 0.650767/0.727846/
0.591437/0.205201/0.018875；相对冻结 XGBoost 分别变化 +0.003356/+0.000724/
-0.000296/-0.000094/-0.004323。五项最低门槛全部通过，但更高目标只通过 ECE10，
下一阶段为 M23：只按 validation Log Loss 做 LightGBM 控制变量调参。

M23 已完成 9 阶段 36 候选和 5 个种子。没有候选达到 `0.0001` 的 validation Log Loss
改善门槛，最终保留 M22 参数和指标。14/14 阻断项、165 项测试和源码编译通过；下一
阶段为 M24 固定模型评估、稳健性和校准。

M24 已完成固定 LightGBM 的系列赛级 2,000 次 bootstrap、四类稳健性分组、五项
LightGBM-XGBoost 配对区间、validation-only 校准和 30 个高置信错误复核。16/16
阻断项、176 项测试和源码编译通过。点指标均略优于 XGBoost，但配对区间全部包含 0；
M25 已完成固定模型的 Gain/Split、20 次编码列/36 个原始特征/五个宏观组
Permutation、原生 TreeSHAP、完整泄漏审计和 M12 XGBoost 解释对照。43 个编码列全部
映射成功，泄漏失败为 0，SHAP 概率重建误差为 7.77e-16。14/14 阻断项、192 项测试和
源码编译通过。M26 已把冻结 LightGBM 封装为单条 JSON/CSV 接口：27 个基础字段自动
生成 9 个差值，严格对齐 36 个原始和 43 个编码特征，JSON/CSV 概率差为 0，10/10
非法案例被拒绝。15/15 阻断项、201 项测试和源码编译通过，模型与校准器哈希不变，
M27 随后完成 4,172 条测试概率最终回放，最大误差为 `1.11e-16`，五项指标误差为 0，
配对 bootstrap 显著领先指标仍为 0。19/19 阻断项、211 项测试、源码编译和三模式
复现入口通过，购买结束 LightGBM 正式关闭。下一步先形成老师查收的独立正式报告，
再进入 M28 首杀后 LightGBM 受控基线。M27 核心文件为：

```text
src\csdemo\m27_pre_round_lightgbm_acceptance.py
tests\test_m27_pre_round_lightgbm_acceptance.py
scripts\run_pre_round_lightgbm_pipeline.ps1
docs\m27_pre_round_lightgbm_final_acceptance_spec.md
reports\esta_full_m27\m27_pre_round_lightgbm_final_acceptance_report.md
reports\esta_full_m27\m27_summary.json
reports\esta_full_m27\m27_experiment_manifest.json
```

从 M11 开始，每个阶段报告还要生成 `external_benchmark_comparison.csv` 和
`external_benchmark_comparison.md`，统一说明与公开模型的数值差和可比性。

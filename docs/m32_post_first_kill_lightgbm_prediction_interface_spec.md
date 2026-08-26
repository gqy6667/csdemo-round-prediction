# M32 首杀后 LightGBM 单条预测接口规格

## 1. 目标

M32 为 M29/M30 已冻结的首杀后 LightGBM 提供严格、可复现的单条 JSON/CSV
预测接口。接口只负责验证输入、构造特征、加载模型与 identity 校准器并返回概率，
不训练、不调参、不改阈值、不重新选择校准方法，也不依据示例输入修改模型。

预测时点固定为：购买结束后，最早有效敌方击杀刚发生，且首杀之后的事件尚未使用。

## 2. 冻结工件

- 模型：`models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib`；
- 校准器：`models/esta_full_m30/post_first_kill_lightgbm_calibrator.joblib`；
- 解释前置验收：`reports/esta_full_m31/m31_summary.json`；
- 评估前置验收：`reports/esta_full_m30/m30_summary.json`；
- 外部比较：`reports/esta_full_m31/external_benchmark_comparison.csv`；
- 模型 SHA-256：
  `35ce17435a3716efcfdd49dd5ca13ff441e75c65512322627249e8920546a5b5`；
- 校准器 SHA-256：
  `c5453403a25dfb03bbda131028fda7bdfde934840093de3e527ad2988c8043e5`。

必须核对任务名 `post_first_kill`、模型名 `lightgbm_tuned`、profile、数据哈希、
40 个原始特征、82 个编码列、8 张地图、36 种首杀武器和 211 棵部署树。
LightGBM Booster 只允许将编码列名中的空格规范化为下划线，列位置和集合不得漂移。

校准器必须绑定上述模型和数据，`selection_data` 为 `validation only`、
`validation_folds` 为 5、方法为 `uncalibrated`。identity 校准前后概率差不得超过
`1e-15`。

## 3. 输入合同

接口接受一个 JSON 对象或一行 CSV，共 31 个用户字段：

- 27 个购买结束基础字段；
- `first_kill_advantage_ct`：仅允许 `-1` 或 `1`；
- `first_kill_time`：有限数，范围 `[0, 180]` 秒；
- `first_kill_headshot`：布尔值或 `0/1`；
- `first_kill_weapon`：必须是训练集中出现的 36 个类别之一。

接口复用 M20 已验收的验证逻辑，自动计算 9 个 CT-T 差值，最终按冻结顺序形成
40 个原始特征，再严格重排为 82 个编码列。未知地图、未知武器、字段缺失、类型错误、
派生差值不一致、标签/主键/身份字段、冗余存活字段及首杀后的未来事件必须拒绝。

## 4. 输出合同

成功时返回 JSON，至少包含：

- `task=post_first_kill`；
- `model_name=lightgbm_tuned`；
- 预测时点定义、模型哈希和 `calibration_method=uncalibrated`；
- 基础 CT 概率、最终 CT/T 概率、预测方、0.5 阈值和概率和；
- 31/9/40/82 特征计数、地图/武器、211 棵树及工件合同状态。

所有概率必须在 `[0,1]`，CT/T 概率和误差不超过 `1e-12`。同一快照的 JSON 与
CSV 概率差不超过 `1e-15`。非法输入返回退出码 2 和可解析的错误 JSON。

## 5. 冻结指标

M32 不产生新的测试集预测或模型选择。正式报告只复用 M30/M31 的冻结指标：

- Accuracy 0.7429256594724221；
- AUC 0.808255446182；
- Log Loss 0.5240626574288818；
- Brier 0.1760026226340809；
- ECE10 0.014190844077173683。

这些数值必须与 M30/M31 完全一致。M32 示例概率不是测试集指标。

## 6. 阻断验收

1. `m31_m30_prerequisite`：M30/M31 状态、模型、校准器、数据和指标冻结；
2. `artifact_contracts`：模型与校准器内部合同通过；
3. `json_csv_validation`：JSON/CSV 都通过输入验证；
4. `json_csv_prediction_match`：同一快照概率差 <= `1e-15`；
5. `probability_contract`：基础/最终/互补概率和 identity 合同通过；
6. `invalid_examples`：10 类非法输入全部拒绝；
7. `feature_alignment`：31/9/40/82、8 地图、36 武器、211 树全部一致；
8. `fixed_metrics`：M30/M31 五项指标完全一致；
9. `artifact_integrity`：模型和校准器运行前后哈希不变；
10. `external_report`：M31 外部比较不改变；
11. `cli_contract`：成功和失败 CLI 路径通过；
12. `automated_tests`：全量测试通过；
13. `source_compile`：源码和测试编译通过；
14. `reproduction_entrypoint`：一键脚本合同通过；
15. `artifact_manifest`：输入输出哈希齐全。

## 7. 使用与产物

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_first_kill_lightgbm `
  --input examples\first_kill_snapshot.json `
  --model models\esta_full_m29\post_first_kill_lightgbm_tuned.joblib `
  --calibrator models\esta_full_m30\post_first_kill_lightgbm_calibrator.joblib
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_post_first_kill_lightgbm_interface.ps1
```

```text
src/csdemo/predict_first_kill_lightgbm.py
src/csdemo/m32_post_first_kill_lightgbm_interface.py
tests/test_m32_post_first_kill_lightgbm_prediction.py
scripts/run_post_first_kill_lightgbm_interface.ps1
reports/esta_full_m32/m32_summary.json
reports/esta_full_m32/m32_experiment_manifest.json
reports/esta_full_m32/m32_post_first_kill_lightgbm_interface_report.md
reports/esta_full_m32/example_prediction.json
reports/esta_full_m32/validation_error_examples.json
reports/esta_full_m32/model_contract_audit.json
```

## 8. 下一阶段

M32 通过后进入 M33：从 M28 到 M32 做最终阶段链、哈希、概率、指标、解释、接口、
全量测试和一键复现验收。第四份老师正式报告必须等 M33 通过后生成。

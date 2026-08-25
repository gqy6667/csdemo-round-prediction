# M26 开局前 LightGBM 单条预测接口规格

## 1. 目标

M26 把 M23 训练、M24 评估、M25 解释通过的购买结束 LightGBM 封装成可直接使用的
单条 JSON/CSV 推理接口。用户提供购买结束、交火前的一条回合快照，接口输出 CT/T
回合胜率，并严格校验输入、模型 bundle、校准器、特征列和工件哈希。

本阶段只做推理封装，不训练、不调参、不改变特征、阈值、校准方法或测试概率。

## 2. 冻结输入

```text
models/esta_full_m23/pre_round_lightgbm_tuned.joblib
models/esta_full_m24/pre_round_lightgbm_calibrator.joblib
reports/esta_full_m24/m24_summary.json
reports/esta_full_m25/m25_summary.json
reports/esta_full_m25/external_benchmark_comparison.csv
examples/pre_round_snapshot.json
examples/pre_round_snapshot.csv
```

必须验证：

- M25 状态为 passed 且 `ready_for_m26=True`；
- 模型 SHA-256 与 M25 合同一致；
- 校准器 SHA-256 与 M24 产物一致；
- 校准器记录的基础模型和数据 SHA-256 与当前模型一致；
- 模型任务为 `pre_round`，名称为 `lightgbm_tuned`，profile 为
  `M14_pre_round_features`；
- 原始特征严格为 36 个，编码列严格为 43 个且顺序与 bundle 一致；
- LightGBM Booster 和 bundle 都使用 115 棵部署树；
- 推理和验收路径的 LightGBM fit 调用为 0，模型和校准器文件运行前后哈希不变。

## 3. 输入合同

### 3.1 必填的 27 个基础字段

```text
map_name, round_num, ct_score, t_score,
ct_eq_value, t_eq_value, ct_cash, t_cash,
ct_armor, t_armor, ct_helmets, t_helmets, ct_defuse_kits,
ct_grenades, t_grenades,
ct_ak47, t_ak47, ct_m4a4, t_m4a4, ct_m4a1_s, t_m4a1_s,
ct_awp, t_awp, ct_rifles, t_rifles, ct_smgs, t_smgs
```

字段类型、范围和装备一致性规则与 M13 相同。地图类别从冻结模型的编码列读取，不维护
一份可能漂移的手工列表。

### 3.2 自动生成的 9 个差值

```text
score_diff_ct, eq_value_diff_ct, cash_diff_ct,
armor_diff_ct, helmet_diff_ct, grenade_diff_ct,
awp_diff_ct, rifle_diff_ct, smg_diff_ct
```

全部统一为 CT 减 T。用户可以省略；如果提供，必须与接口计算值完全一致。

### 3.3 一致性与禁止字段

- `round_num = ct_score + t_score + 1`；
- 头盔人数不能超过护甲人数；
- AK/M4A4/M4A1-S 总数不能大于步枪总数；
- 步枪、AWP、SMG 总数不能超过每方五人；
- 未知地图、缺字段、字符串数字、非有限数、超范围和多行 CSV 全部拒绝；
- ID、split、标签、winner、首杀、击杀、伤害、血量、存活变化、下包、拆包、
  回合结束、战队和选手身份字段全部禁止。

校验失败时 CLI 返回退出码 `2`，stderr 输出结构化错误 JSON。接口不得对非法输入
勉强给出概率。

## 4. 预处理和推理

```text
一条 JSON/CSV
  -> 严格校验 27 个基础字段
  -> 自动生成 9 个 CT-T 差值
  -> 36 个原始特征
  -> 训练时相同的 get_dummies
  -> 严格对齐冻结的 43 个编码列
  -> LightGBM predict_proba
  -> M24 identity 校准器
  -> CT/T 概率
```

未知类别不能静默编码为全零，编码列不能增删或改变顺序。

## 5. 输出合同

成功输出 JSON：

```json
{
  "task": "pre_round",
  "model_name": "lightgbm_tuned",
  "snapshot_definition": "freeze-time end after purchases and before combat",
  "calibration_method": "uncalibrated",
  "validation": {
    "status": "passed",
    "raw_model_feature_count": 36,
    "encoded_model_feature_count": 43,
    "deployment_tree_count": 115
  },
  "prediction": {
    "base_ct_win_probability": 0.0,
    "ct_win_probability": 0.0,
    "t_win_probability": 1.0,
    "predicted_side": "T",
    "decision_threshold": 0.5,
    "probability_sum": 1.0
  }
}
```

`base_ct_win_probability` 是 LightGBM 原始概率；当前校准器为 identity，因此与最终
CT 概率相同。保留两列是为了未来校准方法变化时仍能追踪。

## 6. 指标与外部比较

接口不产生新的测试指标。M25 的五项冻结指标必须原样保留：Accuracy `0.650767`、
AUC `0.727846`、Log Loss `0.591437`、Brier `0.205201`、ECE10 `0.018875`。

外部比较逐行复用 M25 的四行结果。示例概率只用于验证接口，不能当作模型效果。

## 7. 代码和命令

```text
src/csdemo/predict_pre_round_lightgbm.py
src/csdemo/m26_pre_round_lightgbm_interface.py
tests/test_m26_pre_round_lightgbm_prediction.py
scripts/run_pre_round_lightgbm_interface.ps1
examples/pre_round_lightgbm_prediction_output.json
reports/esta_full_m26/
```

单条 JSON 预测：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round_lightgbm `
  --input examples\pre_round_snapshot.json `
  --model models\esta_full_m23\pre_round_lightgbm_tuned.joblib `
  --calibrator models\esta_full_m24\pre_round_lightgbm_calibrator.joblib
```

正式验收：

```powershell
.\scripts\run_pre_round_lightgbm_interface.ps1
```

## 8. 测试策略

- 模型合同：任务、名称、profile、36/43 列、115 棵树和特征数漂移均失败；
- 校准器合同：任务、方法、模型哈希、数据哈希、validation-only 信息漂移均失败；
- 输入合同：合法输入生成准确 36 列，未知地图、错误差值、装备矛盾和禁止字段失败；
- 推理：JSON/CSV 概率完全一致，原始/最终概率有限，CT/T 互补；
- CLI：成功输出和退出码 2 的结构化失败均测试；
- 验收：M25/M24 前置、哈希不变、固定指标、外部表、测试、编译和入口完整。

## 9. 阻断验收

1. M25 和 M24 前置状态、模型哈希及校准器哈希通过；
2. 模型和校准器完整合同通过；
3. JSON/CSV 合法示例均通过且概率最大差不超过 `1e-15`；
4. 原始概率、最终 CT/T 概率有限、位于 `[0,1]` 且 CT+T 误差不超过 `1e-12`；
5. 预定义的全部非法案例被拒绝并返回具体原因；
6. 27 个输入、9 个派生、36 个原始、43 个编码特征和 115 棵树严格对齐；
7. M25 五项指标保持不变，LightGBM fit 调用为 0；
8. 模型和校准器运行前后 SHA-256 不变；
9. M25 四行外部比较、中文报告、实验清单、自动化测试、源码编译和一键入口完整。

## 10. 边界与后续

始终执行：严格输入校验、模型与校准器哈希绑定、保存原始/最终概率、完整错误信息。

另立阶段：批量接口、HTTP API、GUI、首杀后 LightGBM、实时流、战队或选手身份。

禁止执行：接口阶段训练、根据示例概率调参、接受未知类别、加入标签或未来字段、提交
本地 ESTA 数据或未允许提交的模型。

M26 通过后进入 M27：购买结束 LightGBM 最终验收和一键复现。之后再决定首杀后
LightGBM 与实时胜率模块的顺序。

## 11. 实际结果

M26 正式运行通过 15/15 个阻断检查、201 项自动化测试和源码编译。JSON 与 CSV 示例
的 CT 概率差为 `0`，10/10 个非法输入案例全部被拒绝，LightGBM `fit` 调用为 `0`。
模型和校准器运行前后 SHA-256 均未变化。

示例购买结束快照输出 CT/T 胜率为 `0.5507644902 / 0.4492355098`，预测方为 CT；该值
只说明接口可用，不是新增测试指标。五项测试指标原样保持为 Accuracy `0.650767`、
AUC `0.727846`、Log Loss `0.591437`、Brier `0.205201`、ECE10 `0.018875`。

正式状态为 `passed`、`ready_for_m27=True`。完整证据位于
`reports/esta_full_m26/m26_pre_round_lightgbm_interface_report.md`、`m26_summary.json`
和 `m26_experiment_manifest.json`。

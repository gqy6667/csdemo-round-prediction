# M20 首杀后单条预测接口规格

## 1. 目标

M20 把 M17 训练、M18 评估、M19 解释通过的首杀后 XGBoost 封装成可直接使用的
单条 JSON/CSV 推理接口。用户提供购买结束状态和刚发生的首杀事件，接口输出 CT/T
回合胜率，并对输入、模型 bundle、校准器和特征列做严格校验。

本阶段只做推理封装，不训练、不调参、不改变校准方法或测试概率。

## 2. 已确认假设

1. M20 只支持一次预测一条快照，不在本阶段实现批量文件或实时流；
2. 正式模型固定为 `models/esta_full_m17/first_kill_xgboost_tuned.joblib`；
3. 正式校准器固定为 M18 选择的 identity bundle；虽然当前不改变概率，仍作为部署合同加载；
4. 输入直接使用模型字段 `first_kill_advantage_ct`，其中 `1` 表示 CT 首杀，`-1` 表示 T 首杀；
5. 首杀时间单位沿用 ESTA 表中的秒，接口接受 `0` 到 `180` 的有限数；
6. 未知地图和未知首杀武器严格拒绝，不静默编码成全零；
7. M19 的正式指标、目标余量和外部指标继续报告，但不根据接口结果重新评价模型。

## 3. 冻结前置产物

```text
models/esta_full_m17/first_kill_xgboost_tuned.joblib
models/esta_full_m18/first_kill_calibrator.joblib
reports/esta_full_m17/model_comparison.csv
reports/esta_full_m18/m18_summary.json
reports/esta_full_m19/m19_summary.json
benchmarks/external_first_kill_tuned_metrics.csv
```

必须验证：

- M19 状态为 passed 且 `ready_for_m20=True`；
- 模型 SHA-256 与 M19 前置合同一致；
- 校准器 SHA-256 与 M18 产物一致；
- 校准器记录的基础模型 SHA-256 与实际模型一致；
- 模型任务为 `first_kill`、profile 为 `canonical_event`；
- 原始特征严格为 40 个，编码列严格为 82 个且顺序与 bundle 一致；
- 模型使用 409 棵部署树，推理路径不得调用 `fit()`。

## 4. 输入合同

### 4.1 购买结束字段

复用 M13 的 27 个必填基础字段和 9 个自动生成差值：

```text
map_name, round_num, ct_score, t_score,
ct_eq_value, t_eq_value, ct_cash, t_cash,
ct_armor, t_armor, ct_helmets, t_helmets, ct_defuse_kits,
ct_grenades, t_grenades,
ct_ak47, t_ak47, ct_m4a4, t_m4a4, ct_m4a1_s, t_m4a1_s,
ct_awp, t_awp, ct_rifles, t_rifles, ct_smgs, t_smgs
```

自动生成：

```text
score_diff_ct, eq_value_diff_ct, cash_diff_ct, armor_diff_ct,
helmet_diff_ct, grenade_diff_ct, awp_diff_ct, rifle_diff_ct, smg_diff_ct
```

用户可以省略差值；如果提供，必须与 CT 减 T 的结果一致。

### 4.2 首杀事件字段

| 字段 | 类型和范围 | 含义 |
|---|---|---|
| `first_kill_advantage_ct` | 整数 `-1` 或 `1` | `1` 为 CT 首杀，`-1` 为 T 首杀 |
| `first_kill_time` | 有限数，`0 <= x <= 180` | 购买结束后的首杀秒数 |
| `first_kill_headshot` | JSON 布尔或整数 `0/1` | 是否爆头 |
| `first_kill_weapon` | 非空字符串、训练类别之一 | 首杀武器 |

训练模型包含 36 个首杀武器类别。类别从保存的编码列读取，不另建可能漂移的手工名单。

### 4.3 一致性和禁止字段

- `round_num` 必须等于 `ct_score + t_score + 1`；
- 头盔不能多于护甲人数；
- 已命名步枪总数不能大于步枪总数；
- 步枪、AWP、SMG 总数不能超过 5；
- 缺字段、错误类型、非有限数、超范围和未知类别全部拒绝；
- `series_id`、`game_id`、`round_id`、`split`、`ct_win` 和 winner 字段禁止；
- 第二次及后续击杀、伤害、血量、下包、拆包和回合结束字段禁止；
- 冗余的首杀方/死亡方和 4v5/5v4 存活字段不作为接口输入。

## 5. 预处理和预测

```text
31 个必填输入字段
  -> 校验并规范化
  -> 自动生成 9 个 CT-T 差值
  -> 40 个原始模型字段
  -> 训练时相同的 get_dummies
  -> 严格重排为保存的 82 个编码列
  -> 冻结 XGBoost predict_proba
  -> M18 identity calibrator
  -> CT/T 概率
```

不得从接口输入猜测未知类别，不得修改模型 bundle 中保存的列顺序。

## 6. 输出合同

成功输出 JSON：

```json
{
  "task": "first_kill",
  "snapshot_definition": "purchase complete, immediately after earliest valid enemy kill",
  "calibration_method": "uncalibrated",
  "validation": {
    "status": "passed",
    "raw_model_feature_count": 40,
    "encoded_model_feature_count": 82
  },
  "prediction": {
    "ct_win_probability": 0.0,
    "t_win_probability": 1.0,
    "predicted_side": "T",
    "decision_threshold": 0.5,
    "probability_sum": 1.0
  }
}
```

失败时命令退出码为 2，并在 stderr 输出 `error_type`、总错误消息和逐条错误列表。

## 7. 外部指标和目标距离

M20 继续生成 M19 同结构的七行外部指标差值。差值为“本项目 - 外部报告”，保留
`closest_task`、`partial`、`not_comparable` 标签。

M19 十项正式目标必须仍为 10/10 通过，所有 remaining 为 0。M20 不新增模型性能
目标，也不把 JSON/CSV 示例概率当作模型评价指标。

## 8. 技术和项目结构

```text
src/csdemo/predict_first_kill.py        单条预测、校验和 CLI
src/csdemo/m20_first_kill_interface.py  阶段验收、报告和外部比较
tests/test_m20_first_kill_prediction.py 单元和小型集成测试
examples/first_kill_snapshot.json       JSON 示例
examples/first_kill_snapshot.csv        CSV 示例
scripts/run_first_kill_interface.ps1    正式阶段入口
reports/esta_full_m20/                  验收产物
```

复用 `predict_pre_round.py` 的购买字段校验和 `m16_first_kill_baselines.py` 的正式特征
合同，不新增依赖。

## 9. 测试策略

- 合法快照自动生成 9 个差值并形成 40 个原始特征；
- JSON 布尔和整数 0/1 爆头输入都能规范化；
- 缺失首杀字段、优势为 0、时间超限、未知武器全部拒绝；
- 未知地图、不一致差值、inventory 矛盾继续拒绝；
- ID、标签、后续事件和冗余存活字段拒绝；
- 模型/校准器任务、哈希、原始列和编码列合同不匹配时失败；
- 正式模型输出有限互补概率；
- JSON 与 CSV 对同一快照输出完全一致；
- CLI 成功和失败退出码均测试；
- 验收 runner 生成摘要、错误案例、中文报告和外部比较。

采用标准库 `unittest`。先提交会失败的测试，确认 RED，再逐片实现到 GREEN。

## 10. 命令

```powershell
# 单条预测
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_first_kill `
  --input examples\first_kill_snapshot.json `
  --model models\esta_full_m17\first_kill_xgboost_tuned.joblib `
  --calibrator models\esta_full_m18\first_kill_calibrator.joblib

# 正式 M20 验收
.\scripts\run_first_kill_interface.ps1

# 测试和编译
C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests -v
C:\Users\admin\11\envs\game\python.exe -m compileall src tests
```

## 11. 阻断验收

1. M19 前置状态、模型哈希和 M18 校准器哈希通过；
2. 模型与校准器任务、数据和列合同通过；
3. JSON/CSV 合法示例均通过且概率完全一致；
4. 两个概率有限、位于 0 到 1 且和为 1；
5. 预先定义的全部错误案例均被拒绝并返回具体原因；
6. 40 个原始特征和 82 个编码列严格对齐；
7. M18 固定指标和 M19 十项目标保持不变；
8. 七行外部差值及可比性报告生成；
9. CLI 成功/失败行为、全部自动化测试和源码编译通过；
10. 中文报告、示例输入输出、摘要和复现命令完整。

## 12. 边界

始终执行：严格输入校验、模型哈希关联、保存概率和验证细节、完整错误信息。

另立阶段：批量接口、HTTP API、图形界面、LightGBM、实时流、战队/选手身份。

禁止执行：接口阶段重新训练、根据示例概率调参、接受未知类别、加入标签或未来字段、
提交本地 ESTA 数据或未允许提交的模型。

## 13. 交付物与后续

```text
reports/esta_full_m20/m20_summary.json
reports/esta_full_m20/m20_checks.csv
reports/esta_full_m20/example_prediction.json
reports/esta_full_m20/validation_error_examples.json
reports/esta_full_m20/model_contract_audit.json
reports/esta_full_m20/m20_first_kill_interface_report.md
reports/esta_full_m20/external_benchmark_comparison.csv
reports/esta_full_m20/external_benchmark_comparison.md
reports/esta_full_m20/automated_test_output.txt
```

M20 通过后进入 M21：首杀后 XGBoost 最终验收和一键复现。

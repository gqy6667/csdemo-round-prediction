# M26 开局前 LightGBM 单条预测接口验收

## 结论

M26 阻断检查 15/15 通过，状态为 `passed`，可以进入 M27。接口只加载 M23/M24
冻结工件，没有训练、调参或修改概率。

## 如何使用

在项目目录执行：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round_lightgbm `
  --input examples\pre_round_snapshot.json `
  --model models\esta_full_m23\pre_round_lightgbm_tuned.joblib `
  --calibrator models\esta_full_m24\pre_round_lightgbm_calibrator.joblib
```

CSV 只需把 `--input` 改为 `examples\pre_round_snapshot.csv`。需要保存结果时
增加 `--output my_prediction.json`。非法输入返回退出码 2 和错误 JSON。

## 输入与工件合同

接口接收 27 个基础字段，自动计算 9 个 CT-T 差值，形成 36 个原始特征并严格对齐 43 个编码列。地图类别 8 个，部署树 115 棵。

模型 SHA-256 为 `3a95983ed73cd99ae0178a16009036d48510e1ad091d33994cf296dfc69244fd`；校准器 SHA-256 为
`84e6b533e50bb9e169bb34cbbf748d6566482de716510e9f8dd733ec08147ff1`。两者运行前后均未变化，校准器
绑定同一模型和数据，并记录只用 validation 选择。

## 示例输出

示例原始 CT 概率为 0.5507644902，identity
校准后的 CT/T 概率为 0.5507644902 / 0.4492355098，预测方为 `CT`。JSON 与 CSV 的 CT 概率最大差为 0.000e+00。

这只是该示例快照的接口输出，不是测试集指标，也不是投注建议。

## 非法输入

| 案例 | 已拒绝 | 错误数 | 首条错误 |
|---|---|---:|---|
| missing_required_field | True | 2 | missing required fields: round_num |
| unknown_map | True | 1 | map_name 'de_cache' was not seen during training; choose one of ['de_ancient', 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass', 'de_train', 'de_vertigo'] |
| round_score_inconsistency | True | 1 | round_num must equal ct_score + t_score + 1; expected 4, got 5 |
| string_numeric_type | True | 1 | ct_eq_value must be an integer; got '22000' |
| inventory_inconsistency | True | 1 | ct_rifles (1) cannot be smaller than the named rifle total (3) |
| derived_feature_inconsistency | True | 1 | score_diff_ct must equal ct_score - t_score; expected 1, got 99 |
| identifier_field | True | 1 | unknown fields: series_id |
| target_leakage | True | 1 | unknown fields: ct_win |
| future_first_kill_field | True | 1 | unknown fields: first_kill_time |
| team_identity_field | True | 1 | unknown fields: team_name |

## 冻结指标

| Accuracy | AUC | Log Loss | Brier | ECE10 |
|---:|---:|---:|---:|---:|
| 0.650767 | 0.727846 | 0.591437 | 0.205201 | 0.018875 |

M26 不重新计算或选择模型，以上五项与 M25 完全一致。外部比较仍为 4 行，完整差值见 `external_benchmark_comparison.csv`。

## 验收与下一步

十类非法输入全部拒绝，CLI 成功/失败路径通过。自动化测试 201 项通过，源码编译通过。M27 将做
购买结束 LightGBM 最终验收和一键复现；批量、HTTP、GUI、身份特征和实时
胜率仍不在本阶段。

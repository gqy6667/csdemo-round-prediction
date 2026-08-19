# M20 首杀后单条预测接口验收报告

## 阶段结论

阻断验收状态：**passed**；可进入 M21：**True**。
M20 没有训练、调参或改变固定测试概率，只把 M17/M18 模型封装为严格校验的
JSON/CSV 单条预测命令。

## 输入与模型合同

- 购买结束必填字段：27 个；
- 首杀事件必填字段：4 个；
- 自动生成 CT-T 差值：9 个；
- 原始模型特征：40 个；
- 编码模型列：82 个；
- 已知地图：8 类；
- 已知首杀武器：36 类；
- 部署树：409 棵；
- 模型 SHA-256：`ecfaaf93031e78207f81ab5ad9674020657018c0601953238ee6b68e367e8279`；
- 校准器 SHA-256：`661db6964786dde1276dbeb6c0cf3f175858ad0bf02ecb109bb2f72c45074157`。

未知地图/武器、ID、标签、第二次击杀、伤害/血量、下包和冗余存活字段都会被拒绝。

## 示例预测

- 地图：`de_nuke`；
- 首杀方：`CT`；
- 首杀武器：`AK-47`；
- CT 胜率：`0.718764`；
- T 胜率：`0.281236`；
- 判定方：`CT`；
- 概率和：`1.000000000000`；
- JSON/CSV 完全一致：`True`；
- 校准方式：`uncalibrated`。

这是一条局面的模型估计，不表示较高概率一方一定赢得回合。

## 错误输入验收

| 错误类型 | 是否拒绝 | 错误数 | 首条信息 |
|---|---|---:|---|
| missing_first_kill_field | True | 2 | missing required first-kill fields: first_kill_time |
| invalid_first_kill_advantage | True | 1 | first_kill_advantage_ct must be -1 or 1; got 0 |
| first_kill_time_out_of_range | True | 1 | first_kill_time must be between 0 and 180 seconds; got 181 |
| invalid_headshot | True | 1 | first_kill_headshot must be a boolean or 0/1; got 2 |
| unknown_first_kill_weapon | True | 1 | first_kill_weapon 'Unknown Blaster' was not seen during training; choose one of ['AK-47', 'AUG', 'AWP', 'CZ75 Auto', 'Desert Eagle', 'Dual Berettas', 'FAMAS', 'Five-SeveN', 'G3SG1', 'Galil AR', 'Glock-18', 'HE Grenade', 'Incendiary Grenade', 'Knife', 'M4A1', 'M4A4', 'MAC-10', 'MAG-7', 'MP5-SD', 'MP7', 'MP9', 'Molotov', 'Negev', 'Nova', 'P2000', 'P250', 'P90', 'PP-Bizon', 'SCAR-20', 'SG 553', 'SSG 08', 'Tec-9', 'UMP-45', 'USP-S', 'XM1014', 'Zeus x27'] |
| unknown_map | True | 1 | map_name 'de_cache' was not seen during training; choose one of ['de_ancient', 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass', 'de_train', 'de_vertigo'] |
| inconsistent_derived_feature | True | 1 | score_diff_ct must equal ct_score - t_score; expected 1, got 99 |
| target_leakage | True | 1 | forbidden fields: ct_win |
| future_second_kill | True | 1 | forbidden fields: second_kill_weapon |
| redundant_alive_state | True | 1 | forbidden fields: ct_alive_after_fk |

## 固定测试指标

| 指标 | M20 固定值 | 是否因接口改变 |
|---|---:|---|
| Accuracy | 0.744125 | 否 |
| AUC | 0.809837 | 否 |
| Log Loss | 0.523146 | 否 |
| Brier | 0.175656 | 否 |
| ECE10 | 0.015450 | 否 |

M19 十项正式目标继续通过 `10/10`，仍需改善的目标数为 `0`。

## 与外部模型相差多少

以下列出预测时点最接近的公开首杀后逻辑回归。完整七行及可比性标签见
`external_benchmark_comparison.md`。

| 外部工作 | 指标 | 本项目逻辑回归 | 外部 | 差值 |
|---|---|---:|---:|---:|
| CS156 - Round-Win Probability in CS2 via Economic Asymmetry | accuracy | 0.743405 | 0.682400 | +6.10 百分点 |
| CS156 - Round-Win Probability in CS2 via Economic Asymmetry | auc | 0.809059 | 0.760000 | +4.91 百分点 |

## 自动化验收

- 阻断检查通过：10/10；
- 自动化测试：131 项；
- XGBoost fit 调用：0；
- 模型性能改变：False。

## 使用命令

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_first_kill `
  --input examples\first_kill_snapshot.json `
  --model models\esta_full_m17\first_kill_xgboost_tuned.joblib `
  --calibrator models\esta_full_m18\first_kill_calibrator.joblib
```

## 下一阶段

M21 做首杀后 XGBoost 最终验收和一键复现。M21 通过后，首杀后 XGBoost
任务完成，再进入 LightGBM 同数据对照和实时胜率数据模块。

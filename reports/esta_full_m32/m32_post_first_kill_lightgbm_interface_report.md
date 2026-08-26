# M32 首杀后 LightGBM 单条预测接口验收

## 结论

M32 阻断检查 15/15 通过，状态为 `passed`，可以进入 M33。接口只加载 M29/M30 冻结工件，没有训练、调参或修改概率。

## 如何使用

在项目目录执行：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_first_kill_lightgbm `
  --input examples\first_kill_snapshot.json `
  --model models\esta_full_m29\post_first_kill_lightgbm_tuned.joblib `
  --calibrator models\esta_full_m30\post_first_kill_lightgbm_calibrator.joblib
```

CSV 只需把 `--input` 改为 `examples\first_kill_snapshot.csv`。需要保存结果时增加 `--output my_prediction.json`。非法输入返回退出码 2 和错误 JSON。

## 输入与工件合同

接口接收 31 个字段，其中购买基础字段 27 个、首杀事件字段 4 个；自动计算 9 个差值，形成 40 个原始特征并严格对齐 82 个编码列。地图 8 张、首杀武器 36 种、部署树 211 棵。

模型 SHA-256 为 `35ce17435a3716efcfdd49dd5ca13ff441e75c65512322627249e8920546a5b5`；校准器 SHA-256 为
`c5453403a25dfb03bbda131028fda7bdfde934840093de3e527ad2988c8043e5`。两者运行前后均未变化。校准器绑定同一模型与数据，并记录只用 validation 的 5 折结果选择 identity。

## 示例输出

示例基础 CT 概率为 0.7052604307，identity 后的 CT/T 概率为 0.7052604307 / 0.2947395693，预测方为 `CT`。JSON 与 CSV 的 CT 概率差为 0.000e+00。

示例输出只是单个快照的接口检查，不是测试集指标或投注建议。

## 非法输入

| 案例 | 已拒绝 | 错误数 | 首条错误 |
|---|---|---:|---|
| missing_first_kill_field | True | 2 | missing required first-kill fields: first_kill_time |
| invalid_first_kill_advantage | True | 1 | first_kill_advantage_ct must be -1 or 1; got 0 |
| first_kill_time_out_of_range | True | 1 | first_kill_time must be between 0 and 180 seconds; got 181 |
| invalid_headshot | True | 1 | first_kill_headshot must be a boolean or 0/1; got 2 |
| unknown_first_kill_weapon | True | 1 | first_kill_weapon 'Unknown Blaster' was not seen during training; choose one of ['AK-47', 'AUG', 'AWP', 'CZ75 Auto', 'Desert Eagle', 'Dual Berettas', 'FAMAS', 'Five-SeveN', 'G3SG1', 'Galil AR', 'Glock-18', 'HE Grenade', 'Incendiary Grenade', 'Knife', 'M4A1', 'M4A4', 'MAC-10', 'MAG-7', 'MP5-SD', 'MP7', 'MP9', 'Molotov', 'Negev', 'Nova', 'P2000', 'P250', 'P90', 'PP-Bizon', 'SCAR-20', 'SG 553', 'SSG 08', 'Tec-9', 'UMP-45', 'USP-S', 'XM1014', 'Zeus x27'] |
| unknown_map | True | 1 | map_name 'de_cache' was not seen during training; choose one of ['de_ancient', 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass', 'de_train', 'de_vertigo'] |
| inconsistent_derived_feature | True | 1 | score_diff_ct must equal ct_score - t_score; expected 1, got 99 |
| identifier_field | True | 1 | forbidden fields: series_id |
| target_leakage | True | 1 | forbidden fields: ct_win |
| future_second_kill | True | 1 | forbidden fields: second_kill_weapon |

## 冻结指标

| Accuracy | AUC | Log Loss | Brier | ECE10 |
|---:|---:|---:|---:|---:|
| 0.742926 | 0.808255 | 0.524063 | 0.176003 | 0.014191 |

M32 不重新计算或选择模型，以上五项与 M30/M31 完全一致。外部比较仍为 7 行，并逐字节复制自 M31。

## 验收与下一步

十类非法输入全部拒绝，CLI 成功/失败路径通过。自动化测试 264 项通过，源码编译通过。M33 将做首杀后 LightGBM 最终验收与一键复现；批量、HTTP、GUI、身份特征和实时胜率不在本阶段。

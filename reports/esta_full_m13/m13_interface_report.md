# M13 独立预测接口验收报告

## 阶段结论

状态：**passed**。M8 调优模型和 M10 校准选择均未重新训练，
所以 M9 固定测试集指标不变；本阶段只把模型封装成可校验、可重复使用的单回合接口。

## 输入与预处理

- 用户输入基础字段：27 个。
- 接口自动计算 CT-T 差值：9 个。
- 独热编码前模型字段：36 个。
- 按保存的模型列对齐后：43 个。
- 已知地图：de_ancient, de_dust2, de_inferno, de_mirage, de_nuke, de_overpass, de_train, de_vertigo。
- 时间点定义：购买结束、冻结时间结束、第一次交火之前。

训练阶段的 `prepare_features()` 被推理接口直接复用；地图类别编码完成后，
再按模型 bundle 中保存的 43 列重排，缺少的独热列补 0。

## 示例结果

- CT 胜率：`0.567700`
- T 胜率：`0.432300`
- 判定方：`CT`
- 两个概率之和：`1.000000000000`
- JSON/CSV 结果一致：`True`
- 校准方式：`uncalibrated`

概率表示模型在当前数据和特征下的估计，不代表比赛一定按较高概率一方获胜。

## 错误输入验收

| 错误类型 | 是否拒绝 | 返回错误数 | 首条信息 |
|---|---|---:|---|
| missing_required_field | True | 2 | missing required fields: round_num |
| wrong_numeric_type | True | 1 | ct_cash must be an integer; got '3500' |
| out_of_range | True | 2 | ct_helmets must be between 0 and 5; got 6 |
| unknown_map | True | 1 | map_name 'de_cache' was not seen during training; choose one of ['de_ancient', 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_overpass', 'de_train', 'de_vertigo'] |
| inconsistent_derived_feature | True | 1 | score_diff_ct must equal ct_score - t_score; expected 1, got 99 |

## 固定测试指标

| 指标 | M13 当前值 | 是否因接口阶段改变 |
|---|---:|---|
| Accuracy | 0.647411 | 否 |
| AUC | 0.727122 | 否 |
| Log Loss | 0.591733 | 否 |
| Brier Score | 0.205294 | 否 |
| ECE10 | 0.023198 | 否 |

## 与外部模型相差多少

差值固定为“我们的指标 - 外部报告指标”。以下只列预测时点最接近的公开结果；
数据集、年份与切分方式不同，所以这是参考差距，不是受控模型排名。

| 外部工作 | 指标 | 我们 | 外部 | 差值 |
|---|---|---:|---:|---:|
| Predicting the outcome of a round in CS:GO using a DNN | accuracy | 0.647411 | 0.679220 | -3.18 个百分点 |
| Predicting the outcome of a round in CS:GO using a DNN | log_loss | 0.591733 | 0.567860 | +0.023873 |

完整的可比性分组、来源链接和所有数值差见
`external_benchmark_comparison.md`。

## 使用命令

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round `
  --input examples\pre_round_snapshot.json `
  --model models\esta_full_m8_tuned\pre_round_xgb.joblib `
  --calibrator models\esta_full_m10\pre_round_calibrator.joblib
```

## 下一阶段

M14 做开局前 XGBoost 最终验收：从干净环境按文档复跑关键命令、整理未达目标
指标和剩余风险，然后决定先优化特征还是开始首杀后模型。

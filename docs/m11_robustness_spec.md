# M11 分组稳健性与错误分析验收

## 目的

确认开局前 XGBoost 的效果不是只来自某个地图或 LAN/online 来源，并分析模型
最有信心但预测错误的回合。M11 固定使用 M9 保存的测试概率，不训练或调参。

## 方法

- 地图、来源、回合阶段和装备差分组都报告回合数、系列赛数和五项指标。
- 每个分组内部按 `series_id` 整组 bootstrap 2,000 次计算 95% CI。
- 回合阶段固定为 1-10、11-20、21+。
- 装备差分为 T 巨大优势、T 中等优势、均势、CT 中等优势、CT 巨大优势。
- 高置信错误定义为预测方概率至少 0.80，但最终输掉回合。
- 首杀只作为预测完成后的事后诊断字段，绝不是开局前模型特征。

## LAN 与 Online

| 来源 | 回合 | 系列赛 | AUC | AUC 95% CI | Log Loss |
|---|---:|---:|---:|---:|---:|
| LAN | 1,855 | 37 | 0.731693 | [0.712587, 0.747736] | 0.587747 |
| Online | 2,317 | 42 | 0.722690 | [0.702349, 0.743164] | 0.594925 |

LAN 减 Online 的 AUC 为 `0.009003`，差值 95% CI 为
`[-0.018130, 0.036925]`。区间包含 0，且绝对差小于 0.04，没有可靠证据说明
模型只适合其中一种来源。

## 地图结果

| 地图 | 回合 | 系列赛 | AUC | AUC 95% CI |
|---|---:|---:|---:|---:|
| de_dust2 | 401 | 16 | 0.762526 | [0.715624, 0.802407] |
| de_overpass | 491 | 19 | 0.752237 | [0.709682, 0.793722] |
| de_vertigo | 364 | 14 | 0.736202 | [0.675655, 0.795194] |
| de_mirage | 706 | 28 | 0.731116 | [0.696060, 0.761059] |
| de_inferno | 1,015 | 36 | 0.710880 | [0.680989, 0.741383] |
| de_nuke | 718 | 27 | 0.697042 | [0.662218, 0.728324] |
| de_ancient | 338 | 13 | 0.695993 | [0.658606, 0.724378] |
| de_train | 139 | 6 | 0.773839 | [0.716134, 0.829088] |

样本数至少 300 的七张地图中，最低点估计为 Ancient 的 0.695993，达到 0.69
目标。但 Ancient 和 Nuke 的 CI 下界低于 0.67，因此只能说点估计达标，不能
声称每张地图都已统计上稳定超过最低门槛。Train 只有 6 个系列赛，区间较宽，
不能根据最高点估计认定它表现最好。

## 回合阶段

| 阶段 | 回合 | AUC | AUC 95% CI |
|---|---:|---:|---:|
| 1-10 | 1,590 | 0.735876 | [0.712105, 0.756892] |
| 11-20 | 1,583 | 0.728889 | [0.709018, 0.747074] |
| 21+ | 999 | 0.703763 | [0.670574, 0.738344] |

后期回合点估计较低，但三个区间重叠，现阶段不能确认存在真实阶段退化。

## 高置信错误

测试集中共有 90 个高置信错误，按预测置信度审查前 30 个：

- 30/30 都是预测热门方拥有至少 5,000 的装备价值优势，最终仍输掉回合。
- 22/30 的预测热门方输掉首杀。
- 8/30 的预测热门方拿到首杀后仍被翻盘。
- 全部 90 个错误中，61 个输掉首杀，29 个拿到首杀后仍输回合。

这说明当前高置信概率主要由装备优势驱动，而静态购买快照无法描述交火质量、
位置、投掷物效果和后续决策。首杀是重要的事后解释变量，也支持后续单独开发
“首杀后胜率”；这些模式不是经过因果识别的确定原因。

## 验收结论

- LAN/online AUC 差 <= 0.04：通过。
- 样本数至少 300 的地图最低 AUC >= 0.67：通过。
- 样本数至少 300 的地图最低 AUC >= 0.69：点估计通过。
- 所有地图点估计的 95% CI 下界都 >= 0.67：未达到。
- 至少审查 30 个高置信错误：通过。
- 小样本分组同时显示样本量、系列赛数和置信区间：通过。

## 输出文件

```text
reports\esta_full_m11\metrics_by_map_with_ci.csv
reports\esta_full_m11\metrics_by_source_with_ci.csv
reports\esta_full_m11\metrics_by_round_stage_with_ci.csv
reports\esta_full_m11\metrics_by_equipment_band_with_ci.csv
reports\esta_full_m11\source_auc_gap.csv
reports\esta_full_m11\all_high_confidence_errors.csv
reports\esta_full_m11\reviewed_top30_errors.csv
reports\esta_full_m11\top30_error_review.md
reports\esta_full_m11\error_pattern_summary.csv
reports\esta_full_m11\map_auc_with_ci.png
reports\esta_full_m11\error_pattern_counts.png
reports\esta_full_m11\m11_summary.json
reports\esta_full_m11\m11_robustness_report.md
```

## 复现命令

```powershell
cd C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m11_robustness --predictions reports\esta_full_m9\test_predictions.csv --data data\processed\esta_full\pre_round.parquet --kills data\interim\esta_full\kills.parquet --report-dir reports\esta_full_m11 --bootstrap-samples 2000 --seed 42 --review-cases 30
```

下一阶段是 M12 模型解释：补 gain、Permutation Importance 和 SHAP，并检查
前 20 个特征中是否存在不合理信号。

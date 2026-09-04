# ROUNDCAST v1 — 三个真实案例的只读预检

核查日期：2026-09-05。用途：支撑第一版实施计划和后续参考概率断言。
本轮未加载模型、未执行新的推理、未训练或改写数据；表中概率来自正式已保存结果，不代表网页已接通。

## 1. 对齐资格

唯一键为 `series_id + game_id + round_id`。

- 购买结束 XGBoost/LightGBM 测试表各 4,172 行；首杀后各 4,170 行。
- 四表重复键均为 0；一对一共同交集为 4,170 回合。
- 四表交集标签差异为 0，与两份正式 parquet 标签差异也为 0。
- 交集在两个时点的 parquet 均为 test；三例在 M14/M21/M27/M33 的冻结 split_assignments.csv 中也均为 test。
- 三例两时点的 27 个购买结束基础字段一致。
- 三例各自通过现有 validate_snapshot / validate_first_kill_snapshot。允许类别读取已冻结 model_contract_audit.json，没有反序列化模型。

候选清单在实现时仍须重新检查可信文件哈希，然后真正调用四个 Predictor，不能把预检报告当成实时推理结果。

## 2. 参考 CT 获胜概率

以下为文件中的参考值；页面展示时再转百分数和舍入。

| 案例 | 地图与回合 | XGB 购买结束 | LGBM 购买结束 | XGB 首杀后 | LGBM 首杀后 | 真实赢家 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | Ancient R4 | 0.6291529536247253 | 0.6591145000070254 | 0.7656537294387817 | 0.736277887665323 | CT |
| B | Mirage R11 | 0.7336581349372864 | 0.7735134884219927 | 0.41505882143974304 | 0.3991340729586205 | T |
| C | Nuke R17 | 0.9748938679695129 | 0.9704182077374078 | 0.9854957461357117 | 0.9898388852132408 | T |

原始 CSV 与不同浮点读取库可能出现末位舍入差，真实推理验收仍使用已批准的绝对误差 `1e-8`，不可仅比较页面舍入后的百分数。

## 3. 固定回合身份与事实

### 案例 A

- series_id：`1d78bfe0-e598-4a95-b5c0-2227e976dc5d`
- game_id：`online:946b0351-728d-41c6-9964-9b20f21df71d`
- round_id：`online:946b0351-728d-41c6-9964-9b20f21df71d_4`
- 回合前 CT:T 比分为 2:1，装备价值为 23,900:14,650。
- CT 在记录的 25.2578125 秒以 AK-47 爆头取得首杀。
- 演示用途：普通正确预测，四个组合均预测 CT 且实际 CT 胜。

### 案例 B

- series_id：`bdada140-b48c-4be0-93c3-ce2cd7973eed`
- game_id：`online:e941205c-0716-43ea-9836-3ad863fc2193`
- round_id：`online:e941205c-0716-43ea-9836-3ad863fc2193_11`
- 回合前 CT:T 比分为 8:2，装备价值为 26,200:10,000。
- T 在记录的 65.953125 秒以 Desert Eagle 非爆头取得首杀。
- 演示用途：两种算法都从预测 CT 改为预测 T，最终 T 胜；这是两时点预测变化，不是隔离了首杀的因果影响。

### 案例 C

- series_id：`90da2c53-5a02-4f16-8abe-f2235da5ffbd`
- game_id：`online:478d378e-e7c1-4d64-a3f3-679ee18f27b5`
- round_id：`online:478d378e-e7c1-4d64-a3f3-679ee18f27b5_17`
- 回合前 CT:T 比分为 8:8，装备价值为 19,650:1,500。
- CT 在记录的 28.6953125 秒以 MP9 爆头取得首杀。
- 演示用途：两算法首杀后均给 CT 超过 98% 概率，但实际 T 胜，说明概率并非保证。未检查后续事件，不能编造失败原因。

首杀时间沿用正式输入字段的定义；不据此声称已取得精确帧级回放同步。

## 4. 来源文件

以下均为项目内相对路径：

| 用途 | 文件与字段 |
| --- | --- |
| XGB 购买结束参考概率 | reports/esta_full_m10/calibrated_test_predictions.csv，probability_uncalibrated |
| XGB 首杀后参考概率 | reports/esta_full_m18/calibrated_test_predictions.csv，probability_uncalibrated |
| LGBM 购买结束参考概率 | reports/esta_full_m27/replayed_test_predictions.csv，ct_win_probability |
| LGBM 首杀后参考概率 | reports/esta_full_m33/replayed_test_predictions.csv，ct_win_probability |
| 购买结束输入 | data/processed/esta_full/pre_round.parquet |
| 首杀后输入 | data/processed/esta_full/first_kill.parquet |
| 类别白名单审计 | reports/esta_full_m32/model_contract_audit.json |
| 输入验证器 | src/csdemo/predict_pre_round.py；src/csdemo/predict_first_kill.py |

## 5. 选择限制

A 从两阶段两算法均正确且非极端概率的样本中选择；B 从两算法时点变化均至少 20 个百分点且预测方反转的样本中选择；C 从两算法首杀后均向实际错误方给至少 80% 概率的样本中选择。

这是依据结果刻意挑选的三个演示案例，不是随机抽样，也不形成新的性能结论。类别名称与真实赢家只用于预检/主动揭示后的说明；默认页面名称为中性的案例 A/B/C，不能提前剧透。

API 模型输入只使用允许的 27/31 个基础字段；身份、split、真实赢家和参考概率独立存放，不作为模型特征。来源 ID 不需要隐藏为秘密，但不得混入特征。

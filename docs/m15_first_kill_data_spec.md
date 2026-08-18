# M15 首杀后样本修复与验收规格

## 1. 阶段目标

M15 不训练新模型。它负责把“购买完毕、发生首个有效敌方击杀之后”的样本修好并验收，
让 M16 可以在可信数据上比较 Dummy、逻辑回归和 XGBoost。

本阶段必须解决两个已发现的问题：

1. 旧实现按 `time` 最小值选首杀；ESTA 的 `seconds` 在回合后段可能重置，不能用于排序。
2. 旧样本没有生成 `ct_alive_after_fk`、`t_alive_after_fk` 和
   `alive_diff_ct_after_fk`。

## 2. 任务定义

- 预测目标：该回合最终是否由 CT 获胜，`ct_win=1` 表示 CT 获胜。
- 预测时点：购买结束、交火开始后，首个有效敌方击杀刚刚发生。
- 首杀定义：同一 `series_id + game_id + round_id` 内，tick 最小的有效敌方击杀。
- 有效敌方击杀：击杀方和死亡方必须分别为 `CT` 或 `T`，且双方不同。
- 初始状态：只接受购买结束快照为 5v5 的回合。
- 允许信息：M14 的购买结束特征，以及该首杀的时间、双方、武器和爆头信息。
- 禁止信息：第二次及后续击杀、炸弹结果、回合结束状态、胜方身份衍生字段。
- 队伍和选手身份：本阶段仍不加入。

`is_first_kill` 只用于审计，不作为首杀选择的唯一依据。标准化数据已排除自杀和队友击杀；
当原始首杀被排除或标记缺失时，仍按最小 tick 选择首个保留下来的有效敌方击杀。

## 3. 主键和切分合同

- 样本主键固定为 `series_id + game_id + round_id`，三列必须同时参与关联。
- 每个主键最多一条首杀后样本。
- 没有有效击杀的回合不进入本任务，并在报告中单独计数。
- 直接读取 M14 的 `split_assignments.csv`，不重新随机切分。
- 一个系列赛的所有地图和回合只能出现在 train、validation、test 中的一个集合。
- 目标比例仍为系列赛级 70% / 20% / 10%。

## 4. 输出字段

输出保留 M14 的购买结束特征，并新增：

| 字段 | 含义 |
|---|---|
| `first_kill_time` | 首杀事件在 ESTA 中记录的秒数，仅作模型候选特征，不用于事件排序 |
| `first_kill_is_ct` | 击杀方是否为 CT |
| `first_death_is_ct` | 首位死亡者是否为 CT |
| `first_kill_headshot` | 首杀是否爆头 |
| `first_kill_weapon` | 首杀武器 |
| `ct_alive_after_fk` | 首杀后 CT 存活人数 |
| `t_alive_after_fk` | 首杀后 T 存活人数 |
| `alive_diff_ct_after_fk` | 首杀后 CT 存活人数优势 |
| `first_kill_advantage_ct` | 从首杀双方计算的 CT 优势，取值只能为 -1 或 1 |

其中阵营、死亡方、存活人数和优势字段是确定性冗余。M15 保留它们用于审计和解释；M16
训练时必须用显式特征组做控制变量实验，不能把冗余字段带来的重复权重误认为新信息。

## 5. 阻塞验收条件

M15 只有在以下检查全部通过后才能进入 M16：

1. 首杀样本完整主键重复数为 0。
2. 样本中的事件与同一完整主键内最小 tick 的有效击杀完全一致。
3. 孤立首杀样本和跨地图关联均为 0。
4. 击杀双方非法或相同的样本为 0。
5. 首杀后状态只能是 5v4 或 4v5，且优势字段计算一致。
6. 核心首杀字段缺失值为 0。
7. 每个系列赛的 split 与 M14 清单完全一致，跨 split 交集为 0。
8. 自动化测试全部通过。

当前标准输入有 41,074 个 5v5 回合。已知 47 个回合没有有效敌方击杀，因此在源数据不变时，
预期生成 41,027 条首杀后样本；这个数值是数据指纹，不是写死在通用代码中的规则。

## 6. 阶段产物

- `data/processed/esta_full/first_kill.parquet`
- `reports/esta_full_m15/m15_summary.json`
- `reports/esta_full_m15/m15_first_kill_data_report.md`
- `reports/esta_full_m15/m15_checks.csv`
- `reports/esta_full_m15/split_summary.csv`
- `reports/esta_full_m15/excluded_rounds.csv`
- `reports/esta_full_m15/external_benchmark_comparison.md`

M15 没有新模型，因此不制造新的 Accuracy、AUC、Log Loss 或 Brier 数值。外部比较文件要明确
记录“本阶段不适用”；M16 训练出正式首杀后基线后再报告数值差。

## 7. 后续阶段

- M16：同一修复样本和同一 split 上比较 Dummy、逻辑回归、未经调参的 XGBoost。
- M17：只看验证集做单变量调参，并在方案冻结后评估一次测试集。
- M18：校准、稳健性、解释和首杀后单条预测接口。

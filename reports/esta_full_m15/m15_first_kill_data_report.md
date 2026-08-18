# M15 首杀后样本修复与验收报告

## 阶段决定

验收状态：**passed**。
M15 不训练模型；通过后只表示修复主键和首杀事件的数据可以进入 M16 基线实验。

## 本次修复

- 首杀从“`time` 最小”改为同一完整主键内“`tick` 最小”的有效敌方击杀。
- 使用 `series_id + game_id + round_id` 关联，阻止同系列赛不同地图串行。
- 新增首杀后的 CT/T 存活人数和人数差，状态必须为 5v4 或 4v5。
- 直接复用 M14 的 782 个系列赛 split，不重新抽签。

## 数据结果

- 输入回合：41,074；击杀事件：268,640。
- 首杀后样本：41,027；排除无有效敌方击杀回合：47。
- 覆盖系列赛：782；地图 demo：1,558。
- 按秒数与按 tick 选择不一致：14,357 回合。
- 最小 tick 事件没有 ESTA 首杀标记：70 回合；使用明确定义的 tick 兜底。
- 原始首杀标记不在最小 tick：0 回合。
- 修复后首杀时间小于等于 0：0 回合。

## 修复前后

旧产物与新产物主键都为 41,027 条；新增主键 0，移除主键 0。
首杀事件字段发生变化：**14,357 条（34.99%）**。
新增字段：`alive_diff_ct_after_fk, ct_alive_after_fk, t_alive_after_fk`。

| 字段 | 变化行数 |
|---|---:|
| `first_kill_time` | 14,357 |
| `first_kill_is_ct` | 6,697 |
| `first_death_is_ct` | 6,697 |
| `first_kill_headshot` | 6,694 |
| `first_kill_weapon` | 9,669 |
| `first_kill_advantage_ct` | 6,697 |

## 阻塞检查

| 检查 | 结果 |
|---|---|
| `required_columns` | PASS |
| `unique_repaired_key` | PASS |
| `sample_coverage` | PASS |
| `event_linkage` | PASS |
| `core_values_present` | PASS |
| `initial_5v5_state` | PASS |
| `post_kill_state` | PASS |
| `label_linkage` | PASS |
| `split_manifest` | PASS |
| `split_isolation` | PASS |
| `normalized_kill_sides` | PASS |
| `automated_tests` | PASS |

## 固定切分

| split | 系列赛 | 地图 | 样本 | 系列赛占比 | CT 胜率 |
|---|---:|---:|---:|---:|---:|
| train | 547 | 1,083 | 28,489 | 69.95% | 0.5436 |
| val | 156 | 316 | 8,368 | 19.95% | 0.5359 |
| test | 79 | 159 | 4,170 | 10.10% | 0.5247 |

## 特征说明

`first_kill_is_ct`、`first_death_is_ct`、两侧存活人数和优势字段彼此可以确定，
所以它们不是五份独立信息。M16 会固定其他条件，分别比较“仅首杀阵营”、
“阵营加时间/爆头/武器”等特征组，避免把确定性冗余当成性能提升。

## 与外部模型相差多少

M15 没有新模型，因此首杀后差值为“不适用”。最近有效的 M14 开局前结果仍是：
Accuracy 比最接近的公开 DNN 低 3.18 个百分点，Log Loss 高 0.023873。
两者数据和切分不同；而且 M15 的预测时点更晚，不能直接代替首杀后比较。
历史首杀 XGBoost 测试 AUC 0.774750 使用旧主键和旧事件选择，继续标记为无效历史值。

## 可复现产物

- 数据 SHA-256：`06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492`。
- 数据字节数：1,479,873。
- 完整检查明细：`m15_checks.csv`。
- 47 个排除回合：`excluded_rounds.csv`。
- 外部比较状态：`external_benchmark_comparison.csv/.md`。

运行命令：

```powershell
.\scripts\run_first_kill_data_stage.ps1
```

## 下一阶段

M16 在这份固定样本上先训练 Dummy 和逻辑回归，再训练未经调参的 XGBoost。
三者使用完全相同的 train/validation/test 行和特征组；测试集在方案冻结前不参与选择。

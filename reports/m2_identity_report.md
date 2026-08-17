# M2 数据主键修复验收报告

验收日期：2026-08-15

## 结论

M2 已完成。ESTA 的系列赛、地图 demo 和回合现在使用不同层级的标识：

```text
series_id = ESTA matchId，用于 70/20/10 分组切分
game_id   = subset + demo 文件 UUID，用于唯一识别一张地图
round_id  = game_id + round_num，用于唯一识别一个回合
```

## 原始数据

| 来源 | demo 数量 |
|---|---:|
| LAN | 680 |
| Online | 878 |
| 合计 | 1,558 |

全量解析成功写出：

| 表 | 行数 |
|---|---:|
| rounds | 41,076 |
| kills | 268,640 |

## 主键验收

| 检查 | 结果 |
|---|---:|
| `series_id` 数量 | 782 |
| `game_id` 数量 | 1,558 |
| `round_id` 数量 | 41,076 |
| 完整主键重复 | 0 |
| `round_id` 重复 | 0 |
| 同一主键标签冲突 | 0 |
| 孤立击杀回合 | 0 |
| 回合主键缺失 | 0 |
| 击杀主键缺失 | 0 |

49 个回合没有符合当前规则的有效击杀，所以首杀后样本比开局前样本少 49 条。这些回合没有被错误关联到同一系列赛的其他地图。

## 训练表验收

| 项目 | 开局前 | 首杀后 |
|---|---:|---:|
| 样本数 | 41,076 | 41,027 |
| 系列赛数 | 782 | 782 |
| 地图 demo 数 | 1,558 | 1,558 |
| 重复主键 | 0 | 0 |

开局前数据划分：

| split | 系列赛数 | 回合数 |
|---|---:|---:|
| train | 547 | 28,522 |
| val | 156 | 8,382 |
| test | 79 | 4,172 |

三个集合之间的系列赛交集均为 0。开局前与首杀后数据对全部 782 个系列赛的 split 映射完全一致。

## 自动化验证

5 个 `unittest` 测试通过，覆盖：

- 同一系列赛不同地图生成不同 `game_id` 和 `round_id`。
- 首杀事件不会跨地图关联。
- 同一系列赛的所有地图保持相同 split。
- 重复回合身份会被拒绝。
- 示例表遵守新的主键契约。

## 产物位置

```text
data/interim/esta_full/rounds.parquet
data/interim/esta_full/kills.parquet
data/processed/esta_full/pre_round.parquet
data/processed/esta_full/first_kill.parquet
```

旧 M1 产物没有删除，保存在：

```text
data/interim/esta_full_legacy_m1
data/processed/esta_full_legacy_m1
models/esta_full_legacy_m1
reports/esta_full_legacy_m1
```

## 下一模块

进入 M4 数据质量。先检查冻结结束帧、比分关系、人数、武器和手雷计数等异常，再重新训练正式开局前 XGBoost。

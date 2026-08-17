# M4.1 inventory/utility 异常来源报告

检查日期：2026-08-16

后续状态：M4.2 已于 2026-08-16 按本报告建议完成规范化和全量验收，详见 `m4_2_normalization_report.md`。

## 目标

追查 `weapon_count_exceeds_five` 和 `grenade_count_exceeds_twenty` 的来源，确认异常发生在项目的聚合代码、玩家列表，还是 ESTA 原始解析帧。

本次新增 `src/csdemo/inspect_snapshot.py`。它读取指定 `.json.xz` 和回合号，选择最接近 `freezeTimeEndTick` 的帧，并同时输出：

- 队伍上报的存活人数和 `totalUtility`。
- 玩家数量、Steam ID、存活状态和完整 inventory。
- 从存活玩家 inventory 重新计算的步枪数和手雷数。
- 重复玩家和重复 inventory 项。

## 样本证据

| demo / 回合 | 原异常 | 原始帧证据 | 结论 |
|---|---:|---|---|
| `lan:03e1f233-579c-462d-ac0e-1635d4718ef8` / 20 | T 步枪 6 | 5 名不同玩家；`apEX` 同时有 `AK-47` 和 `Galil AR` | 单个玩家含两把不同主武器 |
| `lan:05a57319-2185-4a18-9ad8-89b078bc48cb` / 32 | T 步枪 7 | 5 名不同玩家；`hampus`、`Plopski` 各有两条相同的 `AK-47` | 单个玩家 inventory 含重复主武器项 |
| `lan:85ec4b01-3749-44e8-b369-f7ec14738ee5` / 18 | T 手雷 21 | 队伍 `totalUtility=21`，inventory 重算也是 21；`Liki` 有 5 颗手雷，其中 `Smoke Grenade` 重复两条 | 上游帧已含重复手雷项 |

三个检查帧分别位于冻结结束 tick 后 57、28、58 tick，均是项目当前 `nearest_frame` 规则实际选中的帧。

## 根因结论

1. 没有发现重复 Steam ID；异常不是队伍 `players` 列表重复造成的。
2. 项目的 inventory 累加逻辑准确复现了 ESTA 原始帧中的异常值；异常不是 M2 主键、表连接或切分造成的。
3. 步枪异常来自单个玩家 inventory 中的重复主武器或多把主武器。
4. `totalUtility` 的语义已确认是队伍手雷物品总数；数值 21 来自玩家 inventory 中的重复手雷，并非另一种经济指标。
5. 因此数据源层级的根因是 ESTA/Awpy 已解析帧中的 inventory 异常。原始 `.dem` 是否同样异常不在当前 ESTA JSON 能证明的范围内。

## 下一阶段建议（尚未实施）

M4.2 建议采用可解释、影响范围小的规则：

- 每名存活玩家最多贡献 1 把步枪；同名武器按玩家去重。
- 手雷数从玩家 inventory 重算，每名存活玩家最多贡献 4 颗。
- 排除已确认不代表购买结束状态的 Vertigo 第 5、6 回合。
- 修正后重新生成全量质量报告，目标是上述三类警告归零，同时保持主键、标签和切分检查通过。

这些规则必须先写测试，再修改特征提取；本报告阶段未修改训练数据。

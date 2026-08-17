# 购买完毕、交火前 XGBoost：从初始版本到当前版本的完整报告

报告日期：2026-08-16  
项目路径：`C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction`

## 1. 报告目的

本报告只讨论第一个课题：在 CS:GO 每回合购买完毕、冻结时间结束、正式交火前，
预测 CT 赢得本回合的概率。

报告比较两个正式保留的阶段：

1. 一开始直接从 ESTA 数据提取特征并训练的 XGBoost。
2. 完成主键、数据质量、inventory 规范化和 M6 特征修复后的当前 XGBoost。

首杀后胜率、实时胜率和 LightGBM 不在本报告范围内。

## 2. 问题定义

### 2.1 预测时点

每个回合选择距离 `freezeTimeEndTick` 最近的一帧。此时双方已经购买完装备，
但还没有发生正式交火。

### 2.2 标签

```text
ct_win = 1：CT 最终赢得该回合
ct_win = 0：T 最终赢得该回合
```

模型输出：

```text
CT 胜率 = model.predict_proba(...)[:, 1]
T 胜率  = 1 - CT 胜率
```

### 2.3 禁止使用的信息

开局模型不允许使用首杀、伤害、死亡、下包、拆包、回合结束原因等未来信息。

## 3. 数据与运行环境

| 项目 | 当前设置 |
|---|---|
| 原始数据 | ESTA LAN + Online |
| 原始数据路径 | `C:\project1\data\esta` |
| LAN demo | 680 |
| Online demo | 878 |
| demo 合计 | 1,558 |
| 系列赛 | 782 |
| Conda 环境 | `C:\Users\admin\11\envs\game` |
| Python | 3.10.20 |
| XGBoost | 3.2.0 |
| 切分方式 | 按 `series_id` 做 70/20/10 |
| 随机种子 | 42 |

同一系列赛的所有地图只能进入同一个集合，避免系列赛信息泄漏到验证集或测试集。

## 4. 初始直接训练版本

### 4.1 初始流程

最初版本已经能够完成基本链路：

```text
ESTA JSON.XZ
  -> 选择 freezeTimeEndTick 最近帧
  -> 每回合生成一行特征
  -> 按系列赛切分
  -> 地图独热编码
  -> 训练 XGBoost
  -> 输出 CT 胜率
```

初始模型参数：

| 参数 | 值 |
|---|---:|
| `n_estimators` | 500 |
| `max_depth` | 4 |
| `learning_rate` | 0.03 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.85 |
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |

### 4.2 初始数据和特征

| 项目 | 初始版本 |
|---|---:|
| 开局前样本 | 41,076 |
| train | 28,522 |
| validation | 8,382 |
| test | 4,172 |
| 模型前特征 | 39 |
| 独热编码后特征 | 47 |

主要特征包含地图、回合号、比分、双方经济、护甲、头盔、拆弹器、AK、M4、
AWP、步枪、冲锋枪、投掷物数量，以及双方差值。

### 4.3 初始模型效果

| 集合 | Accuracy | Log Loss | AUC |
|---|---:|---:|---:|
| train | 0.6873 | 0.5605 | 0.7718 |
| validation | 0.6502 | 0.5984 | 0.7144 |
| test | 0.6512 | 0.5941 | 0.7209 |

补充测试指标：

| 指标 | 初始版本 |
|---|---:|
| Test Brier Score | 0.20624 |
| Test ECE（10 桶） | 0.02158 |
| Train-Val AUC 差 | 0.0574 |

初始模型明显优于只预测固定 CT 胜率的常数模型，已经具备真实的排序和概率预测能力。
但当时的数据还没有通过正式质量验收，所以只能作为历史基线。

## 5. 初始版本发现的问题

### 5.1 回合身份不唯一

旧定义把 ESTA `matchId` 同时当成系列赛和地图标识：

```text
旧 round_id = matchId + round_num
```

一个系列赛通常包含多张地图，所以不同地图的相同回合号会得到相同 `round_id`。
检查发现：

| 问题 | 数量 |
|---|---:|
| 受重复键影响的记录 | 18,955 |
| 存在标签冲突的重复键 | 8,179 |

ID 没有进入模型特征，而且旧版本已经按 `matchId` 分组切分，因此开局前指标仍可作为
临时参考；但回合无法可靠追踪，首杀事件也可能跨地图错误关联，不能作为正式数据结构。

### 5.2 inventory 异常直接进入队伍统计

原始 ESTA/AWPY 帧中存在重复武器条目，或者同一玩家同时记录多把步枪。直接求和造成：

| 质量问题 | 初始数量 |
|---|---:|
| 单方步枪人数大于 5 | 83 |
| 单方投掷物数量大于 20 | 6 |

这些值不是正常的 5v5 开局队伍状态。

### 5.3 两个快照不符合 5v5 定义

有 2 个回合在最接近冻结结束的帧中不是 5v5，并伴随比分/回合号关系异常。
它们不符合“购买完毕、交火前”的项目定义。

### 5.4 M4A1-S 特征没有被解析

ESTA 将 M4A1-S 的 inventory 名称记录为 `M4A1`，旧映射只识别 `M4A1-S`，
因此 `ct_m4a1_s` 和 `t_m4a1_s` 两列一直为 0。

### 5.5 没有信息量的模型列

通过 5v5 质量门禁后，以下开局字段必然恒定：

```text
ct_alive = 5
t_alive = 5
alive_diff_ct = 0
```

独热编码还生成过一个始终为 0 的 `map_name_nan`。这些列没有预测信息。

## 6. 修改过程

### 6.1 M2：重新定义三级身份

当前定义：

```text
series_id = ESTA matchId              用于系列赛级切分
game_id   = subset + demo 文件 UUID   唯一表示一张地图
round_id  = game_id + round_num       唯一表示一个回合
```

修复结果：重复主键 0、标签冲突 0、孤立击杀 0；782 个系列赛在 train、validation、
test 之间交集为 0。

### 6.2 M4.1：追查异常来源

使用原始 `.json.xz` 检查异常回合，确认步枪和投掷物超限来自原始帧的重复 inventory
或多主武器记录，不是 pandas 聚合、主键关联或切分造成的错误。

### 6.3 M4.2：按玩家规范化

实施规则：

1. 同一玩家的同名武器只计一次。
2. 每名玩家最多贡献一把普通步枪。
3. 每名玩家最多贡献四颗投掷物。
4. 队伍投掷物从玩家 inventory 重算，不直接信任 `totalUtility`。
5. 冻结结束最近帧不是 5v5 时，排除该回合及其击杀。

修复后：

| 检查 | 当前结果 |
|---|---:|
| 标准开局前回合 | 41,074 |
| 非 5v5 快照 | 0 |
| 单方步枪大于 5 | 0 |
| 单方投掷物大于 20 | 0 |
| 重复 `round_id` | 0 |
| 质量 error | 0 |
| 质量 warning | 0 |

剩余 47 个没有有效击杀的回合只影响首杀后任务，不影响开局前模型。

### 6.4 M6：修复并整理模型特征

完成内容：

1. 将 ESTA `M4A1` 正确映射到 `m4a1_s`。
2. 从开局模型中删除恒定的三个人数特征。
3. 删除全零的 `map_name_nan` 编码列。
4. 建立包含来源、单位、范围和可用时间的特征字典。
5. 为 10 个 CT-T 差值增加单元测试。
6. 验证类别词表由训练集决定，验证/测试列严格对齐。
7. 完成比分、经济、武器、地图、防具道具和差值等消融实验。

修复后的 M4A1-S 分布：

| 特征 | 非零回合比例 | 最大值 |
|---|---:|---:|
| `ct_m4a1_s` | 45.74% | 5 |
| `t_m4a1_s` | 3.45% | 3 |

当前正式特征为 36 个模型前字段，地图独热编码后为 43 列，没有缺失列或常量列。

## 7. 当前正式 XGBoost

### 7.1 当前数据

| 集合 | 系列赛 | 回合数 | CT 胜率 |
|---|---:|---:|---:|
| train | 547 | 28,522 | 0.5429 |
| validation | 156 | 8,380 | 0.5351 |
| test | 79 | 4,172 | 0.5244 |

### 7.2 当前效果

| 集合 | Accuracy | Log Loss | AUC |
|---|---:|---:|---:|
| train | 0.6893 | 0.5598 | 0.7730 |
| validation | 0.6516 | 0.5981 | 0.7150 |
| test | 0.6462 | 0.5938 | 0.7220 |

补充测试指标：

| 指标 | 当前版本 |
|---|---:|
| Test Brier Score | 0.20609 |
| Test ECE（10 桶） | 0.02354 |
| Train-Val AUC 差 | 0.0580 |

## 8. 初始与当前对比

测试集保持为相同的 4,172 个回合，标签 CT 胜率均为 0.5244。

| Test 指标 | 初始 | 当前 | 当前减初始 | 判断 |
|---|---:|---:|---:|---|
| Accuracy | 0.6512 | 0.6462 | -0.0050 | 0.5 阈值分类略降 |
| AUC | 0.7209 | 0.7220 | +0.0011 | 排序能力小幅提高 |
| Log Loss | 0.5941 | 0.5938 | -0.0004 | 概率质量小幅提高 |
| Brier Score | 0.20624 | 0.20609 | -0.00015 | 小幅提高 |
| ECE（10 桶） | 0.02158 | 0.02354 | +0.00196 | 略降，但仍低于 0.03 目标 |
| Train-Val AUC 差 | 0.0574 | 0.0580 | +0.0006 | 仍有过拟合 |

这次修改没有带来很大的分数跃升，但这是合理结果。修改的主要目标是删除错误样本、
修复错误特征、建立唯一身份和质量闸门。当前 AUC 和 Log Loss 没有下降，说明在提高
数据可信度的同时保留了模型预测能力。

Accuracy 使用固定 0.5 阈值，容易受概率分布和类别比例影响。本项目输出的是胜率，
因此现阶段应优先看 AUC、Log Loss、Brier Score 和校准，而不是只看 Accuracy。

## 9. M6 消融结论

公平消融固定使用全部训练行和特征列，避免随机列采样干扰。

| 删除内容 | Test AUC 变化 | 当前决定 |
|---|---:|---|
| 比分 | -0.0037 | 保留 |
| 武器 | -0.0030 | 保留 |
| 地图 | -0.0026 | 保留 |
| 所有差值 | -0.0023 | 保留 |
| 装备价值 | -0.0014 | 保留并继续改进 |
| 现金 | -0.0010 | 贡献较弱，继续观察 |
| 防具道具 | -0.0010 | 保留并拆分道具类型 |
| 全部经济 | +0.0023 | 存在冗余，不直接删除 |

去掉全部经济时 AUC 略升，但验证 Log Loss 变差。经济、武器和防具高度相关，
因此下一步应重构经济表达并重复验证，而不是根据一次消融直接永久删除。

## 10. 当前达到的目标

| 验收项 | 当前状态 |
|---|---|
| 购买结束、交火前快照 | 已实现 |
| 只使用预测时可获得的特征 | 已实现 |
| 系列赛级 70/20/10 切分 | 已实现 |
| train/val/test 系列赛交集为 0 | 已实现 |
| 主键重复和标签冲突为 0 | 已实现 |
| 数据质量 error/warning 为 0 | 已实现 |
| 特征字典 | 已完成 |
| 特征和解析单元测试 | 22 个全部通过 |
| Test AUC 最低目标 >= 0.70 | 已达到：0.7220 |
| Test AUC 阶段目标 >= 0.73 | 尚未达到 |
| Test Log Loss 最低目标 <= 0.61 | 已达到：0.5938 |
| Test Log Loss 阶段目标 <= 0.58 | 尚未达到 |
| Train-Val AUC 差 <= 0.05 | 尚未达到：0.0580 |

## 11. 当前限制

1. 还没有保存 Dummy 和逻辑回归的统一 M7 对照结果。
2. 当前 XGBoost 固定训练 500 棵树，还没有使用验证集 early stopping。
3. Train-Val AUC 差为 0.0580，说明仍有过拟合。
4. 概率校准只进行了指标检查，还没有 Platt 或 isotonic 校准实验。
5. 当前没有战队和选手身份，模型预测的是通用局面强弱。
6. 部分地图测试样本较少，分地图 AUC 还没有置信区间。

## 12. 后续顺序

### 下一步 M7：简单模型对照

在相同特征、相同切分和相同指标上保存：

1. 固定 CT 胜率的 Dummy 基线。
2. 逻辑回归基线。
3. 当前 XGBoost。

### 随后 M8：XGBoost 稳定训练

1. 加入 early stopping。
2. 保存最佳树数量和完整参数。
3. 调节 `max_depth`、`min_child_weight`、`subsample`、`colsample_bytree`、
   `reg_alpha` 和 `reg_lambda`。
4. 主要根据验证集选择方案，测试集只用于最终验收。

### 后续特征实验

优先尝试经济占比、购买档位、投掷物类型、细分武器、手枪局/半场/加时标记，
以及低 AUC 地图中的地图与经济/武器交互。战队和选手身份最后单独实验。

## 13. 重要产物

```text
当前训练数据：data\processed\esta_full\pre_round.parquet
当前模型：models\esta_full_m6\pre_round_xgb.joblib
当前指标：reports\esta_full_m6\pre_round_xgb_metrics.csv
M6 详细报告：reports\esta_full_m6\m6_feature_report.md
特征字典：docs\m6_feature_dictionary.md
历史模型：models\esta_full_legacy_m1\pre_round_xgb.joblib
历史指标：reports\esta_full_legacy_m1\pre_round_xgb_metrics.csv
```

## 14. 复现命令

```powershell
cd C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction

C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests -v

C:\Users\admin\11\envs\game\python.exe -m src.csdemo.train_xgb `
  --task pre_round `
  --data data\processed\esta_full\pre_round.parquet `
  --model-dir models\esta_full_m6 `
  --report-dir reports\esta_full_m6

C:\Users\admin\11\envs\game\python.exe -m src.csdemo.m6_analysis `
  --data data\processed\esta_full\pre_round.parquet `
  --report-dir reports\esta_full_m6
```

## 15. 总结

最初版本已经证明购买结束时的地图、经济、防具、武器和比分信息能够预测回合胜负，
测试 AUC 达到 0.7209。经过 M2、M4 和 M6 后，当前版本测试 AUC 为 0.7220，
分数变化不大，但数据已经从“能够训练”提升为“身份唯一、质量受控、特征可解释、
有测试保护并可以重复实验”。

因此当前结论是：开局前 XGBoost 基线已经成立，M6 已完成；下一步应先完成 M7
简单基线对照，再进入 M8 early stopping 和过拟合控制，而不是立刻加入战队或选手身份。

## 16. M8 控制变量调参更新

2026-08-17 已完成 M8。使用相同数据、切分和特征，只通过验证集逐项调整参数：

```text
max_depth = 2
min_child_weight = 3
learning_rate = 0.03
subsample = 0.85
colsample_bytree = 0.85
early_stopping_rounds = 100
最佳树数 = 213
```

当前正式测试结果更新为 Accuracy 0.6474、Log Loss 0.5917、AUC 0.7271；
Train-Val AUC 差从 M6 的 0.0580 降到 0.0111。M8 完整控制变量过程见
`reports/esta_full_m8_tuned/m8_controlled_tuning_report.md`。

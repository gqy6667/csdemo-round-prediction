# M13 独立预测接口与输入校验

## 1. 这一阶段做什么

M13 不重新训练模型，而是把 M8 保存的 XGBoost 和 M10 保存的概率校准选择封装成
一个可以直接使用的命令。输入一条“购买结束、冻结时间结束、第一次交火之前”的
回合快照，程序输出 CT 和 T 的回合获胜概率。

数据流如下：

```text
一条 JSON/CSV
  -> 校验 27 个基础字段
  -> 自动计算 9 个 CT-T 差值
  -> 复用训练时的 prepare_features()
  -> 按模型保存的 43 列对齐
  -> XGBoost predict_proba()
  -> M10 校准器（当前选择 identity，不改变概率）
  -> CT/T 胜率和校验信息
```

## 2. 在 VSCode 中开始

在 VSCode 选择“文件 -> 打开文件夹”，打开：

```text
C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction
```

打开 VSCode 终端后，可以继续激活 `game` 环境，也可以像本项目文档一样直接调用
该环境里的 Python。直接调用更不容易误用系统 Python：

```powershell
C:\Users\admin\11\envs\game\python.exe --version
```

Conda 环境的作用是固定 Python、XGBoost、pandas、joblib 等依赖。模型文件本身不能
独立执行，必须由装有相应依赖的 Python 环境读取。

## 3. 最简单的运行命令

项目已提供一条 JSON 示例：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round `
  --input examples\pre_round_snapshot.json `
  --model models\esta_full_m8_tuned\pre_round_xgb.joblib `
  --calibrator models\esta_full_m10\pre_round_calibrator.joblib
```

CSV 示例也可以直接替换输入路径：

```powershell
C:\Users\admin\11\envs\game\python.exe -m src.csdemo.predict_pre_round `
  --input examples\pre_round_snapshot.csv `
  --model models\esta_full_m8_tuned\pre_round_xgb.joblib `
  --calibrator models\esta_full_m10\pre_round_calibrator.joblib
```

需要把输出保存为文件时，加上：

```text
--output examples\my_prediction.json
```

## 4. 需要填写的 27 个字段

用户只填写基础字段，不需要填写 `_diff_ct` 差值字段。

| 字段 | 含义 | 接口范围 |
|---|---|---:|
| `map_name` | 地图名 | 训练中出现的 8 张地图之一 |
| `round_num` | 当前回合号，从 1 开始 | 1 到 100 |
| `ct_score`, `t_score` | 回合开始前双方比分 | 各 0 到 99 |
| `ct_eq_value`, `t_eq_value` | 双方总装备价值 | 各 0 到 50,000 |
| `ct_cash`, `t_cash` | 双方五名存活选手现金总和 | 各 0 到 80,000 |
| `ct_armor`, `t_armor` | 有护甲的选手人数 | 各 0 到 5 |
| `ct_helmets`, `t_helmets` | 有头盔的选手人数 | 各 0 到 5 |
| `ct_defuse_kits` | CT 拆弹器数量 | 0 到 5 |
| `ct_grenades`, `t_grenades` | 双方手雷总数，每人最多计 4 枚 | 各 0 到 20 |
| `ct_ak47`, `t_ak47` | 双方持有 AK-47 的人数 | 各 0 到 5 |
| `ct_m4a4`, `t_m4a4` | 双方持有 M4A4 的人数 | 各 0 到 5 |
| `ct_m4a1_s`, `t_m4a1_s` | 双方持有 M4A1-S 的人数 | 各 0 到 5 |
| `ct_awp`, `t_awp` | 双方持有 AWP 的人数 | 各 0 到 5 |
| `ct_rifles`, `t_rifles` | 双方持有步枪的总人数 | 各 0 到 5 |
| `ct_smgs`, `t_smgs` | 双方持有冲锋枪的总人数 | 各 0 到 5 |

允许的地图为：`de_ancient`、`de_dust2`、`de_inferno`、`de_mirage`、
`de_nuke`、`de_overpass`、`de_train`、`de_vertigo`。未见过的地图没有可靠的
类别编码，接口会拒绝，而不是偷偷当成某张已知地图。

## 5. 自动生成的 9 个字段

程序统一按“CT 数值减 T 数值”计算：

```text
score_diff_ct
eq_value_diff_ct
cash_diff_ct
armor_diff_ct
helmet_diff_ct
grenade_diff_ct
awp_diff_ct
rifle_diff_ct
smg_diff_ct
```

例如 CT 装备价值 22,000、T 装备价值 18,500，则
`eq_value_diff_ct = 3500`。用户也可以提供差值，但只用于一致性检查；填错会被拒绝。

## 6. 输入校验规则

接口会一次收集尽可能多的错误并返回，主要规则是：

1. 27 个基础字段必须全部存在，不能加入未知字段。
2. 数量和金额必须是有限整数；字符串 `"3500"` 不等于整数 `3500`。
3. 每个字段必须位于表中的合理范围。
4. `round_num = ct_score + t_score + 1`。
5. 头盔人数不能超过护甲人数。
6. AK、M4A4、M4A1-S 的人数总和不能超过 `rifles`。
7. `rifles + awp + smgs` 不能超过每方 5 名选手。
8. 用户提供的差值必须等于程序计算值。

校验失败时命令退出码为 `2`，错误 JSON 写到终端错误输出。模型不会对不合格输入
勉强给出一个看似正常的概率。

## 7. 如何看输出

当前示例的关键输出为：

```json
{
  "calibration_method": "uncalibrated",
  "prediction": {
    "ct_win_probability": 0.5676999688148499,
    "t_win_probability": 0.43230003118515015,
    "predicted_side": "CT",
    "decision_threshold": 0.5,
    "probability_sum": 1.0
  }
}
```

`uncalibrated` 在这里不是漏做步骤，而是 M10 使用分组验证后主动选择的 identity
校准器：sigmoid 和 isotonic 都使验证集 Log Loss/Brier 变差，因此保留原概率。

`predicted_side` 只是按 0.5 阈值给出的分类。胜率 56.77% 表示在相似训练数据和当前
特征下，模型更偏向 CT；它不是“CT 一定会赢”，也不是投注建议。

## 8. 验收结果

| 检查 | M13 结果 |
|---|---|
| JSON/CSV 输入 | 均通过 |
| 两种格式概率 | 完全一致 |
| 概率范围与总和 | `[0, 1]` 且总和为 1 |
| 错误案例 | 5/5 被拒绝 |
| 用户输入/派生/编码字段 | 27 / 9 / 43 |
| 自动化测试 | 全项目 59/59 通过 |

M13 没有训练或调参，固定测试结果仍为 Accuracy `0.647411`、AUC `0.727122`、
Log Loss `0.591733`、Brier `0.205294`、ECE10 `0.023198`。

与预测时点最接近的公开 DNN 报告相比，本模型 Accuracy 低 `3.18` 个百分点，
Log Loss 高 `0.023873`。两者使用的数据和划分不同，这只能作为参考差距，不能证明
差异完全来自 XGBoost 与 DNN。完整表见
`reports/esta_full_m13/external_benchmark_comparison.md`。

## 9. 代码和产物

```text
src/csdemo/predict_pre_round.py       单条输入校验与预测命令
src/csdemo/m13_interface.py           M13 可重复验收入口
examples/pre_round_snapshot.json      JSON 输入示例
examples/pre_round_snapshot.csv       CSV 输入示例
examples/pre_round_prediction_output.json  示例输出
tests/test_m13_prediction.py          M13 自动化测试
reports/esta_full_m13/                M13 报告目录
```

下一阶段是 M14：做开局前 XGBoost 的最终可复现验收，并明确未达目标指标、剩余风险和
进入首杀后模型前的决定。

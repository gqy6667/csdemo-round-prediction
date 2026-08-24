# M24 开局前 LightGBM 固定模型评估规格

## 1. 目标

M24 对 M23 已冻结的开局前 LightGBM 做独立、可复现的正式评估。本阶段不训练
LightGBM、不调参数、不改特征，也不得根据 test 结果返回 M23 扩大参数网格。需要回答：

1. 五项整体指标的系列赛级 95% 置信区间是多少；
2. LightGBM 相对同测试行 XGBoost 的小幅优势是否具有稳定的配对证据；
3. 不同地图、LAN/online、回合阶段和装备差下是否稳定；
4. validation-only 校准是否值得使用；
5. 高置信错误集中在哪些购买状态及首杀后果中。

## 2. 冻结输入

- 数据：`data/processed/esta_full/pre_round.parquet`；
- 冻结模型：`models/esta_full_m23/pre_round_lightgbm_tuned.joblib`；
- M23 摘要：`reports/esta_full_m23/m23_summary.json`；
- M23 测试概率：`reports/esta_full_m23/test_predictions.csv`；
- 事后首杀诊断：`data/interim/esta_full/kills.parquet`；
- 外部指标：`benchmarks/external_round_model_metrics.csv`；
- 主键：`series_id + game_id + round_id`。

必须验证 M23 已通过、数据和模型 SHA-256、36 个原始/43 个编码特征、
28,522/8,380/4,172 行 split、547/156/79 个系列赛及完整测试主键。重新回放的
LightGBM 测试概率与 M23 保存概率最大绝对误差必须不超过 `1e-12`。

## 3. 整体与配对 Bootstrap

按 `series_id` 有放回抽样 2,000 次，计算 Accuracy、AUC、Log Loss、Brier 和 ECE10
的 95% percentile 区间。预先固定最低线和阶段目标：

| 项目 | 最低验收 | 阶段目标 |
|---|---:|---:|
| AUC 95% CI 下界 | >= 0.700 | >= 0.710 |
| Log Loss 95% CI 上界 | <= 0.610 | <= 0.605 |
| 每项成功次数 | 2,000 | 2,000 |

同一批抽样同时计算 LightGBM 和 XGBoost，保存五项指标的 `LightGBM-XGBoost`
原始差值、统一为越大越好的性能优势及其 95% CI。配对比较完整是阻断项，但不要求
置信区间必须排除 0；是否显著只能由冻结区间决定。

## 4. 固定分组与错误分析

地图、来源、回合阶段（1-10、11-20、21+）和装备差五档均报告回合数、系列赛数、
CT 胜率、五项指标和系列赛级 95% CI。

- LAN 与 online AUC 绝对差必须 <= `0.040`；
- 至少 300 回合的主要地图点估计 AUC 必须 >= `0.670`，阶段目标 >= `0.690`；
- 主要地图最低 AUC CI 下界 >= `0.670` 是研究目标，不作为最低阻断线。

高置信错误定义为预测错误且预测方概率 >= `0.80`。保存全部错误并复核置信度最高的
30 个。首杀只允许作为预测完成后的事后诊断，绝不能进入开局前模型特征。

## 5. 校准协议

固定比较不校准、Sigmoid、Isotonic：

1. validation 按 `series_id` 做 5 折 GroupKFold；
2. 生成三种方法的完整 OOF 概率；
3. 只按 OOF Log Loss、Brier、方法名依次排序选择；
4. 用完整 validation 拟合所选校准器；
5. 选择冻结后才在 test 上比较。

所选方法相对原始概率的 test Log Loss 变差不得超过 `0.002`，Brier 变差不得超过
`0.001`；ECE10 <= `0.030` 是阶段目标。不校准胜出时保存 Identity 校准器，这也是
有效结论。

## 6. 代码、命令与结构

```text
src/csdemo/m24_pre_round_lightgbm_evaluation.py
tests/test_m24_pre_round_lightgbm_evaluation.py
scripts/run_pre_round_lightgbm_evaluation.ps1
models/esta_full_m24/pre_round_lightgbm_calibrator.joblib
reports/esta_full_m24/
```

正式命令：

```powershell
.\scripts\run_pre_round_lightgbm_evaluation.ps1
```

测试与编译：

```powershell
C:\Users\admin\11\envs\game\python.exe -m unittest tests.test_m24_pre_round_lightgbm_evaluation -v
C:\Users\admin\11\envs\game\python.exe -m unittest discover -s tests -v
C:\Users\admin\11\envs\game\python.exe -m compileall src tests
```

代码沿用现有函数式模块、完整主键校验、CSV/JSON/Markdown 证据和 PowerShell 一键入口；
不增加依赖。

## 7. 阻断验收

1. M23 数据、模型、任务和特征契约通过；
2. 70/20/10 split 与完整主键一致；
3. 冻结概率和五项指标精确回放；
4. 五项整体 bootstrap 均完成 2,000 次并通过最低区间目标；
5. 五项 LightGBM-XGBoost 配对 bootstrap 完整；
6. 四类固定分组均生成；
7. LAN/online AUC 差 <= 0.040；
8. 主要地图最低 AUC >= 0.670；
9. 校准严格由 validation OOF 选择且 test 概率指标无明显伤害；
10. 全部高置信错误和前 30 个案例已保存；
11. 外部比较、中文报告、实验清单、一键入口、自动化测试和源码编译通过。

高阶段目标未全部达到时记录“接受现阶段、保留改进项”，不篡改最低验收线。

## 8. 边界

- 始终：完整主键连接、系列赛级抽样、test 不参与选择、保留所有诊断结果；
- 需要另立阶段：改特征、重新调参、改 split、加入战队或选手身份；
- 禁止：逐行随机切分、只按 `round_id` 连接、用 test 选择校准或模型、把首杀作为
  开局前输入。

## 9. 下一阶段

M24 通过后进入 M25：对冻结 LightGBM 做 gain、Permutation Importance、SHAP、
泄漏审计和与 XGBoost 的解释差异分析。

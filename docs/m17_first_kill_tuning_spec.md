# M17 首杀后 XGBoost 控制变量调参规格

## 1. 目标

M17 在 M16 验收通过的首杀后 XGBoost 上进行逐阶段控制变量调参。目标不是保证测试集
一定提高，而是在不泄漏测试集的前提下，寻找 validation Log Loss 更低、泛化差距更小
且对随机种子稳定的配置。

本阶段不改变数据、预测时点、特征、标签或 70/20/10 系列赛切分，不加入身份特征，
不进行校准、SHAP、LightGBM 或实时胜率建模。

## 2. 已冻结假设

- 数据：`data/processed/esta_full/first_kill.parquet`，SHA-256 必须与 M15/M16 一致。
- 样本：41,027 行，train/validation/test 为 28,489/8,368/4,170。
- 特征：M16 `canonical_event` 的 40 个原始特征、82 个训练集编码列。
- 标签：`ct_win=1` 表示 CT 最终赢得回合。
- 随机种子：正式选择使用 42；稳定性使用 42、43、44、45、46。
- 环境：Python 3.10、XGBoost 3.2.0、scikit-learn 1.7.2、CPU。
- 调参期间函数只接收 train/validation；test 只传给冻结模型的最终评估函数。

M16 的未经调参 XGBoost 是搜索起点：500 棵树、depth 4、learning rate 0.03、
subsample/colsample 0.85、seed 42、无 early stopping。

## 3. 选择规则

所有候选使用 validation Log Loss 作为唯一选择指标，AUC、Brier、Accuracy 和 ECE10
只作诊断。每个阶段按以下规则选择：

1. 当前参数必须作为该阶段候选之一。
2. 找出 validation Log Loss 最低的候选。
3. 若它相对当前参数的改善小于 `0.0001`，保留当前参数。
4. 否则选择最低 Log Loss 候选，并作为下一阶段的固定起点。
5. 相同 Log Loss 时按规格中的候选顺序选择，确保结果可复现。

该规则防止为了万分位以内的波动不断增加模型复杂度。测试 AUC、测试 Log Loss 或外部
模型数值不得参与候选选择。

## 4. 候选阶段

| 顺序 | 阶段 | 候选 |
|---:|---|---|
| 1 | tree policy | fixed 500；1500 上限/50 early stop；3000 上限/100 early stop |
| 2 | `max_depth` | 2、3、4、5、6 |
| 3 | `min_child_weight` | 1、3、5、10、20 |
| 4 | `reg_lambda` | 0.5、1、3、5、10 |
| 5 | `reg_alpha` | 0、0.05、0.1、0.5、1 |
| 6 | `subsample` | 0.7、0.8、0.85、0.9、1.0 |
| 7 | `colsample_bytree` | 0.6、0.7、0.8、0.85、0.9、1.0 |
| 8 | `learning_rate` | 0.01、0.02、0.03、0.05、0.1 |

tree policy 是唯一允许同时改变 `n_estimators` 和 `early_stopping_rounds` 的阶段；其他
阶段只允许改变表中一个参数。候选总数为 39。使用 greedy sequential search，因此结果
是受控局部选择，不宣称为全局最优。

## 5. 预先目标

M16 validation 基准：Log Loss `0.529835`、AUC `0.802069`、train-validation AUC 差
`0.039154`。

| 项目 | 阶段目标 | 是否阻塞 |
|---|---:|---|
| validation Log Loss | 至少改善 0.001，即不高于 0.528835 | 否 |
| validation AUC | 不低于 0.800069 | 否 |
| train-validation AUC 差 | 不高于 0.030 | 否 |
| 五种子 validation Log Loss 最大差 | 不高于 0.002 | 是 |
| 五种子 validation AUC 最大差 | 不高于 0.003 | 是 |

最终 test 继续报告 Accuracy、AUC、Log Loss、Brier 和 ECE10。测试集是否优于 M16 不设
为阻塞项，因为用测试结果决定是否接受参数会造成测试泄漏。若测试结果退化，报告必须
保留退化结论，并在部署建议中保留 M16 模型。

## 6. 内部与外部比较

内部同样本比较：

- M17 tuned XGBoost 与 M16 untuned XGBoost。
- M17 tuned XGBoost 与 M16 logistic regression。
- 报告五项测试指标原始差值和性能方向差值。

外部比较继续使用 M16 已核验的三个来源，并把 XGBoost 行映射到 M17 tuned 模型：

- 最接近首杀后任务的 CS156 逻辑回归。
- 使用更丰富整回合状态的 Xenopoulos 等人 WPA XGBoost。
- 预测时点更早的 Aakerholt freeze-time DNN。

所有差值均为“我们的指标减外部指标”。数据、年代、切分和时点不同，不能据此宣称
模型本身优于外部工作。

## 7. 代码与产物

```text
src/csdemo/m17_first_kill_tuning.py    调参、稳定性、验收和报告
tests/test_m17_first_kill_tuning.py    协议与选择逻辑测试
scripts/run_first_kill_tuning.ps1      可重复阶段入口
benchmarks/external_first_kill_tuned_metrics.csv
models/esta_full_m17/                  本地模型，不提交
reports/esta_full_m17/                 调参表、预测和报告
```

代码使用小函数和结构化 DataFrame。例如：

```python
def select_phase_winner(results: pd.DataFrame, minimum_improvement: float) -> str:
    incumbent = results.loc[results["is_incumbent"]].iloc[0]
    best = results.sort_values(["val_log_loss", "candidate_order"]).iloc[0]
    if incumbent["val_log_loss"] - best["val_log_loss"] < minimum_improvement:
        return str(incumbent["candidate_id"])
    return str(best["candidate_id"])
```

## 8. 测试策略

- 单元测试：候选网格、单变量审计、最小改善规则、确定性平局、冻结参数审计。
- 泄漏测试：调参函数签名不接收 test；候选表禁止出现 `test_*` 列。
- 合同测试：M16/M15 哈希、行数、split、特征列和测试主键完全一致。
- 集成验收：生成 39 行候选结果、8 行阶段选择、5 行种子稳定性和一组最终 test 概率。
- 最终验证：`python -m unittest discover -s tests -v` 与 `compileall src tests`。

## 9. 阻塞验收条件

1. M16 已通过，M17 数据哈希、行数、split 与特征合同完全一致。
2. 候选定义通过单变量审计，正式候选数恰为 39。
3. 候选结果和阶段选择不包含任何 test 指标。
4. 八个阶段均按 validation Log Loss 和 `0.0001` 改善规则选择。
5. 冻结模型参数与最后阶段清单一致，正式种子为 42。
6. 五种子 validation Log Loss/AUC 范围通过稳定性门槛。
7. 最终 test 主键与 M16 完全一致，概率有限且位于 `[0, 1]`。
8. 最终模型继续通过 M16 四项最低门槛。
9. 内部差值、外部差值和所有未达阶段目标均写入报告。
10. 自动化测试和编译检查通过。

## 10. 命令

```powershell
.\scripts\run_first_kill_tuning.ps1
```

主要产物：

- `reports/esta_full_m17/controlled_tuning_results.csv`
- `reports/esta_full_m17/phase_selections.csv`
- `reports/esta_full_m17/seed_stability.csv`
- `reports/esta_full_m17/model_comparison.csv`
- `reports/esta_full_m17/test_predictions.csv`
- `reports/esta_full_m17/m17_summary.json`
- `reports/esta_full_m17/m17_first_kill_tuning_report.md`
- `reports/esta_full_m17/external_benchmark_comparison.csv/.md`

## 11. 边界与下一阶段

始终执行：validation-only 选择、完整实验记录、测试集最后评价、保存概率和模型哈希。

需要另立阶段：模型校准、特征消融、身份/位置/血量、LightGBM、实时胜率。

禁止执行：根据 test 改网格或参数、删除失败候选、改变 M16 特征、覆盖 M16 模型、把
本地 ESTA 数据或大模型提交 Git。

M17 完成后，M18 对冻结的首杀后模型做固定测试集 bootstrap、分地图/LAN-online
稳健性和概率校准诊断。

## 12. 实际验收结果

M17 于 2026-08-19 完整运行并通过：

- 8 个阶段共运行 39 个候选；候选表没有任何 `test_*` 列。
- 接受 `tree_cap_1500_es50`、`max_depth=2` 和 `subsample=0.9`，其余阶段因改善不足
  `0.0001` 或没有改善而保留当前值。
- seed 42 最佳迭代为 408，即使用 409 棵树；validation Log Loss `0.527796`、AUC
  `0.803324`，相对 M16 validation Log Loss 改善 `0.002038`。
- 五种子 validation Log Loss 最大差 `0.000130`，AUC 最大差 `0.000318`。
- 最终 test Accuracy `0.744125`、AUC `0.809837`、Log Loss `0.523146`、Brier
  `0.175656`、ECE10 `0.015450`。
- 相对 M16 XGBoost，AUC、Log Loss、Brier 分别改善 `0.000941`、`0.001607`、
  `0.000609`；Accuracy 下降 `0.001199`，ECE10 恶化 `0.004541`。
- 12 个阻塞检查、100 项自动化测试及编译检查全部通过，可以进入 M18。

实际结果详见 `reports/esta_full_m17/m17_first_kill_tuning_report.md`。

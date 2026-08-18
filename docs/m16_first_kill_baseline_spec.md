# M16 首杀后固定切分基线规格

## 1. 目标

M16 在 M15 验收通过的 41,027 条首杀后样本上建立第一组正式模型结果。三个正式模型
必须使用相同的 train/validation/test 行、相同的特征列和相同的指标代码：

1. 训练集 CT 胜率常数模型（Dummy）。
2. 标准化后的逻辑回归。
3. 未经调参的 XGBoost。

本阶段回答“首杀后数据是否有稳定预测信号”和“树模型是否已经优于简单线性模型”，
不进行参数搜索、校准、SHAP、战队/选手身份实验或 LightGBM 对比。

## 2. 已冻结假设

- 预测时点：购买完成后，同一完整回合主键内最小 tick 的有效敌方击杀刚发生。
- 标签：`ct_win=1` 表示 CT 最终赢得该回合。
- 数据：`data/processed/esta_full/first_kill.parquet`，SHA-256 必须与 M15 清单一致。
- 切分：直接使用 M15/M14 的系列赛级 70/20/10，不能重新随机划分。
- 测试集：特征、模型和目标在本规格中冻结后才评价一次。
- 本阶段不加入战队名、选手名、Steam ID 或其他身份特征。

## 3. 特征合同

正式 `canonical_event` 特征组为全部 M14 购买结束特征，加四个首杀信息：

```python
CANONICAL_EVENT_FEATURES = PRE_ROUND_FEATURES + [
    "first_kill_advantage_ct",
    "first_kill_time",
    "first_kill_headshot",
    "first_kill_weapon",
]
```

以下字段用于 M15 审计和解释，但不进入 M16 模型：

- `first_kill_is_ct`
- `first_death_is_ct`
- `ct_alive_after_fk`
- `t_alive_after_fk`
- `alive_diff_ct_after_fk`

原因是它们都能由 `first_kill_advantage_ct` 和固定的 5v5 初始状态确定。重复输入不会增加
新信息，还会让系数、重要性和后续消融难以解释。

额外训练一个 `pre_round_control` XGBoost，只使用 `PRE_ROUND_FEATURES`，但保持相同
41,027 条样本、相同 split 和相同 XGBoost 参数。它只用于量化首杀事件信息的增益，
不替代三个正式基线。

## 4. 模型配置

| 模型 | 固定配置 |
|---|---|
| Dummy | `DummyClassifier(strategy="prior")` |
| Logistic | `StandardScaler` + `LogisticRegression(max_iter=2000, random_state=42)` |
| XGBoost | 500 棵树、depth 4、learning rate 0.03、subsample/colsample 0.85、seed 42 |

XGBoost 使用项目现有 `make_model(task="first_kill")`。M16 不启用 early stopping，
不根据 validation 或 test 改动参数。validation 仅用于报告和与购买结束控制组比较。

## 5. 指标与预先目标

统一使用阈值 0.5，并报告 Accuracy、AUC、Log Loss、Brier Score 和 ECE10。

| 指标 | 最低门槛 | 阶段目标 | 方向 |
|---|---:|---:|---|
| Accuracy | 0.680 | 0.700 | 越高越好 |
| AUC | 0.750 | 0.780 | 越高越好 |
| Log Loss | 0.580 | 0.550 | 越低越好 |
| Brier | 0.200 | 0.185 | 越低越好 |

最低门槛用于判断当前 XGBoost 是否具备进入后续调参阶段的基础；阶段目标记录仍需提高
多少。XGBoost 是否领先逻辑回归不设为阻塞条件，必须如实报告 AUC、Log Loss 和 Brier
差值。`canonical_event` 相对 `pre_round_control` 的 validation AUC 增益目标为至少 0.03。

## 6. 外部比较

M16 使用单独的 `benchmarks/external_first_kill_metrics.csv`：

- 时点最接近：CS156 项目使用经济差和首杀方，在 424 个个人 CS2 回合上报告逻辑回归
  Accuracy 0.6824、AUC 0.76。它按回合行随机 80/20 切分，没有系列赛隔离。
- 更丰富的实时任务：Xenopoulos 等人的职业 CSGO WPA 模型在整回合多个状态上报告
  XGBoost AUC 0.7913、Log Loss 0.5353、Brier 0.1842。
- 更早的购买结束任务：M14 已使用的 freeze-time DNN Accuracy 0.6792、Log Loss
  0.5679，作为预测时点更早的参考。

每条外部记录指定应与本阶段的逻辑回归或 XGBoost 比较。所有差值均为“我们的指标减
外部指标”；数据集、年代、特征和切分不同，不能据此宣称模型本身更优。

## 7. 技术栈与代码风格

- Python 3.10、pandas、scikit-learn、XGBoost 3.2、joblib。
- 使用现有 `probability_metrics`、`prepare_features`、`align_columns` 和模型构造函数。
- 新代码放在 `src/csdemo/m16_first_kill_baselines.py`，测试放在
  `tests/test_m16_first_kill_baselines.py`。
- 函数接收 DataFrame/Path 并返回结构化结果；CLI 只负责参数和文件输出。

示例风格：

```python
def canonical_feature_names() -> list[str]:
    return [*PRE_ROUND_FEATURES, *FIRST_KILL_MODEL_FEATURES]
```

## 8. 测试策略

- 单元测试：特征白名单、冗余字段排除、类别列对齐、模型固定参数、指标方向。
- 合同测试：完整主键唯一、三个 split 都存在、系列赛不跨 split、M15 SHA 匹配。
- 集成验收：完整数据训练三个基线和一个控制组，保存模型、预测、指标和阶段报告。
- 最终运行：`python -m unittest discover -s tests -v` 与 `compileall src tests`。

## 9. 边界

始终执行：训练集拟合、固定 split、保存概率而不只保存类别、报告失败目标。

需要另立阶段：修改预测时点、加入身份/位置/血量、参数搜索、校准、LightGBM。

禁止执行：把 ID 或 label 当特征、把确定性冗余列全塞入模型、根据 test 调参、覆盖
M15 原始审计报告、把本地 ESTA 或大模型文件提交 Git。

## 10. 阻塞验收条件

1. M15 数据哈希、行数和 split 行数完全一致。
2. 完整主键重复、跨 split 系列赛和模型间测试行差异均为 0。
3. 正式特征组只含白名单，身份、标签和五个冗余字段均未进入编码列。
4. 三个正式模型都输出 41,027 个有限且位于 `[0, 1]` 的概率。
5. XGBoost 参数与本规格一致，没有调参或 early stopping。
6. XGBoost 测试集四项最低门槛全部通过。
7. 自动化测试全部通过，报告包含内部控制组和外部数值差。

XGBoost 不领先逻辑回归、或未达到更高阶段目标，必须保留为研究结论，但不通过删改
结果来“修好”。

## 11. 命令与产物

```powershell
.\scripts\run_first_kill_baselines.ps1
```

主要产物：

- `models/esta_full_m16/first_kill_*.joblib`（本地，不提交）
- `reports/esta_full_m16/m16_model_comparison.csv`
- `reports/esta_full_m16/m16_feature_control.csv`
- `reports/esta_full_m16/test_predictions.csv`
- `reports/esta_full_m16/feature_contract.csv`
- `reports/esta_full_m16/m16_summary.json`
- `reports/esta_full_m16/m16_first_kill_baseline_report.md`
- `reports/esta_full_m16/external_benchmark_comparison.csv/.md`

## 12. 下一阶段

M16 通过后，M17 只使用 train/validation 做控制变量调参。先调整树数量/early stopping，
再依次评估 depth、min child weight、正则和采样比例；方案冻结后才重新评价一次 test。

## 13. 实际验收结果

M16 于 2026-08-19 完整运行并通过：

- 数据哈希与 M15 一致，41,027 行，train/validation/test 为 28,489/8,368/4,170。
- XGBoost 测试 Accuracy `0.745324`、AUC `0.808896`、Log Loss `0.524753`、
  Brier `0.176265`，四项阶段目标全部达到。
- 逻辑回归测试 AUC `0.809059`，比 XGBoost 高 `0.000163`；当前没有证据说明树模型
  在排序能力上优于线性基线。
- 首杀事件特征相对同样本开局控制组增加 `0.088015` validation AUC 和 `0.086007`
  test AUC，超过预先设定的 `0.03` validation 目标。
- 8 个阻塞检查、90 项自动化测试及编译检查全部通过，可以进入 M17。

实际结果详见 `reports/esta_full_m16/m16_first_kill_baseline_report.md`。

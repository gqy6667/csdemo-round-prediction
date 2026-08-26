# 老师查收报告规格

## 1. 目的

在进入实时胜率课题前，为已经完成或即将完成的两个预测时点和两个树模型形成四份可独立阅读、可追溯到正式产物的报告：

1. 购买结束、交火前 XGBoost；
2. 购买结束、交火前 LightGBM；
3. 首杀后 XGBoost；
4. 首杀后 LightGBM。

每份报告只陈述已经冻结并验收的结果。首杀后 LightGBM 在控制变量实验及最终验收完成前不得预填指标或宣称完成。

## 2. 共同结构

每份独立报告至少包含以下内容：

1. 研究问题与严格预测时点；
2. 数据来源、清洗结果和样本量；
3. 系列赛级 70%/20%/10% 切分及泄漏检查；
4. 标签、特征合同和禁用字段；
5. 模型、训练规则、调参数据边界和校准规则；
6. 测试集 Accuracy、AUC、Log Loss、Brier、ECE10；
7. 系列赛级统计不确定性或其他已验收稳健性证据；
8. 模型解释及其非因果边界；
9. 局限性、未达目标和适用范围；
10. 默认复现命令、完整重建命令和关键产物链接；
11. 数据、模型、校准器等关键产物的 SHA-256。

报告中的精确数字必须来自对应阶段的 JSON、CSV 或实验清单，不从旧版叙述性文档手工推断。

## 3. 公平比较边界

- 购买结束、交火前 XGBoost 与 LightGBM 可以进行算法控制变量比较，因为二者共享样本、标签、系列赛级切分、特征合同和测试口径。
- 首杀后 XGBoost 与 LightGBM 只有在 M28 及后续阶段确认共享相同合同后，才可以进行算法控制变量比较。
- 购买结束、交火前与首杀后属于不同信息时点。两类指标可以分别报告，但不能把差值解释成模型算法本身的优劣。
- 测试集只用于一次性最终评估和配对比较，不用于选模型、选校准器或继续调参。
- 点指标差异必须和系列赛级配对不确定性一起解释；置信区间包含 0 时不得宣称一方显著更优。

## 4. 冻结来源

| 报告 | 最终验收来源 | 指标与不确定性来源 | 训练、解释、接口来源 |
|---|---|---|---|
| 购买结束、交火前 XGBoost | `reports/esta_full_m14/` | `reports/esta_full_m9/`、`reports/esta_full_m10/`、`reports/esta_full_m11/` | `reports/esta_full_m8_tuned/`、`reports/esta_full_m12/`、`reports/esta_full_m13/` |
| 购买结束、交火前 LightGBM | `reports/esta_full_m27/` | `reports/esta_full_m24/`、`reports/esta_full_m27/` | `reports/esta_full_m23/`、`reports/esta_full_m25/`、`reports/esta_full_m26/` |
| 首杀后 XGBoost | `reports/esta_full_m21/` | `reports/esta_full_m18/` | `reports/esta_full_m17/`、`reports/esta_full_m19/`、`reports/esta_full_m20/` |
| 首杀后 LightGBM | `reports/esta_full_m33/` | `reports/esta_full_m30/`、`reports/esta_full_m33/` | `reports/esta_full_m29/`、`reports/esta_full_m31/`、`reports/esta_full_m32/` |

若同一事实同时出现在多个文件中，以最终验收清单中的身份、哈希和合同检查为准，以专项评估 CSV/JSON 中的完整精度指标及置信区间为准。

## 5. 目标文件与完成门槛

目标文件：

```text
reports/teacher_review/01_pre_round_xgboost_report.md
reports/teacher_review/02_pre_round_lightgbm_report.md
reports/teacher_review/03_post_first_kill_xgboost_report.md
reports/teacher_review/04_post_first_kill_lightgbm_report.md
reports/teacher_review/README.md
```

单份报告只有在以下条件全部满足后才算完成：

- 五项测试指标与冻结源产物一致；
- 样本量、切分量、训练规则和哈希与冻结源产物一致；
- 本地 Markdown 链接全部存在；
- 明确预测时点、测试集隔离和比较边界；
- 专项测试和当时的全量自动化测试通过；
- 在任务清单中记录完成状态。

总索引不代替独立报告。最终索引只做链接、状态和两组公平比较关系的导航，不重复包装四份报告的结论。

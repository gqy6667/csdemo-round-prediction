# M21 首杀后 XGBoost 最终验收报告

## 最终结论

验收状态：**passed**；阻断项：**17/17**；首杀后 XGBoost 完成：**True**。
M21 没有训练或调参，只回放固定 M17 模型、M18 identity 校准器和 M20 接口。

## 数据与切分

- 首杀后样本：41,027；series：782；game：1,558；
- 数据 SHA-256：`06f7f5887388c433870d36a39ca1cd9337236254a08e55751250c87e9e8b7492`；
- 跨 split 的 series/game/round：0/0/0；

| split | series | 样本 | 占比 |
|---|---:|---:|---:|
| train | 547 | 28,489 | 69.44% |
| val | 156 | 8,368 | 20.40% |
| test | 79 | 4,170 | 10.16% |

## 固定测试指标

| 指标 | M21 回放值 |
|---|---:|
| Accuracy | 0.744125 |
| AUC | 0.809837 |
| Log Loss | 0.523146 |
| Brier | 0.175656 |
| ECE10 | 0.015450 |

测试概率最大回放误差：`1.110e-16`；指标最大回放误差：`0.000e+00`。
十项正式目标通过：`10/10`；remaining：`0`。

## 模型与接口合同

- 原始/编码特征：40/82；
- 部署树数：409；
- 模型 SHA-256：`ecfaaf93031e78207f81ab5ad9674020657018c0601953238ee6b68e367e8279`；
- 校准器 SHA-256：`661db6964786dde1276dbeb6c0cf3f175858ad0bf02ecb109bb2f72c45074157`；
- JSON/CSV 示例一致：True；
- XGBoost fit 调用：0。

## 外部指标差距

以下仅列最接近的公开首杀后任务，仍因数据和切分不同而不能直接排名。

| 外部来源 | 指标 | 本项目逻辑回归 | 外部 | 差值 |
|---|---|---:|---:|---:|
| CS156 - Round-Win Probability in CS2 via Economic Asymmetry | accuracy | 0.743405 | 0.682400 | +6.10 个百分点 |
| CS156 - Round-Win Probability in CS2 via Economic Asymmetry | auc | 0.809059 | 0.760000 | +4.91 个百分点 |

## 可复现性

- Git commit：`5f608a20f0cb9b39e85af87c825b990535646f91`；
- Python：`3.10.20`；
- 自动化测试：145；
- 环境锁通过：True；

```powershell
.\scripts\run_first_kill_pipeline.ps1
```

完整重建使用 `-FullRebuild`；只从 M14 产物重建首杀后任务使用 `-RebuildFirstKill`。

## 下一阶段

首杀后 XGBoost 已关闭。下一阶段是 LightGBM 在完全相同数据、切分、特征和指标合同上的控制变量对照；实时胜率仍是之后的独立任务。

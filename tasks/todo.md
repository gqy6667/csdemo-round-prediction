# ROUNDCAST v1 — 审阅与进度

日期：2026-09-05。对应 [实施计划](plan.md) 与 [已批准规格](../docs/roundcast_interactive_v1_spec.md)。

- [x] 用户审阅批准产品范围和技术规格。
- [x] 只读核对三个候选案例的共同测试集身份、输入合法性和四条参考概率。
- [x] 形成 P0–P4 实施计划，保留旧设计与历史任务。
- [x] 用户确认实施顺序与计划（2026-09-05）。
- [x] 用户授权本轮将规格、实施计划、案例预检和进度文档同步至 GitHub。
- [ ] 细化逐项可执行任务与验收步骤，再交付审阅。
- [ ] 任务审阅通过后开始单案例真实端到端实现。

当前没有开始编码、启动网页或执行新的模型推理。上面的勾选仅代表文档/只读预检完成，不代表模型网页已完成。

---

以下为历史清单原文，原有勾选状态保持不变。

# M27-M33 Checklist

- [x] Audit M22-M26 and freeze the M27 final-acceptance specification.
- [x] Add failing M27 stage-chain, replay, metric, uncertainty, and runner tests.
- [x] Implement M27 frozen replay, contracts, report, manifest, and pipeline entrypoint.
- [x] Run focused and complete tests plus source compilation for M27.
- [x] Run formal M27 acceptance and verify every artifact hash.
- [x] Update documentation, commit, and push the completed M27 stage.
- [x] Write and verify the independent pre-round XGBoost teacher report.
- [x] Write and verify the independent pre-round LightGBM teacher report.
- [x] Write and verify the independent post-first-kill XGBoost teacher report.
- [x] Freeze the M28 first-kill LightGBM controlled-baseline specification.
- [x] Add failing M28 data, feature, training, prediction, and paired-comparison tests.
- [x] Implement and train M28 without using test for fitting or selection.
- [x] Report five metrics and paired series-level uncertainty against M21 XGBoost.
- [x] Run M28 formal acceptance, focused/full tests, compile, and hash verification.
- [x] Complete M29 validation-only controlled tuning and formal acceptance.
- [x] Complete M30 paired uncertainty, robustness, calibration, and formal evaluation.
- [x] Complete M31 frozen-model explanation and leakage audit.
- [x] Complete M32 frozen JSON/CSV prediction interface.
- [x] Complete M33 final stage-chain acceptance and one-command reproduction.
- [x] Write the post-first-kill LightGBM report only from completed accepted evidence.
- [x] Create the teacher review index linking all four reports.
- [x] Update documentation and locally commit the completed M31-M33 deliverables.
- [x] Locally commit the fourth report and teacher review index after final verification.

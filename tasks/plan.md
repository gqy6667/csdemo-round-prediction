# ROUNDCAST 本地交互演示 v1 — 实施计划

日期：2026-09-05。状态：规格及实施顺序已获用户批准，首轮规划文档已同步至 GitHub（eb0a0e3），T01–T05 与 Codex 对话已同步为 edfc541。[T01–T06](todo.md) 已完成，P2 的三例切换、四项对比及真实浏览器 12 项推理均通过；下一项为 T07。

规格：[roundcast_interactive_v1_spec.md](../docs/roundcast_interactive_v1_spec.md)。
案例预检：[roundcast_v1_case_readiness.md](../docs/roundcast_v1_case_readiness.md)。

附加请求（2026-09-05）：模型下方的 Codex 对话区已实现并完成真实连接验收。
独立规格：[roundcast_codex_chat_spec.md](../docs/roundcast_codex_chat_spec.md)。
实施按“固定消息/权限边界 → 后端可信上下文及临时对话 → 前端对话区 → 自动测试及真实网页验收”完成。
此附加项与现已完成的 T06 组合选择独立，不提前完成两个分析视图。

## 这次实现什么

一个本机运行的灰白网页，包含观众、比赛分析、技术分析三个联动视图。
三个固定真实案例可以在购买结束/首杀后各切换 XGBoost/LightGBM，实际运行模型。
不重新训练、不加预测时点、不开放自由输入、不部署公网、不覆盖旧静态设计。

## 实现顺序与阶段交付

| 阶段 | 要完成的事情 | 通过后能看到什么 | 必须通过的检查 |
| --- | --- | --- | --- |
| P0 可信输入准备 | 固定三例身份、来源和四模型映射，核验模型/校准器/数据的可信哈希 | 一份可追溯的案例与模型清单 | 两时点同一回合、共同 test、标签一致、输入合法；模型先验哈希后加载 |
| P1 单案例端到端 | 案例 A + 购买结束 XGBoost，接通本地服务和最小观众视图 | 点击运行，页面出现真实胜率和模型来源 | 真实 Predictor 调用，与正式参考概率误差不超过 1e-8；安全和错误状态同时到位 |
| P2 扩成 12 项 | 三个案例、两个时点、两算法的选择；两节点时间轴；运行全部对比 | 可切换全部组合，并生成四条真实对比结果 | 12 项逐一与参考结果比对；未运行项不填充；快速切换不串结果 |
| P3 三页联动 | 完成分析页与精简技术页，统一选择、结果及揭示状态 | 从观众页切到分析/技术页仍是同一回合 | 图表来自真实推理；技术指标来自正式完整测试集；没有剧透或占位数据 |
| P4 交付验收 | 故障、安全、全套回归、浏览器操作与三页截图；整理启动说明及三例演示步骤 | 本地可复现的 v1 演示包 | 所有规格完成条件通过；冻结模型/数据不变；失败不伪造概率 |

P0/P1 的 T01–T04 已验收：十工件哈希、三例准备、A 的真实单链推理、本地 HTTP 和观众页均通过。详见 [验收记录](../reports/roundcast_interactive_v1/t02_t04_verification.md)。
T05 后端 12 项与参考误差为 0；T06 页面选择和四项对比已通过真实浏览器验收。详见 [T06 验收](../reports/roundcast_interactive_v1/t06_verification.md)。
P1 只展示已接通的组合，未完成选项不可伪装为可用。P2 才开放完整选择矩阵。

## 组件与依赖

1. **可信来源层**：固定工件路径、SHA-256、案例唯一键、快照提取、标签/ID 与输入分离；复用现有验证器。
2. **模型服务层**：四个 Predictor 的轻量适配；输出统一概率、时点、版本、校准方式、请求标识及耗时。所有返回概率必须来自本次推理。
3. **本机 HTTP 层**：Python 标准库，同源提供 API 与白名单静态资源；只监听 127.0.0.1，无新增依赖。
4. **共享页面状态层**：案例、时点、算法、请求代次、结果组合键和主动揭示状态；三个视图共用，不各自保存另一套回合。
5. **展示与评估层**：观众胜率条/两节点时间轴；分析对比图/只读快照；技术版本和正式测试指标。

必要顺序：可信来源与接口合同 → 单条真实推理 → 本地网页 → 12 项组合 → 三视图完整展示 → 统一验收。

可并行的准备：确认合同后，后端推理适配与前端灰白布局可分别制作；前端暂时的开发 fixture 必须明确标记、不得进入正式演示。
案例来源审计与正式指标来源审计也可并行。共同状态协议在并行前定好。
不可并行跳过的关卡：P1 必须先证明真实调用，才把同一路径扩展到其他组合；最终视觉审查不能代替 API 与模型验证。

## 分析页如何获得四条真实结果

保留规格中的单组合 `POST /api/predict`，请求只含 `example_id`、`stage`、`algorithm`。
增加“运行全部对比”按钮，对当前案例逐项调用四次同一接口，不新增任意输入或批量文件上传。

- 每条结果独立显示待运行、运行中、成功、失败；部分失败不显示全体成功。
- 结果按案例 + 时点 + 算法 + 模型版本关联，图表只显示已成功的对应项。
- 参考概率表只供测试断言使用，不能补齐未运行的图表。
- 用户切换案例即使旧请求仍在返回，也不能覆盖新案例；清空旧案例的揭示状态。
- 一个案例四条结果形成后可供三个视图复用；再次点击运行仍执行真实 Predictor。
- 只有两节点对比，不插值或平滑成逐秒预测曲线。

## 首条端到端链必须具备的安全与状态

- 白名单模型、案例和静态资源；只允许 127.0.0.1，不能用任意路径读取仓库。
- 未知字段/ID/算法/时点、非法 JSON、过大请求体、不允许的 Origin/Host 返回明确错误；不泄露绝对路径与堆栈。
- 缺文件、工件哈希不符或真实推理异常时失败关闭，不用已保存参考概率回退。
- 选择与结果状态同源；概率数字和条形图共用原始数值，只在显示时舍入。
- 未主动揭示前，案例描述、输入快照、其他页面、工具提示或错误标签均不暴露真实赢家。
- `uncalibrated` 如实显示；CT/T 是阵营，不伪造稳定队伍身份。

## 正式指标与参考数据来源

下列路径已只读确认；实现时由固定映射读取，不由浏览器传路径：

| 用途 | 固定来源 |
| --- | --- |
| XGBoost 购买结束正式指标 | reports/esta_full_m10/m10_summary.json 的 test_selected_metrics，交叉核对 M14 验收 |
| XGBoost 首杀后正式指标 | reports/esta_full_m18/m18_summary.json 的 calibration.selected_test_metrics，交叉核对 M21 验收 |
| LightGBM 购买结束正式指标 | reports/esta_full_m27/fixed_test_metrics.csv |
| LightGBM 首杀后正式指标 | reports/esta_full_m33/fixed_test_metrics.csv |
| 样本量与范围 | 对应正式预测文件及冻结 split；购买结束 4,172，首杀后 4,170，界面分开标明 |
| 三例参考概率与输入 | 案例预检文档中列明的四份 CSV 与两份 parquet |

技术页使用各自完整正式测试集指标，不用三个案例计算整体准确率；两个时点的数值也不当作同一输入上的算法公平对照。

## 验证安排

- **P0**：来源唯一性、共同 test、标签和基础字段一致；保留候选规则及失败处理策略。
- **P1**：先写失败测试，再实现单条 Predictor 适配及 HTTP；真实概率参考比对；浏览器点运行；非法请求和故障失败检查。
- **P2**：扩展为 12 项集成矩阵，绝对误差仍为 1e-8；验证四次对比请求、部分失败及过期响应隔离。
- **P3**：跨视图状态、主动揭示、真实数据来源、空数据和指标范围；图表不写死值、不伪造解释。
- **P4**：完整 unittest、compileall；1440/1280 宽度三页截图和实际交互；安全、服务停止、模型缺失/哈希错误；保存证据与复现步骤。

本计划已获批准；具体任务已写入 [todo](todo.md)，每项明确依赖、实现、验收、验证命令、文件和证据，并控制主要改动规模。
测试与新增行为同步推进，不先一次性写完整应用再补测试。

## 风险与处理

| 风险 | 处理 |
| --- | --- |
| 历史测试概率与新环境推理不一致 | 先排查模型/校准器/特征/环境，不能重训或静默放宽误差标准 |
| 两时点回合错位或后续信息进入输入 | 唯一键、共同 test 和允许字段白名单检查；首杀后仍沿用购买结束状态 |
| 分析对比图偷用参考表 | API 逐项真实调用验证；未运行组合保留空态 |
| 请求竞态、跨页状态不同步 | 从 P1 建立组合键、请求代次和状态；P2/P3 扩展测试 |
| 炸弹、位置、玩家等缺来源 | 第一版隐藏或暂无数据；不编造赢家或输掉回合的原因 |
| 换机缺 Git-ignored 工件 | 启动预检清楚说明缺项；本计划不把 GitHub 当完整备份，也不在本轮迁移 |

## 本轮审阅结论与后续关卡

- 已完成：用户规格审核、只读案例预检、实施计划与 P0–P4 顺序审核。
- 审核依据：用户于 2026-09-05 明确回复“确认实施顺序 并上传GitHub”。
- 已执行：T01–T05 通过完整 12 项真实推理参考核对、全套 316 项测试；T04 浏览器操作验收保持为此前记录。
- 下一步：T06 在网页开放三个案例、两时点、两算法切换和运行全部对比；本轮停在 T05。
- 同步记录：此前授权的四份规划文档已提交并推送为 eb0a0e3；本轮 T01–T05 实现未提交或推送。
- 本轮已启动仅本机可访问的最小观众预览，未重训正式模型、未修改旧网页、未录视频；单链通过不代表三页及 v1 全部完成。
- 后续提交/推送、录制/发布视频、公开部署、新依赖或扩大数据范围仍保留规格中的先询问边界。

---

以下 M27–M33 为历史计划原文，保留追溯，不作为 ROUNDCAST 当前执行清单。

# M27-M28 Implementation Plan

## Scope

First close the frozen pre-round LightGBM line with M27 final acceptance. Then create
independent teacher-review reports for the accepted pre-round XGBoost, pre-round
LightGBM, and post-first-kill XGBoost lines. Only after those reports are verified,
start M28 by replacing the accepted first-kill M21 XGBoost algorithm with a fixed
LightGBM baseline while retaining data, grouped split, prediction point, features,
and metrics. The fourth report must use completed LightGBM evidence, never placeholders.

## Slices

1. Freeze M27 inputs, blockers, reproduction modes, outputs, and no-fit policy.
2. Add failing M27 contract tests, implement replay and acceptance, then run the
   formal real-artifact gate.
3. Commit and push M27 before opening the first-kill LightGBM stage.
4. Build and verify three independent frozen-result reports with one consistent
   review structure; do not compare different prediction times as algorithm effects.
5. Freeze M28 baseline parameters, metrics, paired-series uncertainty, and
   acceptance thresholds before training.
6. Add failing M28 tests, implement the controlled baseline, train using train with
   validation-only early stopping, and evaluate test exactly once.
7. Complete the post-first-kill LightGBM evaluation and acceptance, then write its
   independent report and the teacher review index.
8. Run focused/full tests and compile checks, verify manifests and links, document,
   commit, and push the report deliverables and LightGBM stages.

## Risks

- Test leakage: no test metric can appear in M27 replay selection or M28 training.
- Contract drift: require exact M21 first-kill rows, keys, split, and feature order.
- False superiority: paired confidence intervals govern claims, not point metrics.
- Artifact drift: hash model, calibrator, data, summaries, and outputs.
- Scope mixing: M27 must pass and be committed before M28 implementation begins.
- Report drift: every number and hash must trace to a frozen machine-readable artifact.
- Timing confusion: pre-round and post-first-kill metrics are not fair algorithm comparisons.

## Review Gates

- Gate A: M27 specification exists before M27 tests and code.
- Gate B: M27 focused tests fail for the missing module, then pass after implementation.
- Gate C: M27 formal artifacts, hashes, full suite, compile, commit, and push pass.
- Gate D: the first three teacher reports agree with their accepted source artifacts.
- Gate E: M28 specification exists before training code or model fitting.
- Gate F: the fourth report is created only after completed M28+ evidence exists.
- Gate G: all four reports and the index pass number, hash, link, and test checks.

## Outcome

M27, the first three teacher reports, and M28-M33 are complete. The fourth
post-first-kill LightGBM teacher report and teacher review index are now unblocked and
are the only remaining deliverables before the real-time track.

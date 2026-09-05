# ROUNDCAST v1 — 可执行任务清单（T01–T11 已验收）

更新日期：2026-09-06。对应 [实施计划](plan.md) 与 [已批准规格](../docs/roundcast_interactive_v1_spec.md)。

- [x] 用户审阅批准产品范围和技术规格。
- [x] 只读核对三个候选案例的共同测试集身份、输入合法性和四条参考概率。
- [x] 形成 P0–P4 实施计划，保留旧设计与历史任务。
- [x] 用户确认实施顺序与计划（2026-09-05）。
- [x] 首轮规格、计划、案例预检和进度文档按授权同步至 GitHub，提交 eb0a0e3。
- [x] 细化逐项可执行任务与验收步骤，形成下方 T01–T11 清单。
- [x] 用户确认从模型任务 T01 开始（2026-09-05）。
- [x] 用户批准并完成 T02–T04 单案例真实推理、本地接口与最小观众预览（2026-09-05）。

当前已实现并验收 T01–T11。T11 已交付三页 1440/1280 完整截图、六页图文 PDF、三个案例的操作说明及启动/关闭指南；12 项实际浏览器预测与参考误差为 0，三页共享记录、主动揭示、同端口重启推理均通过。T10 的 39 项 JavaScript 与全套 357 项 Python 回归记录保留，本轮未改应用逻辑。此前已同步至 T06（db6242a）；本次同步提交按用户授权补齐 T07–T11。详见 [T11 交付验收](../reports/roundcast_interactive_v1/t11_verification.md) 与 [案例演示指南](../reports/roundcast_interactive_v1/demo_guide.md)。

## 第一段做到哪里可以预览

### 附加项 C01–C04：Codex 回合对话（2026-09-05）

- [x] C01：固定只读解释范围、同源接口与上下文/隐私边界。
- [x] C02：以本机登录的 Codex CLI 接通临时对话、可信预测绑定、限额、停止及超时。
- [x] C03：模型下方加入灰白对话区、快捷问题、上下文标记、追问和清空。
- [x] C04：自动化测试、桌面/移动布局及一次真实网页回复核验通过。

证据：[对话验收](../reports/roundcast_interactive_v1/chat_verification.md)。这是额外完成的功能，主线顺序不变。

完成 T01–T04 后，可以在本机打开最小观众网页，选择案例 A（Ancient R4），点击运行购买结束 XGBoost，看到真实模型胜率、来源与错误状态。
这时只宣告单案例链路可用，不把尚未实现的三个完整页面或全部模型组合标成完成。

T05–T06 扩到三例、两个时点、两算法；T07–T09 补齐三页；T10–T11 统一验收与交付。

## 执行规则

- 按下面依赖顺序实施。每项先有失败测试或失败的浏览器操作检查，再实现并复验。
- 复用现有四个 Predictor 和输入验证器，不改冻结模型、校准器、阈值、训练数据或旧静态样板。
- 文件列表包含主要实现与测试文件；每次保持小改动，日志/截图另按阶段保存，不把多项任务合并成一次大改写。
- 参考 CSV 只用于验收断言，不向页面提供替代真实推理的成功结果。
- 每项完成才勾选，并记录测试命令、结果和证据；不因文档齐全就宣告网页完成。
- 当前 T01–T11 与回合对话已实现，本地交互演示 v1 交付完成。
- 本任务清单确认后进入实现；不再重开已批准的产品范围或 P0–P4 顺序。发现需要新增依赖、改变模型或扩大范围时再提出。
- 2026-09-05 用户再次授权“上传 github 下一个阶段”：先同步已有完成内容，再实现并保存 T06。录制视频、公开部署和换机备份不在本轮范围内。
- 2026-09-05 用户继续“下一步”：实施 T07 比赛分析视图和共享导航；本轮不自动提交/推送或进入 T08。
- 2026-09-05 用户继续“一直到 t09”：实施并验证 T08 正式指标 API 与 T09 技术视图；完成后停在 T09，不自动执行 T10/T11、提交或推送。
- 2026-09-05 用户再继续“下一步”：完成 T10 统一验收及小范围缺陷修复；本轮停在 T10，不自动执行 T11、提交或推送。
- 2026-09-06 用户继续“下一步 网页”：完成 T11 截图、图文文档、三个案例流程及实际重启复验，交付本地演示 v1。
- 2026-09-06 用户明确要求“上传至 github”：将已完成的 T07–T11 网页、测试、证据与展示材料同步到现有仓库 main，不纳入 Git 忽略的模型/数据或登录凭据。

## T01 — 固定可信工件与三个案例（P0）

- [x] 完成 T01（2026-09-05）。
  - 依赖：任务清单获批。
  - 文件：`examples/roundcast_v1_cases.json`、`src/csdemo/roundcast_service.py`、`tests/test_roundcast_service.py`。
  - 实现：登记四条模型/校准器映射、八个冻结工件与两份输入 parquet 的可信哈希来源及初始哈希、三例唯一键；从正式 parquet 提取允许的 27/31 个字段。身份、结果和参考概率独立存放。
  - 验收：三例均是两个时点共同 test，唯一键无歧义、标签一致、27 个购买结束字段一致并通过已有校验；缺文件、未知案例、重复键或坏哈希明确失败。哈希验证前不加载 joblib。
  - 验证：`& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_roundcast_service.py" -v`；先观察缺少新服务的测试失败，再实现并通过。
  - 证据：来源路径/哈希及三例资格结果。此项通过不代表已经执行模型推理。
  - 结果：17 项集中测试通过；全套 295 项测试通过；compileall 与 git diff --check 通过。十工件均与历史 manifest 条目一致，三例均通过 27/31 字段和共同 test 检查。
  - 记录：[机器可读来源与资格证据](../reports/roundcast_interactive_v1/t01_readiness.json)、[验收说明与命令](../reports/roundcast_interactive_v1/verification.md)。三份辅助来源为本次初始哈希登记，不冒称历史 manifest 已绑定。

## T02 — 案例 A 的真实购买结束 XGBoost 推理（P1）

- [x] 完成 T02（2026-09-05）。真实 CT=0.6291529536247253，与冻结参考误差为 0；22 项服务测试通过。
  - 依赖：T01。
  - 文件：`src/csdemo/roundcast_service.py`、`tests/test_roundcast_service.py`。
  - 实现：受限 `predict_example` 入口，加载正确冻结模型及校准器，实际调用 `PreRoundPredictor.predict`；返回案例/时点/模型/请求标识、原始概率、阵营、阈值、`uncalibrated`、特征版本、输入验证结果、哈希摘要和耗时。
  - 验收：案例 A 的 CT 概率与正式参考 `0.6291529536247253` 绝对误差不超过 `1e-8`；T 概率为补数。改动测试用参考值不能改变真实预测，真实调用异常不能退回参考概率。
  - 验证：运行 T01 同一集中测试命令，新增真实工件集成测试与缺模型/坏哈希/非法组合用例，保留先失败后通过证据。
  - 证据：真实 Predictor 响应及参考误差；不得仅凭响应字段形状或页面百分数判定接通。

## T03 — 本地 HTTP 与安全边界（P1）

- [x] 完成 T03（2026-09-05）。8 项实际 localhost HTTP 测试通过；仅 127.0.0.1，静态白名单和拒绝边界已验收。
  - 依赖：T02。
  - 文件：`src/csdemo/roundcast_server.py`、`src/csdemo/roundcast_service.py`、`tests/test_roundcast_server.py`。
  - 实现：Python 标准库服务，只监听 `127.0.0.1`；提供 models、examples、案例详情、单项 predict、独立 outcome 接口和明确的静态资源白名单。未接通组合标明未就绪或拒绝执行。
  - 验收：服务器只接受固定案例/时点/算法；拒绝未知字段、非法 JSON、过大请求体、不允许的 Origin/Host 和路径穿越；不泄露本机路径/堆栈。案例详情无赢家，含标签/参考概率的案例 JSON 不能作为静态文件下载。
  - 验证：`& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_roundcast_server.py" -v`；使用真实临时 localhost 服务完成成功与拒绝请求，测试结束关闭服务。
  - 证据：成功预测 HTTP 响应、错误状态码和静态资源拒绝结果；暂无前端不冒充可预览成品。

## T04 — 最小灰白观众网页（P1，第一个预览点）

- [x] 完成 T04（2026-09-05）。2 项 Python 页面合同测试（其中执行 6 项 JavaScript 状态测试）及实际 Chrome 点击/断开检查通过；1440/1280 截图已检查，全套 310 项通过。
  - 依赖：T03。
  - 文件：`web/roundcast/index.html`、`web/roundcast/app.js`、`web/roundcast/styles.css`、`tests/test_roundcast_browser_contract.py`。
  - 实现：案例 A 的回合信息、购买结束装备/现金、运行按钮、真实胜率条和来源；默认隐藏赢家，显式点击后揭示。前端从第一版就统一选择组合、请求代次和待运行/运行中/成功/失败状态。
  - 验收：浏览器点击确实调用 HTTP 并返回 T02 的真实概率；条形图和数字同源；重复点击、连接中断和失败不会把旧成功值当新结果。只开放就绪组合，缺失玩家/位置/炸弹数据不保留假数值。
  - 验证：`& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_roundcast_browser_contract.py" -v`；另启动 `& "C:\Users\admin\11\envs\game\python.exe" -m src.csdemo.roundcast_server --host 127.0.0.1 --port 8765`，按同一浏览器步骤记录实现前失败与实现后成功。静态检查不能代替实际点击。
  - 证据：页面截图、实际预测请求/响应、主动揭示与服务断开检查；完成后首次给用户可用的本机预览入口。

## T05 — 扩展四条模型与 12 项推理矩阵（P2）

- [x] 完成 T05（2026-09-05）。12 项真实 HTTP 预测与正式参考误差全部为 0；27 项服务测试、9 项 HTTP 测试、全套 316 项测试通过。详见 [验收](../reports/roundcast_interactive_v1/t05_verification.md) 与 [完整矩阵](../reports/roundcast_interactive_v1/t05_prediction_matrix.json)。
  - 依赖：T04 已证明单链端到端。
  - 文件：`src/csdemo/roundcast_service.py`、`tests/test_roundcast_service.py`、`tests/test_roundcast_server.py`。
  - 实现：加入其余三条 Predictor，统一外部 `post_first_kill` 与内部 XGBoost `first_kill` 映射；开放三个固定案例及全部有效组合。
  - 验收：三个案例乘四链共 12 项真正推理，各自与正式文件参考误差不超过 `1e-8`；元数据指向对应模型/校准器，首杀后仍使用购买结束状态加首杀字段。缺文件/坏哈希不回退。
  - 验证：分别运行 service 和 server 集中测试；断言完整 12 项而非随机一项，验证非法或交叉错配时点拒绝。
  - 证据：12 行概率、参考值、误差与对应模型哈希的结果矩阵。

## T06 — 案例/时点/算法选择与运行全部对比（P2）

- [x] 完成 T06（2026-09-05）。三例 × 两时点 × 两算法均通过真实浏览器推理；四项对比、单项失败重试、快速切换与迟到响应隔离通过。全套 344 项 Python、26 项 JavaScript 测试通过。详见 [验收记录](../reports/roundcast_interactive_v1/t06_verification.md)。
  - 依赖：T05。
  - 文件：`web/roundcast/index.html`、`web/roundcast/app.js`、`web/roundcast/styles.css`、`tests/test_roundcast_browser_contract.py`。
  - 实现：三例、两时点、两算法选择；两节点时间轴；“运行全部对比”对当前案例逐项发出四次单项预测请求，不从参考表补图。
  - 验收：每项有未运行/运行中/成功/失败状态；部分失败、重试和快速切案例不会串结果。切换案例清除揭示状态；再次点击运行重新调用模型。购买结束视图不提前展示首杀字段，首杀后只显示截止该时点已知事实；两节点之间不插值成实时胜率。
  - 验证：集中浏览器合同测试加实际浏览器点击；检查四条请求组合唯一、均属当前案例；在可控延迟/失败场景中先复现旧响应问题再验证防护。测试替身只能用于故障用例，正式演示不用模拟预测。
  - 证据：全部选择组合的操作记录、四项对比请求、部分失败与过期响应未覆盖的结果。

## T07 — 比赛分析视图与共享导航（P3）

- [x] 完成 T07（2026-09-05）。五项新增 JavaScript 行为测试及两项 Python 合同/字段白名单测试通过；全套 346 项通过。真实浏览器 12 个点、六快照、前进后退、刷新、失败重试和运行中切视图通过。详见 [T07 验收](../reports/roundcast_interactive_v1/t07_verification.md)。
  - 依赖：T06。
  - 文件：`web/roundcast/index.html`、`web/roundcast/app.js`、`web/roundcast/styles.css`、`tests/test_roundcast_browser_contract.py`。
  - 实现：加入分析视图；真实两时点/两算法对比、装备差、只读输入快照。复用已有结果与运行全部对比入口，跨视图保留相同选择和结果身份。
  - 验收：导航、刷新或浏览器前进/后退不产生选项和结果错配；未运行组合仍为空；两时点变化不作因果说明。仅三例的图表标明范围，未揭示赢家前不出现错误案例标签或结局提示。
  - 验证：浏览器合同测试和实际跨视图操作；断言图表数据与对应 API 原始值一致，而不是只验布局。
  - 证据：同一案例在观众/分析视图的对应截图和组合标识、空态和无剧透检查。

## T08 — 正式评估指标 API（P3）

- [x] 完成 T08（2026-09-05）。四份正式文件与五项指标逐项一致，样本数分别 4,172/4,170；四项集中测试通过。缺指标或坏哈希只使该项不可用，真实模型仍能推理。[来源证据](../reports/roundcast_interactive_v1/t08-metrics-evidence.json)。
  - 依赖：T05；按顺序在 T07 后集成，可与前端布局只读准备并行。
  - 文件：`src/csdemo/roundcast_service.py`、`src/csdemo/roundcast_server.py`、`tests/test_roundcast_metrics.py`。
  - 实现：固定映射读取计划列明的 M10/M18/M27/M33 指标来源，提供 metrics 接口；返回模型、时点、样本量、评估范围和来源摘要。
  - 验收：五项指标逐一与对应冻结正式文件一致；购买结束 4,172、首杀后 4,170 分开标注；更换当前案例不改变整体指标；无文件或缺字段显示不可用，不用三个案例重新估计。
  - 验证：`& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_roundcast_metrics.py" -v`；含四来源真实读取、HTTP 和缺字段拒绝用例。
  - 证据：四模型指标/样本量/来源一致性记录。

## T09 — 精简技术分析视图（三页完整）

- [x] 完成 T09（2026-09-05）。精简技术页、三页导航、正式指标和本次运行记录分离、主动揭示后判定对错已实现。五项新增 JavaScript 状态测试（含受控加载先后顺序）、实际 Chrome 12 项组合及故障检查通过；全套 351 项通过。[验收记录](../reports/roundcast_interactive_v1/t08_t09_verification.md)。
  - 依赖：T07、T08。
  - 文件：`web/roundcast/index.html`、`web/roundcast/app.js`、`web/roundcast/styles.css`、`tests/test_roundcast_browser_contract.py`。
  - 实现：技术视图展示当前模型版本、哈希摘要、验证状态、推理耗时、正式指标；主动揭示后可解释当前案例正确/错误。
  - 验收：三个视图选择和结果统一；指标属于所选模型与时点；无虚构特征贡献、校准曲线或技术指标；校准方式如实显示。页面不出现受众之外的“教师/老师”称谓。
  - 验证：浏览器合同测试和三页往返操作，核对当前响应与指标接口。调整任一选项或揭示状态不会让另一页显示旧案例。
  - 证据：三页一致性、未运行/失败/已揭示状态检查和无假数据审查。

## T10 — 全套功能、安全与故障验收（P4）

- [x] 完成 T10（2026-09-05）。79 项集中测试、全套 357 项 Python 测试（其中运行 39 项 JavaScript 测试）、编译及实际浏览器检查通过。12 项真实预测误差为 0；27 份已登记文件哈希未变；24 项工件故障、指标来源故障、异常请求、迟到响应和实际停服/恢复检查通过。详见 [T10 验收](../reports/roundcast_interactive_v1/t10_verification.md)。
  - 依赖：T01–T09。
  - 文件：主要维护 `tests/test_roundcast_service.py`、`tests/test_roundcast_server.py`、`tests/test_roundcast_metrics.py`、`tests/test_roundcast_browser_contract.py`；发现缺陷则单独作小修复并先补复现测试，不顺带重构模型代码。
  - 实现：补齐跨层验收测试，执行故障、安全和浏览器场景，汇总真实证据；缺陷按小改动分别修复并复验。
  - 验收：12 项参考比对、六个 API 类别、请求拒绝、服务停止、缺工件、坏哈希、快速切换、重复点击、部分失败、三页状态/揭示边界全部通过；浏览器无未捕获页面异常，正常流程无控制台错误。故意停服/返回 503 产生的预期网络错误单独留证，不混作程序异常；冻结工件/数据哈希不变。
  - 验证：`& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_roundcast_*.py" -v`；再执行 `& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_*.py"` 与 `& "C:\Users\admin\11\envs\game\python.exe" -m compileall -q src`；补实际浏览器故障操作，不能用截图代替推理证明。
  - 证据：完整命令/结果、浏览器检查、真实请求/模型来源、冻结哈希前后对比，统一记录到交付目录。不得在测试失败时仍声明 v1 完成。

## T11 — 截图、启动说明与案例演示步骤（P4）

- [x] 完成 T11（2026-09-06）。三页各两种桌面宽度完整截图、六页图文 PDF、三例演示及启动/关闭说明已交付。实际浏览器 12 项预测误差为 0；跨页记录/揭示一致，重启后得到新请求，页面异常与控制台错误均为 0。27 份已登记来源与本轮应用文件哈希不变。详见 [T11 交付验收](../reports/roundcast_interactive_v1/t11_verification.md)。
  - 依赖：T10。
  - 文件：`reports/roundcast_interactive_v1/README.md`、`reports/roundcast_interactive_v1/verification.md`、`docs/roundcast_interactive_v1_spec.md`、`tasks/todo.md`；截图为自动生成的交付附件，逐视图检查，不涉及原样板修改。
  - 实现：保存 1440/1280 两种宽度下三个视图的截图；编写启动/关闭、缺工件提示、三例逐步演示和能力限制；仅在证据通过后勾选规格完成项。
  - 验收：重新启动后按说明可重复所有案例；截图是真实运行状态，不是旧样板。截图中未运行/失败项如实显示；三例说明不夸大算法优劣或编造后续败因。
  - 验证：从说明重新走一遍启动、运行、三页切换、揭示和关闭流程；检查文档链接和截图可读性；核对最后一次代码变化后的测试仍有效。
  - 证据：三个案例操作步骤、真实输出来源、已知限制及可点击本机预览入口；完成后交付用户审阅。不自动录视频、发布或推送。

## 审阅点

上述任务均沿用已批准范围，没有新增模型、算法、预测时点或公开部署。
用户已批准并完成 T01–T11，三个视图与正式指标 API 已联动，统一验收及最终演示包通过。本地交互演示 v1 已交付；可按指南试讲或安排录屏。本次同步提交包含 T07–T11，历史阶段记录保持原样。

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

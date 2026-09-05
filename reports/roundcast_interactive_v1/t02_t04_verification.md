# ROUNDCAST T02–T04 验收

日期：2026-09-05。用户明确授权“做 t02 -t04”。结论：**T02、T03、T04 完成，停在 T04**。
当前只支持案例 A／购买结束／XGBoost；没有完成三页或全部 12 项推理。

## T02：真实推理

先新增测试，观察 `predict_example` 不存在和就绪状态失败，再实现受限入口。
对固定模型、校准器、购买结束 parquet 每次重查 SHA-256；用本次通过检查的模型 bytes 构造 BytesIO 并加载。
复用 `PreRoundPredictor.predict`，只传 27 项基础特征；不改冻结 Predictor、模型、校准器或训练数据。
精确确认 IdentityCalibrator 类型和 `uncalibrated` 方法，原始 CT 概率因此等于最终 CT 概率。

| 项目 | 本次结果 |
| --- | --- |
| 模型 | xgb_pre_round，M8 tuned / M10 identity |
| CT 概率 | 0.6291529536247253 |
| T 概率 | 0.37084704637527466 |
| 冻结参考绝对误差 | 0，标准保持 ≤1e-8 |
| 阈值 / 预测阵营 | 0.5 / CT |
| 输入 | 27 基础 + 9 派生，43 编码特征 |
| 工件及数据 | 10 个核心文件全部与 T01 相同，9 个辅助文件也相同 |

每次执行产生新请求 ID，带三项回合唯一键、源数据/模型/校准器哈希、特征版本、验证结果和耗时。
耗时包含文件校验、加载及推理，不能当作纯模型性能基准。
篡改测试用参考值并禁止运行中读取 CSV，真实概率仍不变；Predictor 异常明确失败，不回退为参考/缓存/默认概率。

独立审查注入“额外模型身份覆盖”和错误 probability_sum，复现成功状态的校验缺口；先加失败测试，再严格限制 Predictor 返回字段并核对概率和。正常模型结果不变。

## T03：本地 HTTP

先观察新 server 模块不存在；逐步加入路由、输入校验和安全边界，8 项实际 localhost 测试通过。

- 仅绑定 127.0.0.1；Host 必须为该地址加实际端口，Origin 如提供必须完全同源。
- 提供 models、examples、购买结束快照、单项 predict、独立 outcome；只有 A/pre/XGB 可执行。
- 非法 JSON、重复键、未知字段、未知值、异常类型返回 400；未实现组合 409；过大内容 413；错误 Content-Type 415；失败推理 503。
- 拒绝路径穿越、参数化文件路径及仓库目录下载。只允许 index.html、app.js、styles.css。
- 含标签/参考值的注册表不可经 HTTP 下载；详情不含赢家或首杀后信息。
- 输出不泄漏异常绝对路径或堆栈；无跨域允许头，设置 no-store、nosniff 和 CSP。
- 原始 socket 测试复现 5000 位 Content-Length 导致断连；增加长度上限后返回明确 413，未调用模型。

此服务是本地演示用的标准库 HTTP 服务，不作为生产服务或完整抗拒绝服务系统。

## T04：真实浏览器与状态

先写 JS 状态测试和页面合同测试，观察页面/脚本缺失；实现后实际浏览器等待待运行状态先失败，再接通页面与 API。
最终使用本机已有 Node、Playwright 和 Chrome，无安装/下载。

实际浏览器证据：[t04-browser-evidence.json](t04-browser-evidence.json)。

- 初始无预测、无 outcome 请求；用户点击才执行。
- 点击运行返回真实 CT 值，数字与条形来自同一响应。
- 图形 CSS 序列化显示 62.9153%，源值为 62.915295362472534%；浏览器样式序列化检查采用 0.0001 个百分点误差，**模型参考标准仍为 1e-8，未放宽**。
- 主动揭示赛果；重新运行生成新请求标识。
- 延迟真实请求时，重复点击不能多发；运行中清空旧结果。
- 1440 和 1280 宽度无横向溢出；两张截图已人工视觉检查。
- 真正停止测试服务，再次预测与揭示都明确失败，旧概率/赛果清除；不是仅用网络 mock 模拟停机。
- 正常交互无控制台错误；断开后的连接拒绝作为预期故障单独记录。
- JS 状态测试还覆盖重新连接使慢旧请求失效、错组合/异常概率拒绝。

网页技能仅影响灰白布局、可读性、空态和错误处理，遵循用户本地限定，不注册或部署外部站点。另有可选页面工具复用同一个运行动作，支持检测后注册；Node 合同测试覆盖输入拒绝/共享状态/注销。当前浏览器没有验证原生 WebMCP registry，此非用户要求的附加能力不作为已实测浏览器功能宣传。

## 最终检查

在项目根目录执行：

```powershell
& 'C:\Users\admin\11\envs\game\python.exe' -m unittest discover -s tests -p 'test_roundcast_*.py' -v
# 32 tests, 3.411s, OK（包含 6 项 JavaScript 状态测试）
& 'C:\Users\admin\11\envs\game\python.exe' -m unittest discover -s tests -p 'test_*.py'
# 310 tests, 23.474s, OK
& 'C:\Users\admin\11\envs\game\python.exe' -m compileall -q src
# exit 0
& 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' scripts/verify_roundcast_browser.cjs
# passed：9 组真实浏览器检查，证据写入本目录
git diff --check
# exit 0
```

浏览器脚本可通过 ROUNDCAST_PYTHON、ROUNDCAST_PLAYWRIGHT、ROUNDCAST_BROWSER 环境变量指定新机已有环境；Python 页面合同测试可用 ROUNDCAST_NODE 指定 Node。运行网页本身不需要 Node 或 Playwright。

增量实现/TDD/调试技能用于先失败后修复与逐段验收。技能引用的通用 Definition of Done 文件仍缺失，使用已批准 T02–T04 逐项验收条件。依据规格，未执行技能通常建议的 Git 提交。
T01 的 readiness_report 仍是准备层证据，`inference_executed: false` 不代表之后没有推理；本次真实推理以浏览器响应和集成测试为证据。

保留旧静态设计，未重训、未改变模型和数据、未新增运行依赖、未提交/推送 GitHub、未录视频、未公开部署。

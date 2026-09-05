# ROUNDCAST T10 — 统一功能、安全与故障验收

日期：2026-09-05。结论：**T10 通过；T11 最终演示包尚未开始。**

本轮按“下一步”执行已批准的 T10 清单，补齐跨层检查并修复复现的问题。没有新增算法、预测时点、自由输入或运行依赖；没有重训、修改冻结模型/数据、同步 Figma、录视频或提交/推送 GitHub。原有 T07–T09 本地改动与历史证据保留。

## 1. 本轮修复

| 复现的问题 | 修复与验证 |
| --- | --- |
| 成功响应缺少耗时、验证记录或模型哈希时，页面可能先认定成功，再在渲染时异常 | 发布成功状态前验证运行元数据、概率/阵营一致性、模型组合与输入维度。缺字段的 200 响应也显示失败，不保留当前胜率、运行记录或图点。新增状态测试及实际浏览器注入检查。 |
| 重连期间旧就绪列表仍可用；切案例可能使重连结果过期，留下旧列表 | 重连立即清空旧列表，等待就绪前禁用案例/时点/算法切换；过期请求不能发布结果。新增延迟重连测试。 |
| 模型列表读取失败时，连纯视图切换也被阻止；畸形列表可能触发 includes 异常 | 纯视图切换不依赖模型列表，技术页仍能单独读取正式指标；验证列表字段类型，缺项不能伪装就绪。 |
| 受控替身返回额外嵌套诊断字段或错误类型时，部分元数据可通过后端验证 | 精确核对各冻结部署链的快照定义、验证字段、类型及树数；布尔值不能冒充概率和。HTTP 故障响应不泄露内部诊断信息。 |
| 注册表内同一时点的两份参考来源互换时，原检查不能证明模型与参考路径正确对应 | 固定四条模型各自的参考路径和概率列；内存互换测试明确拒绝，原始文件不改动。 |
| 指标 CSV 额外列可能被宽松解析为隐式索引；极大 JSON 整数可能触发溢出 | 严格检查 CSV 表头、行列数和指标名，并将非有限/溢出数值转为明确验证失败。四份指标及四份样本来源的缺失/坏哈希逐项验证。 |

以上问题来自异常响应或内存故障注入，不表示现有冻结模型被替换错误，也不表示正式指标已经算错。四模型真实输出及正式来源再次核对通过。CSV 改用严格解析后，二进制浮点末位可能与旧 pandas 读取相差一个浮点单位；与源文件值的差在 `1e-14` 内，显示指标未变。

测试按复现失败 → 小范围修复 → 集中通过 → 全套回归执行；另有只读独立复核，未发现剩余阻断项。

## 2. 验收结果

| 检查 | 本次结果 |
| --- | --- |
| ROUNDCAST 集中 Python 测试 | 79 项通过，7.779 秒 |
| 全套 Python 测试 | 357 项通过，26.133 秒 |
| JavaScript 状态/选择/分析/技术/对话测试 | 39 项通过，由 Python 浏览器合同测试调用；不额外加到 357 的总数中 |
| 编译与差异空白检查 | compileall、git diff --check 通过 |
| 六个核心 API 类别 | models、examples、snapshots、predict、outcome、metrics 均检查 |
| 三案例 × 两时点 × 两算法 | 12 项真实 HTTP 推理与对应冻结参考的最大绝对误差 **0.0**；请求标识各自独立 |
| 两时点输入 | 六份快照逐项核对，购买结束 27 项、首杀后 31 项；不含赛果或参考概率 |
| 冻结来源 | 27 份已登记模型、校准器、数据、manifest 及参考/指标文件的测试前后 SHA-256 均与固定值一致 |
| 核心工件故障 | 四部署链 × 模型/校准器/数据 × 缺失/坏哈希，共 24 项；均返回受控 503，校验失败前未加载 joblib |
| 指标来源故障 | 四份指标和四份样本文件各测试缺失/坏哈希，共 16 个子场景；只让对应指标不可用，恢复后可重新读取 |
| HTTP 拒绝边界 | 非法字段/身份、歧义头、跨源、异常长度、编码及 JSON 等请求被拒绝；不触发真实预测或泄露路径 |
| 实际 Chrome | 三视图联动、12 个真实图点/运行记录、正式指标、刷新/历史导航、部分失败/重试通过；1440/1280/390 宽度无整页横向溢出 |
| 浏览器故障与恢复 | 7 类检查通过，含不完整响应、重连、重复点击、迟到结果/赛果、实际停止服务并原端口重启 |

浏览器未捕获页面异常为 0。专用故障脚本记录了正常及恢复阶段控制台错误为 0；人为返回 503 或实际停服时出现的预期网络错误单独留在证据中，未隐藏，也不宣称断网时控制台完全无错误。

## 3. 停服与状态边界

故障脚本创建自己的临时本机服务，真正停止该进程，再在相同端口重启；不使用模拟成功结果验证恢复，也不停止用户正在使用的 8765 预览。

- 停服后再次运行：当前选中组合显示错误，当前胜率、运行记录及正确/错误判断清除。
- 停服后重新读取赛果和指标：分别显示不可用，不显示旧赛果或旧指标为本次读取结果。
- 同案例其他组合的既有成功结果仍是已完成的历史运行记录；不能把它们误读成停服后新算出的结果。本次截图保留三条先前成功记录，当前失败行显示失败。
- 服务恢复后重新连接、读取指标并运行，得到新的真实成功预测与请求标识。
- 页面不做后台持续在线监测：未操作前不会仅因为服务器刚关闭就主动清空所有内容。
- 切案例清空旧预测与揭示状态；旧案例迟到的赛果不得揭示当前案例。纯视图切换保留同案例结果。

## 4. 证据文件

- [真实 HTTP 12 项矩阵、24 项故障与前后哈希](t10-api-evidence.json)
- [比赛分析浏览器复测](t10-analysis-browser-evidence.json)
- [技术分析浏览器复测](t10-technical-browser-evidence.json) / [四模型正式指标来源](t10-technical-metrics-evidence.json)
- [浏览器异常响应、实际停服及恢复](t10-fault-browser-evidence.json) / [实际停服截图](t10-actual-service-stopped.png)
- 比赛分析：[1440](t10-analysis-analyst-1440.png) / [1280](t10-analysis-analyst-1280.png) / [390](t10-analysis-analyst-390.png)
- 技术分析：[1440](t10-technical-technical-1440.png) / [1280](t10-technical-technical-1280.png) / [390](t10-technical-technical-390.png)

本轮截图是验收附件，已检查灰白布局、空态和窄屏内部滚动；不代替 T11 的最终三视图截图与三个案例演示包。

## 5. 复验命令

在项目根目录的 PowerShell 中执行。以下路径为当前电脑路径，换机须按实际环境修改。

```powershell
Set-Location 'C:\Users\admin\Documents\Codex\2026-07-06\th\work\csdemo_round_prediction'
& 'C:\Users\admin\11\envs\game\python.exe' -m unittest discover -s tests -p 'test_roundcast_*.py' -v
& 'C:\Users\admin\11\envs\game\python.exe' -m unittest discover -s tests -p 'test_*.py'
& 'C:\Users\admin\11\envs\game\python.exe' -m compileall -q src scripts/verify_roundcast_acceptance.py
& 'C:\Users\admin\11\envs\game\python.exe' -m scripts.verify_roundcast_acceptance
git diff --check
```

前两份浏览器脚本需要 [启动说明](README.md) 中的本地服务已运行；第三份会自动创建、关闭自己的临时服务。使用本机现有 Node、Chrome 与 Playwright，不新增运行依赖。证据前缀保留旧 T07/T09 文件。

```powershell
$roundcastNode = 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$env:ROUNDCAST_EVIDENCE_PREFIX = 't10-analysis'
& $roundcastNode scripts/verify_roundcast_analysis_browser.cjs
$env:ROUNDCAST_EVIDENCE_PREFIX = 't10-technical'
& $roundcastNode scripts/verify_roundcast_technical_browser.cjs
Remove-Item Env:ROUNDCAST_EVIDENCE_PREFIX
& $roundcastNode scripts/verify_roundcast_fault_browser.cjs
```

测试替身仅用于模拟故障；正常成功及恢复使用真实模型。T10 没有向 Codex 发送新的真实问题，也未额外消耗远程对话额度；本轮验证的是既有对话边界自动测试，历史真实连接验收见 [对话记录](chat_verification.md)。

## 6. 当前交付边界

[本机预览](http://127.0.0.1:8765/) 保持可用，已打开页面需刷新以加载本轮修复。

下一项是 **T11：整理三个视图的最终截图、启动/关闭说明和三个可复现案例演示步骤**。仍不自动录视频或公开部署。GitHub 最近已同步阶段保持 T06（db6242a）；T07–T10 尚未提交/推送，GitHub 也不包含 Git 忽略的完整模型及数据备份。

# ROUNDCAST 本地交互演示 v1 — 实施规格

日期：2026-09-04。

状态：产品范围和技术规格已于 2026-09-05 经用户审阅通过（“好的 没问题 下一步”）；实施顺序也已获确认，且本轮文档同步 GitHub 已获授权（“确认实施顺序 并上传GitHub”）。下一阶段为具体任务拆分，尚未开始实现。现有静态页面、冻结模型、校准器与数据均不改动。下文启动命令和新增路径是实现约定，当前尚不存在，不代表网页已经可用。

## 1. Objective — 目标与范围

将已有灰白配色的三类受众设计转为 Windows 本机浏览器可运行的演示网页。三个视图分别面向比赛观众、比赛分析人员和技术分析人员；共享案例、预测时点、算法以及真实推理结果。

已确认的产品范围：

- 本地网页，不公开部署，不接实时比赛。
- 三个页面联动，保持相同的案例和选项。
- 仅支持购买结束、首杀后两个预测时点；每个时点分别支持 XGBoost 和 LightGBM。
- 三个来自冻结测试集的真实历史回合；两个时点必须是同一回合。
- 暂不开放自由编辑特征、上传数据或上传模型。
- 点击运行时调用已保存的模型，而非读取预先写好的概率冒充推理。

明确的技术假设：

1. 复用现有 Python 环境和四个 Predictor，增加一个仅监听 `127.0.0.1` 的轻量本地服务。
2. 前端使用 HTML/CSS/JavaScript；沿用灰白风格，不增加前端构建链或第三方在线脚本。
3. 本地 HTTP 层使用 Python 标准库，第一版不新增运行依赖。它仅为本机演示服务，不作为生产服务器。
4. 原设计作为独立归档保留；运行版使用新目录，不能把未接数据的占位值直接改名为真实数据。
5. 第一版提供可录屏的演示流程和说明，视频文件的录制不属于当前已确认范围。

## 2. Tech Stack 与冻结模型

当前环境已只读验证：Python 3.10.20，pandas 2.3.3，numpy 2.2.6，scikit-learn 1.7.2，XGBoost 3.2.0，LightGBM 4.6.0。导入成功不等于模型已完成运行时验收；实施时仍须真实加载和推理验证。

| 对外模型 ID | 预测时点 | 现有模型文件 | 现有校准器文件 |
| --- | --- | --- | --- |
| xgb_pre_round | 购买结束 | models/esta_full_m8_tuned/pre_round_xgb.joblib | models/esta_full_m10/pre_round_calibrator.joblib |
| lgbm_pre_round | 购买结束 | models/esta_full_m23/pre_round_lightgbm_tuned.joblib | models/esta_full_m24/pre_round_lightgbm_calibrator.joblib |
| xgb_post_first_kill | 首杀后 | models/esta_full_m17/first_kill_xgboost_tuned.joblib | models/esta_full_m18/first_kill_calibrator.joblib |
| lgbm_post_first_kill | 首杀后 | models/esta_full_m29/post_first_kill_lightgbm_tuned.joblib | models/esta_full_m30/post_first_kill_lightgbm_calibrator.joblib |

复用 `PreRoundPredictor`、`PreRoundLightGBMPredictor`、`FirstKillPredictor` 和 `PostFirstKillLightGBMPredictor`。外部统一使用 `post_first_kill`，内部适配 XGBoost 的 `first_kill` 命名，禁止串用模型。

启动前校验八个可信本地工件的冻结 SHA-256，再加载；不接受客户端提供文件路径。四条部署链当前使用 identity 校准，界面应如实显示 `uncalibrated`，不能称为经过非恒等校准后的概率。

## 3. 用户操作与三个视图

共同流程：选择案例 A/B/C → 选择购买结束或首杀后 → 选择算法 → 运行模型 → 查看结果。切换页面保留选择；切换任一输入选项后，不得把上次结果显示为当前组合的新结果。显示待运行、运行中、成功或失败状态，防止慢请求覆盖更新的选择。

| 视图 | 第一版内容 | 数据规则 |
| --- | --- | --- |
| 比赛观众 | 双方阵营与回合信息、真实胜率条、两节点时间轴、装备/现金、首杀事实、主动揭示结果 | 数字和图形必须来自同一响应；真实结果与预测输入分离 |
| 比赛分析 | 两模型及两时点概率对比、装备差对比、只读输入快照、案例切换与结果复核 | 两时点差异称为预测变化，不称为首杀的因果效果；仅三个案例的图必须注明样本范围 |
| 技术分析 | 当前模型/版本、工件哈希摘要、输入验证、推理耗时、正式测试集指标、案例错误提示 | 正式指标不能从三个展示案例重新计算；没有真实解释产物就不展示假特征贡献 |

页面内不出现“教师”或“老师”等受众称谓。技术页保持精简。三页都应有清晰的“历史回合演示 / 非实时比赛”标识。

默认观众状态不提前展示最终赢家；“查看实际结果”是显式操作。案例名称使用 A/B/C 等中性名称，避免在运行前以“预测错误案例”等名字提前暴露答案。技术分析中可在揭示后解释选例类别。

未取得可靠来源的玩家姓名、个人数据、地图位置、烟雾轨迹、炸弹状态和整场比分趋势，第一版隐藏或显示暂无数据，不沿用静态样板中的虚构数字。可追溯的完整比赛数据若后续获得，再单独确认接入范围。

## 4. 案例与时间边界

候选类型：A 常规正确预测；B 两时点预测变化较明显；C 高置信度错误或两算法分歧。固定挑选规则与回合键，在后续样例清单中保存来源、标签和四条参考概率，不能只保存漂亮的截图。

- 按 `series_id + game_id + round_id` 唯一对齐两个预测时点，要求标签一致，并同时属于各自冻结测试集。
- 三个案例应是三个不同回合，并全部通过现有输入校验。候选不符合条件时更换案例，不修改验证器迎合案例。
- 参考概率用于验收比对，不用于代替在线模型调用。
- 购买结束输入来自正式 pre_round 数据；首杀后仍为购买结束状态加首杀四个字段，不擅自换成首杀后剩余库存或现金。
- 标签、最终比分、之后的击杀/下包/拆除等未来事实不能进入模型输入。
- 时间轴只有两个可推理节点；不插值生成逐秒胜率，也不把点击烟雾、下包等事件当作新推理。
- 数据没有保留精确预测截止 tick 或选取 frame tick 时，明确“精确帧对齐未提供”，不伪造秒级同步保证。
- 不能默认以 5v4 等人数展示代替真实状态；仅在数据能证明时展示人数，否则只显示首杀阵营与首杀时间。
- 当前输出是 CT/T 回合获胜概率；没有可靠队伍身份与换边映射时，使用阵营名称，不把 CT/T 当作固定队伍。
- 三例是演示性筛选，不构成算法优劣、泛化或总体准确率的新证据。

## 5. 本地接口约定

同源提供网页与 API；只允许已声明的静态资源，不暴露仓库目录或任意文件下载。不得开放局域网监听、模型上传、任意路径读取或模型训练操作。

| 接口 | 用途 |
| --- | --- |
| GET /api/models | 返回四条部署链及其时点、就绪状态、版本摘要 |
| GET /api/examples | 返回三个案例的中性描述、地图、回合身份、可用时点 |
| GET /api/examples/{id} | 返回只读输入与有来源的回合状态；不混入真实赢家 |
| POST /api/predict | 仅接受 example_id、stage、algorithm；服务端取固定快照并执行 Predictor |
| GET /api/examples/{id}/outcome | 主动揭示已结束历史回合的真实赢家，用于复核 |
| GET /api/metrics | 读取四条冻结部署链的正式指标、评估样本数及来源 |

预测响应至少包含：请求标识、案例标识、模型 ID、时点、CT/T 概率、预测阵营、阈值、校准方式、模型/特征版本、模型哈希摘要、推理耗时、输入验证结果。记录标识仅用于追踪，不作为特征。

请求拒绝未知字段、无效 ID、不支持时点、未知算法、过大请求体、非法 JSON 和不允许的跨源访问；返回明确状态码与可理解错误。不向客户端泄露本机绝对路径或堆栈。模型/数据缺失或哈希不符必须停止相应推理，不回退为演示概率。

## 6. Commands — 命令约定

在项目根目录的 PowerShell 终端运行。Python 绝对路径仅为当前开发机示例，换机后需替换为对应环境路径。以下新模块和测试命令为实现目标，当前尚未提供：

```powershell
# 计划提供的本地启动方式
& "C:\Users\admin\11\envs\game\python.exe" -m src.csdemo.roundcast_server --host 127.0.0.1 --port 8765

# 计划新增的集中测试
& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_roundcast_*.py" -v

# 仓库 unittest 全套回归
& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_*.py"

# Python 语法编译校验
& "C:\Users\admin\11\envs\game\python.exe" -m compileall -q src
```

第一版没有独立前端 build 或 lint 依赖；浏览器加载即运行，不虚构已存在的 npm 命令。使用现有环境，不在不确认的情况下升级模型相关依赖。迁移电脑时路径需替换为新机器的对应环境。

## 7. Project Structure — 新增内容位置

```text
docs/roundcast_interactive_v1_spec.md     当前规格
tasks/plan.md                            规格批准后加入实施计划，保留现有记录
tasks/todo.md                            计划批准后加入分步任务，保留现有记录
src/csdemo/roundcast_service.py           受限模型注册、样例与推理服务
src/csdemo/roundcast_server.py            仅本机 HTTP 与静态资源适配
web/roundcast/index.html                 三个路由/视图共用的页面入口
web/roundcast/app.js                     共享选择、请求状态、图表及视图联动
web/roundcast/styles.css                 灰白样式与响应式布局
examples/roundcast_v1_cases.json          三个真实案例键和来源清单
tests/test_roundcast_*.py                合同、推理、HTTP 和静态入口测试
reports/roundcast_interactive_v1/         完成后保存验收记录、截图与演示步骤
```

路径是建议结构；具体文件切分在计划审阅阶段确认。原有 `reports/roundcast_design/` 不覆盖。

## 8. Code Style — 代码与输出风格

保持仓库 Python 风格：四空格、snake_case、类型提示、Path 路径、unittest。前端使用清晰的 camelCase 命名和原生 DOM API；文本使用 textContent 等安全写入方式，不将数据作为 HTML 执行。

最小风格示例（不是已实现代码）：

```python
def predict_example(example_id: str, stage: str, algorithm: str) -> dict:
    snapshot = resolve_trusted_snapshot(example_id, stage)
    predictor = resolve_frozen_predictor(stage, algorithm)
    result = predictor.predict(snapshot)
    return add_prediction_metadata(result, example_id, stage, algorithm)
```

显示概率时统一格式；图表与数字共用未提前舍入的原始值。缺失为 null/暂无数据，不当作 0。加载错误要明确提示，不能悄悄保留别的案例结果。

## 9. Testing Strategy — 测试与验收

遵循测试先行：每个新行为先写可复现的失败测试，再实现并观察通过。沿用 unittest，不为测试引入新框架。

- 小测试：模型/时点映射、严格请求校验、回合键、错误状态、输入字段隔离和展示数据格式。
- 集成测试：可信工件哈希后加载、三个案例全部四条部署链真实推理、相应 HTTP 请求、缺文件和非法请求拒绝。
- 参考一致性：三例乘四模型共 12 项概率，与冻结正式预测文件对齐，绝对误差不超过 1e-8；如存在环境差异，调查原因后再决定是否调整标准，不静默放宽。
- 浏览器验证：实际启动本地服务；三页、三例、两时点、两算法可切换；页面切换保留选择；改变选项不显示过期结果；运行失败、重复点击、快速切换没有错误覆盖；无控制台错误。
- 视觉验证：桌面宽度 1440 和 1280 下检查三页截图，缩放或窄窗口允许正常滚动；不截断主要控件和图表；无来源的占位数字不残留。
- 回归：完整 unittest 与 compileall 通过；冻结模型和数据哈希保持不变。

不把文字匹配或静态截图当成端到端接通证明；必须记录真实 API 请求、模型结果、模型来源及浏览器操作证据。

## 10. Boundaries — 行为边界

**Always：** 分步实现与验证；保持四条冻结部署链不变；校验可信工件后加载；固定测试案例可复现；区分回放事实、模型输出、派生统计与人工说明；保留旧设计。

**Ask first：** 新依赖或环境升级；接入真实比赛/全场轨迹；公开部署；新增预测时点或特征；自由输入模拟；重新训练；录制或发布视频；提交或推送 GitHub。

**Never：** 上传凭据；暴露任意模型路径；加载上传的 joblib；修改冻结结果迎合页面；用示例数字冒充推理；插值伪造逐秒胜率；泄漏未来事件进入特征；覆盖用户现有未提交改动。

## 11. Success Criteria — 完成标准

- [ ] 一条本机命令可启动网页；关闭服务后网页如实显示连接不可用。
- [ ] 三个视图保持案例/时点/算法一致，选择与结果组合明确。
- [ ] 三个不同真实回合通过唯一键对齐两个时点且均属于冻结测试集。
- [ ] 全部 12 项真实推理通过参考概率比对。
- [ ] 点击运行确实调用模型，不能以参考概率表代替。
- [ ] 技术指标可追溯到各自正式评估集，不使用三个案例估计整体表现。
- [ ] 未实现数据项明确缺失；时点、阵营、输入截止及事后结果不混用。
- [ ] 新测试、全套回归、编译及实际浏览器检查通过并记录。
- [ ] 提供三页截图、启动/关闭说明及三个可复现演示步骤。

## 12. Review Gate — 当前审阅点

2026-09-05：用户已批准本规格及 [实施计划](../tasks/plan.md) 的 P0–P4 顺序，并授权将本轮规格、计划、案例预检和进度文档同步至 GitHub。下一阶段细化可执行任务及验收步骤；计划与任务保留仓库原有 M27–M33 历史记录。批准与文档同步不等于网页已实现，也不包含训练、公开部署或发布视频。

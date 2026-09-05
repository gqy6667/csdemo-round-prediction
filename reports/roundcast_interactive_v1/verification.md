# ROUNDCAST v1 — 验收索引

当前状态（2026-09-05）：T01–T05 已完成，T06–T11 尚未实施。
最新的 12 项真实推理、HTTP 及全套 316 项测试证据见 [T05 验收记录](t05_verification.md)。
此前单案例浏览器验收见 [T02–T04 验收记录](t02_t04_verification.md)。
本机预览和操作方式见 [启动说明](README.md)。仅本地保存，未新增提交/推送。

---

# T01 验收历史记录（以下保留当时状态）

日期：2026-09-05。范围：**只完成 T01 可信工件与三个案例准备**。

结论：T01 通过。T02–T11 尚未实现；没有本机网页入口，也没有本次演示模型推理结果。
本轮未提交或推送 GitHub，未更改旧静态设计、正式模型、校准器或两份输入 parquet。

## 交付文件

- [固定案例、来源及初始哈希](../../examples/roundcast_v1_cases.json)
- [可信来源与只读案例服务](../../src/csdemo/roundcast_service.py)
- [17 项集中测试](../../tests/test_roundcast_service.py)
- [机器可读验收证据](t01_readiness.json)
- [后续任务清单](../../tasks/todo.md)

## 来源核验

四份历史 M14/M21/M27/M33 manifest 本身按固定 SHA-256 验证，随后检查指定条目的路径、大小和哈希。
八个模型/校准器及两份输入 parquet 共十个核心文件全部一致。读取 parquet 使用已验证的字节，不在验证后重新打开另一份输入。

| 内容 | 数量 | 结果 |
| --- | ---: | --- |
| 固定历史 manifest | 4 | 本身哈希匹配 |
| 模型与校准器 | 8 | 指定历史条目及初始哈希匹配 |
| 两时点输入 parquet | 2 | 指定历史条目及初始哈希匹配 |
| 辅助 split、参考 CSV、类别审计 | 9 | 已登记并核验哈希 |
| 带历史 manifest 条目的文件合计 | 16 | 精确条目匹配 |
| 本轮新增初始哈希登记 | 3 | 与未修改的 Git tracked 基线核对 |

三份本轮初始登记文件为 M21 `split_assignments.csv`、M10 和 M18 的 `calibrated_test_predictions.csv`。没有发现它们在旧 manifest 中的独立文件哈希条目，因此没有冒称它们早已受到该项历史绑定。基线提交为 `eb0a0e348826398df479a7cdcc93c1cc713efba3`。

当前正式首杀输入使用 M33 的 `inputs[0]` 绑定，不使用 M14 较早的首杀数据哈希。这里只验证 T01 指定来源，不声称重做了所有历史阶段的整份验收。

这些初始哈希是本地来源控制下的信任基线，不是数字签名。换机若文本换行方式改变导致哈希不符，应核查来源，不能自动改写可信值或跳过检查。

## 三个案例

| 案例 | 地图/回合 | 两时点共同 test | 四份 series split | 输入字段 | 标签/购买状态一致 |
| --- | --- | --- | --- | --- | --- |
| A | Ancient R4 | 通过 | 全部 test | 27 / 31 | 通过 |
| B | Mirage R11 | 通过 | 全部 test | 27 / 31 | 通过 |
| C | Nuke R17 | 通过 | 全部 test | 27 / 31 | 通过 |

以 `series_id + game_id + round_id` 对齐。两个 parquet 的重复回合键、split 文件的重复 series、注册表重复案例 ID/回合键均拒绝。
现有 `validate_snapshot` 与 `validate_first_kill_snapshot` 均通过；它们内部可计算派生特征，但对外只读基础输入仍严格保留 27/31 个白名单字段。

身份、真实结果与参考概率不进入输入。案例列表为中性名称，真实赢家仅通过独立 `outcome()` 读取。首杀后输入沿用购买结束状态，不将其描述为首杀后实时库存。
调用方修改返回快照不会改变服务保存的案例；修改注册表中的参考概率不能改变快照或案例列表。

四份正式预测 CSV 中三例的身份、`y_true` 和十二个参考概率已核对（原始浮点差不超过 `1e-15`）。这些只是历史参考数据，不是本次模型调用结果；T02 起实际推理仍按已批准的 `1e-8` 误差验收。

## 测试先行与修复证据

1. 先增加集中测试，执行时因服务尚不存在出现 `ModuleNotFoundError: No module named 'src.csdemo.roundcast_service'`；退出码 1。
2. 加入最小来源层与案例准备，再验证十文件、三例和故障场景。缺失注册表阶段明确失败，注册表创建后首批 9 项通过。
3. 新增重复 JSON 键检查，先观察错误类型不符，再实现重复键拒绝。
4. 参考表核对测试曾误用输入字段 `ct_win`；检查四份原始 CSV 表头，确认标签均为 `y_true` 后修正测试。未修改历史数据，也未放宽参考误差。
5. 独立审查复现顶层注册表、`files`、`manifest_pins` 为数组时抛出裸 `AttributeError`。补失败测试后，入口现统一返回 `RoundcastValidationError`。
6. 补测试复现参考来源省略可信 pin 仍可初始化的问题，现要求所有引用来源均登记可信文件哈希。
7. 修复后独立复核及最终集中测试均通过。

覆盖：十文件验证、四模型映射、未知案例/时点/算法、缺文件、实际字节破坏、坏初始哈希、坏 manifest 哈希/条目、越界路径、重复 JSON/案例/回合/series、非 test、标签错位、两阶段购买字段不一致、非法首杀时间、快照隔离、参考值来源及不参与输入。

T01 路径使用测试替身禁止 `joblib.load`，确认准备期间没有反序列化模型。全套回归中原有模型测试仍按仓库惯例运行，不能把它们当成新网页推理已完成的证据。

## 最终执行结果

在项目根目录 PowerShell 运行：

```powershell
& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_roundcast_service.py" -v
# Ran 17 tests in 1.665s — OK

& "C:\Users\admin\11\envs\game\python.exe" -m unittest discover -s tests -p "test_*.py"
# Ran 295 tests in 19.873s — OK

& "C:\Users\admin\11\envs\game\python.exe" -m compileall -q src
# exit 0

git diff --check
# exit 0
```

机器可读准备证据可重新检查，不加载模型：

```powershell
& "C:\Users\admin\11\envs\game\python.exe" -c "import json; from src.csdemo.roundcast_service import RoundcastService; print(json.dumps(RoundcastService().readiness_report(), indent=2))"
```

本机运行环境沿用已有 `game` Python，不增加依赖或前端构建工具。
增量实施技能引用的通用 Definition of Done 文件在本机缺失，本轮采用已批准的项目 T01 验收条件、集中测试、全套回归及编译检查完成验收。
按已批准规格的边界，本轮未执行技能通常建议的提交动作；提交和推送仍需单独确认。

## 下一项

T02：只接通案例 A 的购买结束 XGBoost。真正调用 `PreRoundPredictor.predict`，核对 CT 概率与 `0.6291529536247253` 的绝对误差不超过 `1e-8`；失败不能退回参考概率。
T04 才提供第一个可预览的真实网页链路。当前所有模型元数据仍标记 `inference_ready: false`。

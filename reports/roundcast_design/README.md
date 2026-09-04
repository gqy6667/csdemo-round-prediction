# ROUNDCAST 三类受众界面设计

归档日期：2026-09-04。本目录保存最新的灰白色静态样板与功能说明。

## 查看入口

- [六页设计说明 PDF](roundcast-three-audience-design-review-v1.pdf)
- [观众页面样板图](roundcast-viewer-mockup-v2.png)
- [比赛分析页面样板图](roundcast-analyst-mockup-v1.png)
- [技术分析页面样板图](roundcast-technical-mockup-v1.png)

GitHub 可直接查看 PDF 和 PNG。要查看网页排版，请下载整个目录后在浏览器打开对应 HTML；GitHub 的文件页不直接运行 HTML。

| 受众 | HTML 文件 | 对应样板图 |
|---|---|---|
| 比赛观众 | [观众 HTML](roundcast-viewer-mockup-v1.html) | `roundcast-viewer-mockup-v2.png` |
| 比赛分析人员 | [分析 HTML](roundcast-analyst-mockup-v1.html) | `roundcast-analyst-mockup-v1.png` |
| 技术分析人员 | [技术 HTML](roundcast-technical-mockup-v1.html) | `roundcast-technical-mockup-v1.png` |

观众 HTML 沿用 v1 文件名，但内容已更新；最新截图为 v2。分析与技术 HTML 依赖同目录的 `roundcast-static-pages.css`，下载时请一起保留。

## 当前阶段与能力边界

- 这是设计确认阶段，不是已上线产品。
- 三页中的比赛数值、地图位置、图表和胜率均为设计演示，不是真实模型输出。
- 尚未连接模型、比赛数据、实时事件或实际交互。
- 比分阶梯图、回合色带、关键玩家突出显示、位置光环及炸弹状态等用于说明未来界面布局；静态图不代表动画或推理已经实现。
- 模型已有的预测时点仍是购买结束和首杀后。烟雾、下包、残局及连续胜率展示不代表这些时点已有模型。
- 技术页面采用精简布局。下一步先确认三页信息结构与视觉方案，再决定实施范围。

## 与模型报告的关系

[模型报告与补充材料索引](../teacher_review/README.md)保留冻结实验和审计证据。其中第 7 份可视化报告是前期方案研究，本目录是之后形成的三类受众静态设计；两者不应被理解为同一阶段的实现结果。

本次归档未包含个人工作记录、原始论文全文、临时构建脚本、过时截图或任何登录凭据。PDF 上传副本移除了文档作者等元数据，原始文件保持不变。

训练数据和大部分模型二进制仍按仓库的 `.gitignore` 留在本机，GitHub 同步不等同于完整的换机备份。

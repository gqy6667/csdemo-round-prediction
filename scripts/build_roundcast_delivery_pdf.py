"""Create the T11 Chinese demonstration guide from captured UI and real run evidence."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/roundcast_interactive_v1'
evidence = json.loads((OUT / 't11-delivery-evidence.json').read_text(encoding='utf-8'))
assert evidence['status'] == 'passed' and len(evidence['matrix']) == 12
pdfmetrics.registerFont(TTFont('YaHei', 'C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('YaHeiBold', 'C:/Windows/Fonts/msyhbd.ttc'))
W, H = 1280, 800
INK, MUTED, BLUE, LINE, BG = map(HexColor, ['#222a35', '#667584', '#627e9d', '#dde3eb', '#f4f5f7'])
pdf = canvas.Canvas(str(OUT / 'roundcast-v1-demo-guide.pdf'), pagesize=(W, H))
pdf.setTitle('ROUNDCAST v1 - 三页面与三案例演示指南')
pdf.setAuthor('ROUNDCAST')


def text(value, x, top, width=1160, size=19, color=INK, bold=False):
    style = ParagraphStyle('text', fontName='YaHeiBold' if bold else 'YaHei', fontSize=size,
                           leading=size * 1.5, textColor=color, wordWrap='CJK')
    paragraph = Paragraph(value, style)
    _, height = paragraph.wrap(width, H)
    assert top + height < H - 40, (value, top, height)
    paragraph.drawOn(pdf, x, H - top - height)
    return height


def page(number, title, subtitle):
    pdf.setFillColor(BG); pdf.rect(0, 0, W, H, fill=1, stroke=0)
    pdf.setFillColor(INK); pdf.setFont('Helvetica-Bold', 14); pdf.drawString(42, H - 35, 'ROUNDCAST / V1')
    pdf.setFont('YaHei', 11); pdf.setFillColor(MUTED); pdf.drawRightString(W - 42, H - 35, '本地交互演示 · 2026-09-06')
    text(title, 42, 61, size=32, bold=True)
    text(subtitle, 42, 113, size=15, color=MUTED)
    pdf.setStrokeColor(LINE); pdf.line(42, 41, W - 42, 41)
    pdf.setFont('YaHei', 10); pdf.setFillColor(MUTED)
    pdf.drawString(42, 24, '真实历史回合 / 已保存模型实际推理 / 页面截图未经模拟数据替换')
    pdf.drawRightString(W - 42, 24, f'{number:02d} / 06')


def card(x, top, width, height, title, body):
    pdf.setFillColor(HexColor('#ffffff')); pdf.roundRect(x, H-top-height, width, height, 14, fill=1, stroke=0)
    text(title, x+24, top+22, width-48, 23, bold=True)
    text(body, x+24, top+76, width-48, 18, MUTED)


page(1, '让一个回合的预测，可以看、可以切换、可以复核。', '三个联动页面，三个真实案例，购买结束与首杀后各运行 XGBoost / LightGBM。')
for x, title, body in [(42, '比赛观众', '看 CT / T 胜率、已知首杀事实和经济快照。通过两个时间节点切换输入时点。'),
                        (448, '比赛分析', '看同回合的四个预测点、两时点变化和原始输入。对照模型实际看到了什么。'),
                        (854, '技术分析', '看正式测试集指标与当前运行记录。揭示赛果后，复核这一项预测是否正确。')]:
    card(x, 184, 384, 225, title, body)
text('推荐操作顺序', 42, 449, size=23, bold=True)
text('选择案例 → 运行全部对比 → 切换三个页面 → 主动揭示赛果', 42, 498, size=25, color=BLUE)
text('切页面保留同一回合结果；切案例清空旧结果；每次运行生成新的模型调用记录。', 42, 558, size=19)
text('本次交付含：三页 1440 / 1280 完整截图、三个案例操作指南、启动说明及可复验记录。', 42, 604, size=17, color=MUTED)
text('网页入口：<link href="http://127.0.0.1:8765/" color="#627e9d">http://127.0.0.1:8765/</link>（在启动本地服务的电脑上打开）', 42, 656, size=19)
pdf.showPage()

screens = [
    ('viewer', '01 / 比赛观众', [('读当前预测', 'CT 39.91%，T 60.09%。数字、胜率条和来源属于同一次真实运行。'), ('看已知事实', '首杀后显示输入中已有事件。页面下方保留购买结束经济和两个时间节点。'), ('最后揭示', '此截图赛果隐藏；点击“查看实际结果”才进入结果复核。')]),
    ('analyst', '02 / 比赛分析', [('比较四个点', '上方为购买结束，下方为首杀后；同一时点可以比较两种算法。'), ('读预测变化', '案例 B 的 CT 概率下降 31.86 / 37.44 个百分点，不解释为独立因果效果。'), ('核对输入', '向下滚动可查看 27 / 31 项原始字段。经济数据一直对应购买结束。')]),
    ('technical', '03 / 技术分析', [('整体表现', '左侧五项指标来自完整正式测试集。首杀后有 4,170 个测试回合。'), ('本次运行', '右侧对应当前模型版本、输入验证和处理耗时；展开可看指纹及请求标识。'), ('单回合复核', '截图尚未评判。主动揭示后，当前组合显示正确或错误。')])]
for number, (view, title, notes) in enumerate(screens, 2):
    page(number, title, '案例 B · Mirage R11 · 首杀后 · LightGBM · 四项已运行 · 赛果未揭示')
    pdf.drawImage(str(OUT / f't11-{view}-1440-screen.png'), 42, H-172-554, width=760, height=554)
    for index, (heading, body) in enumerate(notes):
        text(heading, 842, 178+index*168, 386, 23, bold=True)
        text(body, 842, 222+index*168, 386, 18, MUTED)
    text('图为真实页面首屏；完整长图含下方四项对比、输入表或回合对话，见配套截图。', 842, 694, 386, 12, MUTED)
    pdf.showPage()

page(5, '三个案例，三种展示重点。', '下表含实际结果，供演示者预习。所有百分数均为 CT 获胜概率；现场最后再揭示赢家。')
rows = [['案例 / 回合', '购买结束\nXGBoost', '购买结束\nLightGBM', '首杀后\nXGBoost', '首杀后\nLightGBM', '实际赢家']]
for case, label in [('A', 'A · Ancient R4'), ('B', 'B · Mirage R11'), ('C', 'C · Nuke R17')]:
    records = [r for r in evidence['matrix'] if r['example_id'] == case]
    rows.append([label] + [f"{r['ct_probability']*100:.2f}%" for r in records] + [records[0]['winner']])
table = Table(rows, colWidths=[266, 182, 182, 182, 182, 202], rowHeights=[64, 66, 66, 66])
table.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'YaHei'), ('FONTSIZE', (0,0), (-1,-1), 17),
    ('LEADING', (0,0), (-1,-1), 24), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN',(1,0),(-1,-1),'CENTER'),
    ('TEXTCOLOR',(0,0),(-1,-1), INK), ('BACKGROUND',(0,0),(-1,0),HexColor('#e5eaf0')),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#ffffff'), HexColor('#edf1f5')]), ('LEFTPADDING',(0,0),(-1,-1),16)]))
table.wrapOn(pdf, 1196, 262); table.drawOn(pdf, 42, H-184-262)
for top, title, detail in [(481, 'A / 一致', '四项都预测 CT，实际 CT 获胜。演示基本操作与结果复核。'),
                            (556, 'B / 改变', '购买结束倾向 CT，首杀后倾向 T；实际 T 获胜。演示两个信息时点的差异。'),
                            (631, 'C / 错误', '四项 CT 概率都很高，实际却由 T 获胜。高概率并不保证结果。')]:
    text(title, 42, top, 200, 22, bold=True); text(detail, 242, top+2, 970, 20)
pdf.showPage()

page(6, '照着演示，也能重新运行。', '建议 4–6 分钟：概览 → A 的基本操作 → B 的三页联动 → C 的高概率错误 → 数据边界。')
text('在当前 Windows 电脑启动', 42, 180, size=23, bold=True)
commands = ["Set-Location 'C:\\Users\\admin\\Documents\\Codex\\2026-07-06\\th\\work\\csdemo_round_prediction'", "& 'C:\\Users\\admin\\11\\envs\\game\\python.exe' -m src.csdemo.roundcast_server --host 127.0.0.1 --port 8765"]
pdf.setFillColor(HexColor('#e5eaf0')); pdf.roundRect(42, H-235-86, 1196, 86, 10, fill=1, stroke=0)
pdf.setFont('Courier', 14); pdf.setFillColor(INK)
for i, command in enumerate(commands): pdf.drawString(58, H-262-i*29, command)
text('看到 ready 后打开网页。保留终端运行；关闭时在同一终端按 Ctrl+C。重启后重新点击运行。', 42, 343, size=19)
text('若已占用端口，先查看已有预览。若缺少模型/数据或校验失败，检查本地文件与环境；换机需另备份 Git 忽略的工件。', 42, 391, size=17, color=MUTED)
card(42, 468, 582, 230, '如何准确说明能力', '仅两个离散预测时点；首杀后沿用购买结束基础状态。尚未接入连续胜率、位置、烟雾、炸弹和具体失利归因。')
card(650, 468, 588, 230, '整体指标与回合对话', '正式指标来自 4,172 / 4,170 个测试回合。三例不能用来排名模型。下方对话按需发送，使用本机已有 Codex 登录。')
text('复验：12 项真实预测与参考误差为 0；三页结果一致；重启后可重新推理。完整步骤与证据见 demo_guide.md。', 42, 717, size=12, color=MUTED)
pdf.save()
print(OUT / 'roundcast-v1-demo-guide.pdf')

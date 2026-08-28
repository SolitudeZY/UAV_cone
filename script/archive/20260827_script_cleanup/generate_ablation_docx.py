#!/usr/bin/env python3
"""Create a patent-oriented Word report from generated ablation artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "script" / "runs" / "ablation_patent_summary_20260825"


def set_cell_text(cell, text: object, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(9)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def shade_cell(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_width(cell, width_cm: float) -> None:
    cell.width = Cm(width_cm)
    properties = cell._tc.get_or_add_tcPr()
    width = properties.find(qn("w:tcW"))
    if width is None:
        width = OxmlElement("w:tcW")
        properties.append(width)
    width.set(qn("w:w"), str(int(width_cm * 567)))
    width.set(qn("w:type"), "dxa")


def add_table(document: Document, headers: list[str], rows: list[list[object]], widths: list[float] | None = None):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True)
        shade_cell(table.rows[0].cells[index], "D9EAF7")
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            set_cell_text(cells[index], value)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                set_cell_width(row.cells[index], width)
    document.add_paragraph()
    return table


def add_bullet(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run(text)


def add_body(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(text)
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.line_spacing = 1.15
    paragraph.paragraph_format.space_after = Pt(6)


def row_for(rows: list[dict[str, str]], test: str, scenario: str, group: str) -> dict[str, str]:
    return next(r for r in rows if r["test"] == test and r["scenario"] == scenario and r["group"] == group)


def main() -> None:
    rows = list(csv.DictReader((OUT / "ablation_metrics.csv").open(encoding="utf-8-sig")))
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    styles = doc.styles
    styles["Normal"].font.name = "Microsoft YaHei"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles["Normal"].font.size = Pt(10.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("交通锥异常监测方法消融实验说明")
    run.bold = True
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(18)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("三类锁定视频回放证据（专利技术交底附件）").font.size = Pt(11)
    doc.add_paragraph()

    doc.add_heading("1. 实验目的", level=1)
    add_body(doc, "本实验用于验证交通锥异常监测后处理链路中四类机制的独立或组合技术效果：基于稳定目标身份的 FALLEN 时序投票、OBB 几何与三分类证据融合、过小/边缘/孤立候选的资格约束，以及已确认倒伏目标占据缺口预期位置时的 FALLEN-MISSING 互斥。实验采用固定模型、固定视频片段和固定分辨率，逐次改变预先定义的参数组合，保留逐帧事件、业务激活日志、配置和哈希证据。")
    add_body(doc, "本文只对已回放的锁定片段给出场景级证据，不将片段中的候选数、稳定事件数或业务激活数表述为完整测试集的准确率、召回率或误报率。")

    doc.add_heading("2. 共同实验条件与可复现性", level=1)
    add_bullet(doc, "OBB 权重 SHA-256：974d056f6cdaca63be3885b075425b7e2b9da0e54bbe79526e811e51c7bef76e")
    add_bullet(doc, "MobileNet 权重 SHA-256：a4e945b4ab7b55c3027ddcaaa3036d6f58c5e4b306694b454840b224e88edbcb")
    add_bullet(doc, "分类器阈值文件 SHA-256：b139728d33ba2179202d7d45df8253dc11bf3c001d51a17aeee36d4d494ead62")
    add_bullet(doc, "推理尺寸 1280；YOLO conf=0.25、IoU=0.5；分类器使用 full 模式和 1.6 倍候选上下文。")
    add_bullet(doc, "每次运行的 run_config.json 记录完整参数、模型哈希、阈值文件哈希和视频哈希。")

    doc.add_heading("3. 消融组定义", level=1)
    add_table(doc, ["组", "证据通道", "FALLEN 身份时序", "资格过滤/跨异常"], [
        ["A", "角度规则", "关闭，逐帧", "关闭"],
        ["B", "角度规则", "开启，7/5/3/5/30", "关闭"],
        ["C", "角度 + 三分类融合", "关闭，逐帧", "关闭"],
        ["D", "角度 + 三分类融合", "开启，7/5/3/5/30", "关闭"],
        ["E", "角度 + 三分类融合", "开启，7/5/3/5/30", "开启过小/边缘/人员/孤立及 FALLEN-MISSING 互斥"],
    ], [1.0, 4.0, 4.0, 7.0])

    doc.add_heading("4. 第一组：水面光斑与倒伏/正向倒伏片段", level=1)
    add_body(doc, "水面光斑片段为 269 帧；倒伏/正向倒伏片段为 75 帧。raw/stable FALLEN 表示逐帧候选事件帧数和经过身份时序后的事件帧数；业务激活表示 alert.jsonl 中的激活转换条数。")
    table_rows = []
    for scenario, label in (("water_glare", "水面光斑"), ("fallen_positive", "倒伏/正向倒伏")):
        for group in "ABCDE":
            r = row_for(rows, "test1", scenario, group)
            table_rows.append([label, group, f"{r['raw_fallen']}/{r['stable_fallen']}", f"{r['raw_missing']}/{r['stable_missing']}", r["fallen_activations"], r["missing_activations"], r["total_activations"], r["unique_seen"]])
    add_table(doc, ["场景", "组", "raw/stable FALLEN", "raw/stable MISSING", "FALLEN 激活", "MISSING 激活", "总激活", "unique"], table_rows, [3.0, 1.0, 2.2, 2.2, 1.6, 1.6, 1.4, 1.3])
    doc.add_picture(str(OUT / "charts/01_water_fallen.png"), width=Inches(6.35))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_picture(str(OUT / "charts/02_fallen_positive.png"), width=Inches(6.35))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_body(doc, "水面光斑片段中，A 组业务 FALLEN 激活为 103，B 组降为 1，表明稳定身份与 7/5 时序投票能够显著压制光斑候选导致的重复激活。C 组在加入三分类后，累计稳定轨迹由 44 降为 19；D 组同时启用分类与时序，水面 FALLEN 激活为 0。该结果支持“时序机制抑制重复触发、分类证据减少非锥桶候选进入业务链路”的技术效果。")
    add_body(doc, "倒伏/正向倒伏片段中，C 组产生 71 个分类 FALLEN 候选事件帧，D/E 经过时序后保留 69 个稳定 FALLEN 事件帧；但 B 组角度规则的稳定 FALLEN 为 0，说明该旧版 OBB 在该片段上的角度候选能力不足。因此本组不能表述为完整的倒伏召回率证明，只能作为分类证据与时序状态机在该回放中的行为证据。")

    doc.add_heading("5. 第二组：过小目标和孤立点片段", level=1)
    add_body(doc, "该片段为 26-08-03_small_cone.mp4，3942 帧，约 131.8 秒，视频中的锥桶目标整体较小并包含孤立点。D/E 共同使用分类器和完整 FALLEN 时序，仅改变资格过滤及跨异常参数。")
    small_rows = []
    for group in ("D", "E"):
        r = row_for(rows, "test2", "small_cone", group)
        small_rows.append([group, r["too_small"], r["not_cone_suppressed"], r["isolated"], r["missing_edge_ignored"], r["stable_fallen"], r["stable_missing"], r["total_activations"], r["unique_seen"]])
    add_table(doc, ["组", "too_small", "not_cone 抑制", "isolated", "边缘缺失抑制", "stable FALLEN", "stable MISSING", "总激活", "unique"], small_rows, [1.0, 1.5, 2.0, 1.4, 2.2, 1.8, 1.8, 1.4, 1.2])
    doc.add_picture(str(OUT / "charts/03_small_cone_de.png"), width=Inches(6.35))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_body(doc, "E 组新增排除 613 个过小候选，累计稳定轨迹由 105 降为 102，并产生 286 条边缘缺失抑制记录；业务总激活由 5 降为 4。该结果支持资格过滤减少信息不足候选进入时序和路线异常判断的技术效果。由于本片段同时改变了过小、边缘、人员和孤立端点参数，不能单独把孤立端点规则的贡献从组合效果中分离。")

    doc.add_heading("6. 第三组：FALLEN-MISSING 互斥片段", level=1)
    add_body(doc, "从 26-08-07-多角度.mp4 截取原视频 36–50 秒，得到 420 帧、约 14.014 秒的测试片段。该片段包含已确认倒伏目标与路线缺口候选同时出现的时段。D 关闭互斥；E 开启互斥。")
    fm_rows = []
    for group in ("D", "E"):
        r = row_for(rows, "test3", "fallen_missing", group)
        fm_rows.append([group, f"{r['raw_missing']}/{r['stable_missing']}", f"{r['raw_fallen']}/{r['stable_fallen']}", r["suppressed_missing_by_fallen"], r["missing_activations"], r["fallen_activations"], r["total_activations"]])
    add_table(doc, ["组", "raw/stable MISSING", "raw/stable FALLEN", "被 FALLEN 抑制", "MISSING 激活", "FALLEN 激活", "总激活"], fm_rows, [1.0, 2.3, 2.3, 2.0, 1.8, 1.8, 1.4])
    doc.add_picture(str(OUT / "charts/04_fallen_missing_de.png"), width=Inches(6.35))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_body(doc, "E 组产生 57 条 suppressed_missing_by_fallen 审计记录，首次出现在测试片段 1.935 秒，对应原视频约 37.935 秒，最后一处约对应原视频 40.238 秒。E 保留 5 次 FALLEN 激活，同时将 MISSING 业务激活由 2 次降为 1 次；D 组没有互斥记录并出现额外 MISSING 激活。该结果直接支持“倒伏目标占据缺口预期位置时优先保留 FALLEN、抑制同一物理事件的 MISSING 双报”的技术效果。")
    add_body(doc, "互斥不会删除原始缺口证据：原始 MISSING 仍保存在 events.jsonl，只有在倒伏目标已经通过时序确认并且空间位置覆盖缺口预期位置时，才阻止其进入业务 MISSING 状态。")

    doc.add_heading("7. 可用于专利撰写的技术效果表述", level=1)
    add_body(doc, "实验结果表明，所述方法通过共享的稳定目标身份将连续帧中的检测候选关联为同一物理目标，并基于滑动窗口投票、确认、清除和重激活冷却规则形成稳定异常状态；由此可降低 OBB 逐帧检测抖动造成的重复异常激活。")
    add_body(doc, "实验结果还表明，将 OBB 的姿态几何证据与候选三分类结果进行融合，可在水面光斑场景中减少非锥桶候选进入路线和异常判断；结合目标尺寸、图像边缘、人员区域和孤立端点等资格条件，可进一步限制信息不足候选对业务状态的影响。")
    add_body(doc, "对于路线缺口与倒伏同时出现的情况，方法通过比较已确认倒伏目标与缺口预期位置的空间关系，在保留倒伏异常的同时抑制同一物理事件的缺失业务告警，从而避免同一目标被重复或矛盾地上报。")

    doc.add_heading("8. 证据边界与未覆盖内容", level=1)
    add_bullet(doc, "本文件是固定视频片段的回放证据，不是完整数据集统计报告。")
    add_bullet(doc, "没有人工逐目标真值时，raw/stable 事件帧数、activation 和 unique 不能直接换算为 precision、recall 或真实锥桶数量。")
    add_bullet(doc, "第二组未对孤立端点做独立单变量消融；第三组是目前验证 FALLEN-MISSING 互斥的有效片段。")
    add_bullet(doc, "文档中的专利技术效果应由代理人结合权利要求范围、完整标注数据和后续对照实验进一步定稿。")

    doc.add_heading("9. 证据文件位置", level=1)
    add_bullet(doc, "第一组：script/runs/ablation_20260824_old_yolo_glare_fallen/")
    add_bullet(doc, "第二组：script/runs/ablation_20260825_small_cone/")
    add_bullet(doc, "第三组：script/runs/ablation_20260825_fallen_missing_36_50/")
    add_bullet(doc, "本汇总图表和数据：script/runs/ablation_patent_summary_20260825/")

    path = OUT / "交通锥异常监测消融实验专利说明.docx"
    doc.save(path)
    print(path)


if __name__ == "__main__":
    main()

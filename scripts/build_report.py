"""Convert reports/REPORT.md into reports/MLOps_Assignment1_Report.docx.

    python scripts/build_report.py

Supports the Markdown subset used in the report: headings, paragraphs, **bold**,
`code`, bullet/numbered lists, tables, fenced code blocks and images. Images that do
not exist yet (screenshots still to be taken) are replaced by a visible placeholder.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "REPORT.md"
OUT = ROOT / "reports" / "MLOps_Assignment1_Report.docx"

INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")
IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")


def add_inline(paragraph, text: str) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        elif part.startswith("["):
            label, url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part).groups()
            run = paragraph.add_run(
                label if url.startswith(("#", ".", "/")) else f"{label} ({url})"
            )
            run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        else:
            paragraph.add_run(part)


def add_image(doc: Document, rel_path: str, alt: str) -> None:
    path = (SRC.parent / rel_path).resolve()
    if path.exists():
        doc.add_picture(str(path), width=Inches(6.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p = doc.add_paragraph()
        run = p.add_run(f"[Screenshot pending: {rel_path}]")
        run.italic = True
        run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    if alt:
        cap = doc.add_paragraph(alt)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)


def add_table(doc: Document, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
    table = doc.add_table(rows=len(cells), cols=len(cells[0]))
    table.style = "Light Grid Accent 1"
    for i, row in enumerate(cells):
        for j, value in enumerate(row[: len(cells[0])]):
            cell = table.cell(i, j)
            cell.text = ""
            add_inline(cell.paragraphs[0], value)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9)
                run.bold = run.bold or i == 0
    doc.add_paragraph()


def build() -> Path:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    lines = SRC.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            if lang != "mermaid":  # mermaid is rendered as a PNG elsewhere in the report
                p = doc.add_paragraph()
                run = p.add_run("\n".join(block))
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
        elif stripped.startswith("<!--"):
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
        elif stripped == "\\pagebreak":
            doc.add_page_break()
        elif m := re.match(r"^(#{1,4})\s+(.*)", stripped):
            level = len(m.group(1))
            if level == 1 and not doc.paragraphs:
                doc.add_heading(m.group(2), level=0)
            else:
                doc.add_heading(m.group(2), level=min(level, 3) if level > 1 else 1)
        elif m := IMAGE.match(stripped):
            add_image(doc, m.group("path"), m.group("alt"))
        elif stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            add_table(doc, rows)
            continue
        elif re.match(r"^\s*[-*]\s+", line):
            indent = (len(line) - len(line.lstrip())) // 2
            p = doc.add_paragraph(style="List Bullet 2" if indent else "List Bullet")
            add_inline(p, re.sub(r"^\s*[-*]\s+", "", line))
        elif re.match(r"^\s*\d+\.\s+", line):
            p = doc.add_paragraph(style="List Number")
            add_inline(p, re.sub(r"^\s*\d+\.\s+", "", line))
        elif stripped.startswith(">"):
            p = doc.add_paragraph(style="Intense Quote")
            add_inline(p, stripped.lstrip("> "))
        elif stripped:
            para = [stripped]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not re.match(r"^(#|!\[|\||```|\s*[-*]\s|\s*\d+\.\s|>)", lines[i + 1].strip())
            ):
                i += 1
                para.append(lines[i].strip())
            add_inline(doc.add_paragraph(), " ".join(para))
        i += 1

    doc.save(OUT)
    print(f"Saved {OUT}")
    return OUT


if __name__ == "__main__":
    build()

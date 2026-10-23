"""Structured Office document parsing with targeted reading.

DOCX/PPTX/XLSX are parsed into a light index (headings, slides, sheets, ranges) so
the model can be asked about one specific part instead of the whole file. Nothing
here executes macros, follows external links or evaluates formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentIndex:
    """What a document contains, for targeted reads."""

    kind: str
    blocks: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def outline(self) -> str:
        rows = []
        for block in self.blocks:
            label = block.get("label") or ""
            title = block.get("title") or ""
            if label or title:
                rows.append(f"{label} {title}".strip())
        return "\n".join(rows[:60])

    def render(self, *, limit_chars: int = 12_000) -> str:
        parts = []
        used = 0
        for block in self.blocks:
            text = block.get("text") or ""
            if not text:
                continue
            header = f"[{block.get('label', '')}]".strip()
            chunk = f"{header}\n{text}".strip() if header != "[]" else text
            if used + len(chunk) > limit_chars:
                parts.append(f"…（已达到 {limit_chars} 字上限，后续内容可用指定范围的工具读取）")
                break
            parts.append(chunk)
            used += len(chunk)
        return "\n\n".join(parts)

    def select(self, target: str) -> list[dict[str, Any]]:
        """Return the blocks matching a page/slide/sheet/section selector."""

        key = (target or "").strip()
        if not key:
            return list(self.blocks)
        lowered = key.lower()
        hits = []
        for block in self.blocks:
            names = {
                str(block.get("label", "")).lower(),
                str(block.get("title", "")).lower(),
                str(block.get("name", "")).lower(),
            }
            if any(name and (lowered in name or name in lowered) for name in names):
                hits.append(block)
        if hits:
            return hits
        digits = "".join(char for char in key if char.isdigit())
        if digits:
            for block in self.blocks:
                if str(block.get("number", "")) == digits:
                    hits.append(block)
        return hits


def _docx(path: Path, limit_chars: int) -> DocumentIndex:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    index = DocumentIndex(kind="docx")
    heading = ""
    section = 0
    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = (paragraph.text or "").strip()
            if not text:
                continue
            style = (paragraph.style.name or "") if paragraph.style else ""
            if style.lower().startswith("heading") or style.lower() == "title":
                section += 1
                heading = text
                index.blocks.append(
                    {
                        "label": f"章节 {section}",
                        "number": section,
                        "title": text,
                        "text": text,
                        "kind": "heading",
                        "level": style,
                    }
                )
            else:
                index.blocks.append(
                    {
                        "label": f"章节 {section}" if section else "正文",
                        "number": section,
                        "title": heading,
                        "text": text,
                        "kind": "paragraph",
                    }
                )
        elif tag == "tbl":
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [(cell.text or "").strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                index.blocks.append(
                    {
                        "label": f"章节 {section} 表格" if section else "表格",
                        "number": section,
                        "title": heading,
                        "text": "\n".join(rows),
                        "kind": "table",
                        "rows": len(rows),
                    }
                )
    # Inline images are described by their position so the caller can render them.
    images = []
    for relation in getattr(document.part, "rels", {}).values():
        target = getattr(relation, "target_ref", "")
        if target and str(target).lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
        ):
            images.append(str(target))
    index.meta["images"] = images[:20]
    index.meta["sections"] = section
    return index


def _pptx(path: Path, limit_chars: int) -> DocumentIndex:
    from pptx import Presentation

    presentation = Presentation(str(path))
    index = DocumentIndex(kind="pptx")
    for number, slide in enumerate(presentation.slides, 1):
        texts: list[str] = []
        tables = 0
        charts = 0
        pictures = 0
        title = ""
        for shape in slide.shapes:
            name = type(shape).__name__
            if name == "Picture":
                pictures += 1
                continue
            if getattr(shape, "has_table", False):
                tables += 1
                for row in shape.table.rows:
                    cells = [(cell.text or "").strip() for cell in row.cells]
                    if any(cells):
                        texts.append(" | ".join(cells))
                continue
            if getattr(shape, "has_chart", False):
                charts += 1
                continue
            if getattr(shape, "has_text_frame", False):
                text = (shape.text_frame.text or "").strip()
                if text:
                    if not title:
                        title = text.splitlines()[0][:120]
                    texts.append(text)
        notes = ""
        try:
            notes_frame = slide.notes_slide.notes_text_frame
            notes = (notes_frame.text or "").strip() if notes_frame else ""
        except Exception:  # pragma: no cover - notes are optional
            notes = ""
        body = "\n".join(texts)
        if notes:
            body += f"\n[备注] {notes}"
        index.blocks.append(
            {
                "label": f"幻灯片 {number}",
                "number": number,
                "title": title,
                "text": body or f"（第 {number} 页只有图片/图表：图片 {pictures}、表格 {tables}、图表 {charts}）",
                "kind": "slide",
                "pictures": pictures,
                "tables": tables,
                "charts": charts,
            }
        )
    index.meta["slides"] = len(index.blocks)
    return index


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _xlsx(path: Path, limit_chars: int, *, max_rows: int = 400) -> DocumentIndex:
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    # data_only=True yields cached formula results without evaluating anything.
    workbook = load_workbook(str(path), read_only=False, data_only=True)
    formula_book = None
    try:
        formula_book = load_workbook(str(path), read_only=False, data_only=False)
    except Exception:  # pragma: no cover - formulas are optional
        formula_book = None
    index = DocumentIndex(kind="xlsx")
    try:
        for sheet in workbook.worksheets:
            rows: list[str] = []
            header: list[str] = []
            merged = [str(rng) for rng in sheet.merged_cells.ranges][:40]
            formula_cells = []
            for row_number, row in enumerate(
                sheet.iter_rows(min_row=1, max_row=min(sheet.max_row, max_rows)), 1
            ):
                cells = [_cell_text(cell.value) for cell in row]
                if not any(cells):
                    continue
                if not header:
                    header = [c for c in cells if c]
                coordinate = f"第 {row_number} 行"
                rows.append(f"{coordinate}: " + " | ".join(cells))
                if formula_book is not None:
                    try:
                        source_row = formula_book[sheet.title][row_number]
                        for cell in source_row:
                            if isinstance(cell.value, str) and cell.value.startswith("="):
                                formula_cells.append(
                                    f"{cell.coordinate}={cell.value}"
                                )
                    except Exception:  # pragma: no cover
                        pass
            text = "\n".join(rows)
            if formula_cells:
                text += "\n[公式]\n" + "\n".join(formula_cells[:40])
            if merged:
                text += "\n[合并单元格] " + ", ".join(merged)
            index.blocks.append(
                {
                    "label": f"工作表 {sheet.title}",
                    "name": sheet.title,
                    "title": sheet.title,
                    "text": text,
                    "kind": "sheet",
                    "dims": f"A1:{get_column_letter(max(1, sheet.max_column))}{max(1, sheet.max_row)}",
                    "rows": sheet.max_row,
                    "columns": sheet.max_column,
                    "header": header,
                }
            )
    finally:
        workbook.close()
        if formula_book is not None:
            formula_book.close()
    index.meta["sheets"] = [block["name"] for block in index.blocks]
    return index


def parse_document(path: Path, *, limit_chars: int = 12_000) -> DocumentIndex:
    """Parse a supported Office document into an index of readable blocks."""

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _docx(path, limit_chars)
    if suffix == ".pptx":
        return _pptx(path, limit_chars)
    if suffix in {".xlsx", ".xlsm"}:
        return _xlsx(path, limit_chars)
    raise ValueError(f"不支持的 Office 格式：{suffix or '未知'}")


def is_office(path: Path) -> bool:
    return path.suffix.lower() in {".docx", ".pptx", ".xlsx", ".xlsm"}


def read_target(index: DocumentIndex, target: str, *, limit_chars: int = 12_000) -> str:
    """Render only the requested part of the document."""

    blocks = index.select(target) if target else index.blocks
    if not blocks:
        available = ", ".join(
            str(block.get("label") or block.get("name")) for block in index.blocks[:20]
        )
        raise ValueError(f"没有找到“{target}”；可用范围：{available or '无'}")
    view = DocumentIndex(kind=index.kind, blocks=blocks, meta=index.meta)
    return view.render(limit_chars=limit_chars)

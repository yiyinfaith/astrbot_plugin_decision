"""Structured Office document parsing with targeted reading.

DOCX/PPTX/XLSX are parsed into a light index (headings, slides, sheets, ranges) so
the model can be asked about one specific part instead of the whole file. Nothing
here executes macros, follows external links or evaluates formulas.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack, closing
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
            # Keep the actual relationship bytes in memory.  A target_ref is
            # only a package-internal name and cannot be handed to a provider.
            blob = getattr(getattr(relation, "target_part", None), "blob", None)
            images.append({"name": Path(str(target)).name, "data": bytes(blob or b"")})
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
                image = getattr(shape, "image", None)
                blob = getattr(image, "blob", None)
                filename = getattr(image, "filename", f"slide-{number}-image-{pictures}.bin")
                if blob:
                    index.meta.setdefault("images", []).append(
                        {
                            "name": str(filename),
                            "data": bytes(blob),
                            "slide_number": number,
                        }
                    )
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
                chart = getattr(shape, "chart", None)
                # Chart XML is not a visual raster, but exposing its series and
                # categories prevents a chart-only slide from becoming empty
                # when no Office renderer is installed on the host.
                try:
                    for series in chart.series:
                        title = str(getattr(series, "name", "") or "")
                        values = ", ".join(str(value) for value in series.values)
                        texts.append(f"[图表] {title}: {values}".strip())
                except Exception:  # pragma: no cover - chart variants differ
                    pass
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
    index.meta["images"] = index.meta.get("images", [])[:40]
    index.meta["slides"] = len(index.blocks)
    return index


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


_XLSX_MAX_ROWS = 1_048_576
_XLSX_MAX_COLUMNS = 16_384
_XLSX_WINDOW_CELLS = 100_000


def _check_xlsx_budget(deadline: float, cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise RuntimeError("XLSX 读取已取消")
    if time.monotonic() >= deadline:
        raise TimeoutError("XLSX 读取超时，请缩小目标区域或提高理解超时")


def _xlsx_sheet_details(
    path: Path,
    sheet: Any,
    bounds: tuple[int, int, int, int],
    deadline: float,
    cancelled: Callable[[], bool] | None,
) -> tuple[list[int], list[str], int, int]:
    """Read metadata absent from openpyxl's read-only worksheet API.

    Only the internal package member name comes from openpyxl. XML elements
    are discarded as they are consumed; no complete sheet tree is retained.
    """

    from xml.etree.ElementTree import iterparse
    from zipfile import ZipFile

    from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries

    start_col, start_row, end_col, end_row = bounds
    hidden = []
    merged = []
    stack = []
    rows = max(1, sheet.max_row or 1)
    columns = max(1, sheet.max_column or 1)
    with ZipFile(path) as archive, archive.open(sheet._worksheet_path.lstrip("/")) as stream:
        for step, (event, element) in enumerate(iterparse(stream, events=("start", "end"))):
            if step % 512 == 0:
                _check_xlsx_budget(deadline, cancelled)
            if event == "start":
                stack.append(element)
                continue
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "row":
                _check_xlsx_budget(deadline, cancelled)
                number = int(element.get("r", "0"))
                rows = max(rows, number)
                if start_row <= number <= end_row and element.get("hidden") in {"1", "true"}:
                    hidden.append(number)
            elif tag == "c" and not sheet.max_column:
                # Write-only producers may omit <dimension>. Recover physical
                # dimensions during the metadata pass we already perform.
                coordinate = element.get("r")
                if coordinate:
                    row, column = coordinate_to_tuple(coordinate)
                    rows, columns = max(rows, row), max(columns, column)
            elif tag == "mergeCell":
                _check_xlsx_budget(deadline, cancelled)
                value = element.get("ref", "")
                min_col, min_row, max_col, max_row = range_boundaries(value)
                rows, columns = max(rows, max_row), max(columns, max_col)
                if min_row <= end_row and max_row >= start_row and min_col <= end_col and max_col >= start_col:
                    merged.append(value)
            stack.pop()
            element.clear()
            if stack:
                stack[-1].remove(element)
    _check_xlsx_budget(deadline, cancelled)
    return hidden, merged, rows, columns


def _xlsx_sheet_window(
    path: Path,
    sheet: Any,
    formula_sheet: Any,
    *,
    start_col: int,
    start_row: int,
    end_col: int,
    end_row: int,
    limit_chars: int,
    deadline: float,
    cancelled: Callable[[], bool] | None,
) -> dict[str, Any]:
    """Collect a bounded window using physical coordinates and streaming IO."""

    from openpyxl.utils import get_column_letter
    from openpyxl.utils.cell import range_boundaries

    _check_xlsx_budget(deadline, cancelled)
    start_col = start_col or 1
    # This limits the amount of data materialized for one response, not the
    # coordinate that can be addressed. A tiny target on row 1,048,576 works.
    cell_budget = max(1, min(_XLSX_WINDOW_CELLS, limit_chars // 8))
    hidden, merged, total_rows, columns = _xlsx_sheet_details(
        path, sheet,
        (start_col, start_row, end_col or _XLSX_MAX_COLUMNS, min(end_row, start_row + cell_budget - 1)),
        deadline, cancelled,
    )
    end_col = end_col or columns
    end_col = min(end_col, start_col + cell_budget - 1)
    end_row = min(end_row, start_row + max(1, cell_budget // (end_col - start_col + 1)) - 1)
    kwargs = dict(min_row=start_row, max_row=end_row, min_col=start_col, max_col=end_col)
    rows = []
    cells_data: dict[str, dict[str, dict[str, str]]] = {}
    formula_cells = []
    header = []
    used = 0
    last_row = start_row - 1
    cached_rows = sheet.iter_rows(**kwargs)
    formula_rows = formula_sheet.iter_rows(**kwargs)
    try:
        for number, (cached, formulas) in enumerate(zip(cached_rows, formula_rows, strict=True), start_row):
            _check_xlsx_budget(deadline, cancelled)
            values = []
            for column, (cell, source) in enumerate(zip(cached, formulas, strict=True), start_col):
                value = _cell_text(cell.value)
                formula = source.value if source.data_type == "f" else ""
                if value or formula:
                    coordinate = f"{get_column_letter(column)}{number}"
                    # A cell may be longer than the whole output budget.
                    # Preserve the fact that its content is incomplete.
                    if len(value) > limit_chars:
                        value = value[:limit_chars] + "…（单元格内容已截断）"
                    if formula and len(formula) > limit_chars:
                        formula = formula[:limit_chars] + "…（公式已截断）"
                    cells_data.setdefault(str(number), {})[str(column)] = {
                        "coordinate": coordinate, "value": value, "formula": formula,
                    }
                    if formula and len(formula_cells) < 40:
                        formula_cells.append(f"{coordinate}={formula}")
                values.append(value)
            if not header:
                header = list(values)
            rows.append(f"第 {number} 行: " + " | ".join(values))
            used += len(rows[-1]) + 1
            last_row = number
            if used >= limit_chars:
                break
        else:
            # An exhausted stream also proves that any trailing requested
            # coordinates are empty (read-only worksheets may omit them).
            last_row = end_row
    finally:
        cached_rows.close()
        formula_rows.close()
    hidden = [number for number in hidden if number <= last_row]
    merged = [
        value for value in merged
        if range_boundaries(value)[0] <= end_col and range_boundaries(value)[1] <= last_row
    ]
    hidden_set = set(hidden)
    for offset, number in enumerate(range(start_row, start_row + len(rows))):
        if number in hidden_set:
            rows[offset] = rows[offset].replace(f"第 {number} 行:", f"第 {number} 行 [隐藏]:", 1)
    text = "\n".join(rows)
    if formula_cells:
        text += "\n[公式]\n" + "\n".join(formula_cells)
    if merged:
        text += "\n[合并单元格] " + ", ".join(merged)
    truncated = last_row < total_rows or end_col < columns
    if truncated:
        text += f"\n[概览未覆盖全部单元格；已索引至第 {last_row} 行，可指定区域按需读取]"
    preview_budget = max(0, limit_chars - len(f"[工作表 {sheet.title}]\n"))
    if len(text) > preview_budget:
        note = "\n…（概览已截断，可指定单元格或区域按需读取）"
        text = text[:max(0, preview_budget - len(note))] + note[:preview_budget]
    return {
        "label": f"工作表 {sheet.title}", "name": sheet.title,
        "title": sheet.title, "kind": "sheet", "text": text,
        "dims": f"A1:{get_column_letter(columns)}{total_rows}",
        "rows": total_rows, "columns": columns, "header": header,
        "rows_data": rows, "cells_data": cells_data,
        "hidden_rows": hidden, "merged_ranges": merged, "truncated": truncated,
        "indexed_start_row": start_row, "indexed_end_row": last_row,
        "indexed_start_col": start_col, "indexed_end_col": end_col,
    }


def _xlsx(
    path: Path, limit_chars: int, *, max_rows: int | None = None,
    deadline: float | None = None, cancelled: Callable[[], bool] | None = None,
) -> DocumentIndex:
    from openpyxl import load_workbook

    deadline = deadline if deadline is not None else time.monotonic() + 90
    _check_xlsx_budget(deadline, cancelled)
    index = DocumentIndex(kind="xlsx")
    with ExitStack() as resources:
        workbook = resources.enter_context(closing(load_workbook(
            str(path), read_only=True, data_only=True, keep_links=False,
        )))
        formula_book = resources.enter_context(closing(load_workbook(
            str(path), read_only=True, data_only=False, keep_links=False,
        )))
        for sheet in workbook.worksheets:
            index.blocks.append(_xlsx_sheet_window(
                path, sheet, formula_book[sheet.title],
                start_col=1, start_row=1, end_col=0,
                end_row=min(sheet.max_row or _XLSX_MAX_ROWS, max_rows or _XLSX_MAX_ROWS),
                limit_chars=limit_chars, deadline=deadline, cancelled=cancelled,
            ))
    stat = path.stat()
    index.meta.update(source_path=str(path.resolve()), source_version=(stat.st_size, stat.st_mtime_ns))
    index.meta["sheets"] = [block["name"] for block in index.blocks]
    return index


def parse_document(
    path: Path, *, limit_chars: int = 12_000,
    deadline: float | None = None, cancelled: Callable[[], bool] | None = None,
) -> DocumentIndex:
    """Parse a supported Office document into an index of readable blocks."""

    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _docx(path, limit_chars)
    if suffix == ".pptx":
        return _pptx(path, limit_chars)
    if suffix in {".xlsx", ".xlsm"}:
        return _xlsx(path, limit_chars, deadline=deadline, cancelled=cancelled)
    raise ValueError(f"不支持的 Office 格式：{suffix or '未知'}")


def is_office(path: Path) -> bool:
    return path.suffix.lower() in {".docx", ".pptx", ".xlsx", ".xlsm"}


def _parse_xlsx_target(target: str) -> tuple[str, int, int, int, int] | None:
    """Parse an Excel target into sheet and physical row/column bounds.

    The returned coordinates are one-based and inclusive.  A target without
    explicit columns (``Sheet1!401:450``) means every indexed column in the
    sheet; the caller supplies the sheet's last column when rendering.
    """

    import re

    value = str(target or "").strip()
    if not value:
        return None
    sheet_name = ""
    address = value
    if "!" in value:
        sheet_name, address = value.split("!", 1)
        sheet_name = sheet_name.strip().strip("'")
    match = re.fullmatch(
        r"\s*\$?([A-Za-z]*)\$?(\d+)\s*"
        r"(?:[:\-~到]\s*\$?([A-Za-z]*)\$?(\d+))?\s*行?\s*",
        address,
    )
    if not match:
        # Keep the legacy row-only spelling, such as ``Sheet 401-450``.
        match = re.fullmatch(
            r"\s*(?:第\s*)?(\d+)\s*[-~到]\s*(\d+)\s*行?\s*", address
        )
        if not match:
            return None
        return _parse_xlsx_target(f"{sheet_name}!{match.group(1)}:{match.group(2)}")
    start_col_text, start_row_text, end_col_text, end_row_text = match.groups()
    start_row = int(start_row_text)
    end_row = int(end_row_text or start_row_text)
    if end_row < start_row:
        start_row, end_row = end_row, start_row
    from openpyxl.utils.cell import column_index_from_string

    try:
        start_col = (
            column_index_from_string(start_col_text) if start_col_text else 0
        )
        end_col = (
            column_index_from_string(end_col_text)
            if end_col_text
            else start_col
        )
    except ValueError:
        return None
    if start_col and end_col < start_col:
        start_col, end_col = end_col, start_col
    if not (1 <= start_row <= end_row <= _XLSX_MAX_ROWS):
        raise ValueError(f"Excel 行号必须在 1–{_XLSX_MAX_ROWS:,} 内")
    if max(start_col, end_col) > _XLSX_MAX_COLUMNS:
        raise ValueError("Excel 列号必须在 A–XFD 内")
    return sheet_name, start_col, start_row, end_col, end_row


def _xlsx_target_window(
    index: DocumentIndex, block: dict[str, Any],
    bounds: tuple[int, int, int, int], limit_chars: int,
    deadline: float, cancelled: Callable[[], bool] | None,
) -> dict[str, Any]:
    """Reuse an indexed window or stream the requested coordinates from source."""

    from openpyxl import load_workbook

    start_col, start_row, end_col, end_row = bounds
    path_text = index.meta.get("source_path")
    if path_text:
        path = Path(path_text)
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError("XLSX 源文件不可用，无法读取未索引区域，请重新提供文件") from exc
        if (stat.st_size, stat.st_mtime_ns) != tuple(index.meta.get("source_version", ())):
            raise ValueError("XLSX 源文件已变化，请重新解析后读取，不能复用旧索引")
    if (
        start_row >= block.get("indexed_start_row", 1)
        and end_row <= block.get("indexed_end_row", block.get("rows", 0))
        and (start_col or 1) >= block.get("indexed_start_col", 1)
        and (end_col or block.get("columns", 1)) <= block.get("indexed_end_col", block.get("columns", 1))
    ):
        return block
    if not path_text:
        raise ValueError("指定区域尚未索引，且没有可按需读取的 XLSX 源文件；请重新提供文件")
    _check_xlsx_budget(deadline, cancelled)
    with ExitStack() as resources:
        cached = resources.enter_context(closing(load_workbook(
            str(path), read_only=True, data_only=True, keep_links=False,
        )))
        formulas = resources.enter_context(closing(load_workbook(
            str(path), read_only=True, data_only=False, keep_links=False,
        )))
        name = block["name"]
        return _xlsx_sheet_window(
            path, cached[name], formulas[name],
            start_col=start_col, start_row=start_row, end_col=end_col, end_row=end_row,
            limit_chars=limit_chars, deadline=deadline, cancelled=cancelled,
        )


def _xlsx_merged_range(
    ranges: list[str], row_number: int, column_number: int
) -> str:
    """Return the merged range containing a physical cell, if any."""

    from openpyxl.utils.cell import range_boundaries

    for value in ranges:
        try:
            min_col, min_row, max_col, max_row = range_boundaries(value)
        except ValueError:
            continue
        if min_row <= row_number <= max_row and min_col <= column_number <= max_col:
            return value
    return ""


def _render_xlsx_region(
    block: dict[str, Any],
    *,
    start_col: int,
    start_row: int,
    end_col: int,
    end_row: int,
    limit_chars: int,
) -> str:
    """Render exactly the requested physical cells from one worksheet."""

    from openpyxl.utils import get_column_letter

    indexed_columns = max(1, int(block.get("columns", 1) or 1))
    # An explicit target is a coordinate query, not a request to clamp to the
    # sheet's last populated column.  Clamping ``Z5`` to ``D5`` for an A:D
    # sheet silently answers a different question.  Keep the requested bounds
    # intact and render an explicit out-of-index marker for cells beyond the
    # indexed worksheet dimensions.
    explicit_start_col = start_col > 0
    explicit_end_col = end_col > 0
    start_col = max(1, start_col or 1)
    end_col = max(start_col, end_col or indexed_columns)
    cells_data = block.get("cells_data") or {}
    hidden = {int(row) for row in block.get("hidden_rows", [])}
    merged_ranges = [str(value) for value in (block.get("merged_ranges") or [])]
    lines: list[str] = []
    used = 0
    notice = f"…（已达到 {limit_chars} 字上限，未显示的单元格请缩小范围读取）"
    budget = max(0, limit_chars - len(notice) - 1)
    for row_number in range(max(1, start_row), max(start_row, end_row) + 1):
        row_data = cells_data.get(str(row_number), {})
        marker = " [隐藏]" if row_number in hidden else ""
        line = f"第 {row_number} 行{marker}: "
        for column_number in range(start_col, end_col + 1):
            coordinate = f"{get_column_letter(column_number)}{row_number}"
            cell = row_data.get(str(column_number)) or {}
            formula = str(cell.get("formula") or "")
            value = str(cell.get("value") or "")
            outside_index = (
                column_number > indexed_columns
                and (explicit_start_col or explicit_end_col)
            )
            if row_number > block.get("indexed_end_row", block.get("rows", 0)) or column_number > block.get("indexed_end_col", indexed_columns):
                unread = f"…（{coordinate} 及后续区域尚未读取，达到本次处理预算，请缩小范围精读）"
                return ("\n".join(lines) + "\n" + unread)[:limit_chars].strip()
            if outside_index:
                rendered = (
                    f"{coordinate}=（超出工作表索引范围 "
                    f"A:{get_column_letter(indexed_columns)}）"
                )
            elif formula:
                rendered = f"{coordinate}={formula}"
                rendered += f" [缓存值: {value or '（空）'}]"
            elif value:
                rendered = f"{coordinate}={value}"
            else:
                merged = _xlsx_merged_range(merged_ranges, row_number, column_number)
                rendered = (
                    f"{coordinate}=（合并单元格 {merged}）"
                    if merged
                    else f"{coordinate}=（空）"
                )
            piece = (" | " if column_number > start_col else "") + rendered
            if used + len(line) + len(piece) + 1 > budget:
                remaining = max(0, budget - used - len(line))
                lines.append((line + piece[:remaining])[: max(0, budget - used)])
                lines.append(notice)
                return "\n".join(lines)
            line += piece
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _page_selection(target: str, page_count: int) -> list[int]:
    """Return zero-based page numbers selected by a slide/page target."""

    import re

    value = str(target or "").strip().lower()
    match = re.search(r"(?:slide|幻灯片|page|页|第)\s*(\d+)(?:\s*[-~到]\s*(\d+))?", value)
    if not match and value.isdigit():
        match = re.match(r"(\d+)", value)
    if not match:
        return list(range(page_count))
    start = max(1, int(match.group(1)))
    end_group = match.group(2) if (match.lastindex or 0) >= 2 else None
    end = max(start, int(end_group or start))
    return [number - 1 for number in range(start, min(end, page_count) + 1)]


def render_pages(
    path: Path,
    workdir: Path,
    *,
    target: str = "",
    max_pages: int = 12,
    timeout: float = 90.0,
) -> list[str]:
    """Render complete DOCX/PPTX pages through a sandboxed LibreOffice job.

    The conversion happens in a request-scoped profile and temporary directory;
    macros, user profiles, network proxies and external state are not reused.
    PyMuPDF then turns the resulting PDF pages into full-page PNGs, preserving
    shapes, arrows, charts, text and relative layout for a vision provider.
    """

    if path.suffix.lower() not in {".docx", ".pptx"}:
        return []
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise RuntimeError("缺少 LibreOffice/soffice，无法进行 Office 整页视觉渲染")
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 PyMuPDF，无法读取 LibreOffice 的整页渲染结果") from exc

    job = Path(tempfile.mkdtemp(prefix="office-render-", dir=str(workdir)))
    profile = job / "profile"
    output = job / "output"
    profile.mkdir()
    output.mkdir()
    env = os.environ.copy()
    env.update({"HOME": str(job), "USERPROFILE": str(job), "TMPDIR": str(job)})
    command = [
        executable,
        "--headless",
        "--nologo",
        "--nodefault",
        "--norestore",
        "--nolockcheck",
        f"-env:UserInstallation={profile.as_uri()}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output),
        str(path),
    ]

    def _limits() -> None:
        # Apply hard process limits where the host supports them.  Windows
        # safely falls back to the subprocess timeout.
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (max(5, int(timeout)), max(5, int(timeout) + 2)))
            memory = 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        except (ImportError, OSError, ValueError):
            return None

    try:
        subprocess.run(
            command,
            capture_output=True,
            check=True,
            timeout=timeout,
            env=env,
            cwd=str(job),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            preexec_fn=_limits if os.name == "posix" else None,
        )
        pdfs = list(output.glob("*.pdf"))
        if not pdfs:
            raise RuntimeError("LibreOffice 未生成 PDF 页面")
        rendered: list[str] = []
        with fitz.open(str(pdfs[0])) as document:
            wanted = _page_selection(target, document.page_count)
            for number in wanted[: max(1, int(max_pages))]:
                if number < 0 or number >= document.page_count:
                    continue
                page = document.load_page(number)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                target_path = workdir / f"office-page-{number + 1:02d}.png"
                pixmap.save(str(target_path))
                rendered.append(str(target_path))
        return rendered
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Office 整页渲染超时") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace")[-400:]
        raise RuntimeError(f"LibreOffice 整页渲染失败：{detail}") from exc
    finally:
        shutil.rmtree(job, ignore_errors=True)


def read_target(
    index: DocumentIndex, target: str, *, limit_chars: int = 12_000,
    deadline: float | None = None, cancelled: Callable[[], bool] | None = None,
) -> str:
    """Render only the requested part of the document."""

    blocks = index.select(target) if target else index.blocks
    # Spreadsheet precision reads: ``Sheet!A401:C450``, ``Sheet 401-450`` or
    # ``第 401-450 行`` reuse the index or stream an unindexed physical window.
    if target and index.kind == "xlsx":
        deadline = deadline if deadline is not None else time.monotonic() + 90
        parsed = _parse_xlsx_target(target)
        if parsed is not None:
            sheet_name, start_col, start, end_col, end = parsed
            candidates = [
                block
                for block in index.blocks
                if not sheet_name
                or str(block.get("name", "")).casefold() == sheet_name.casefold()
            ]
            blocks = []
            if candidates:
                for block in candidates:
                    # Exact physical-coordinate rendering: no compressed row
                    # or column list is used for a target request.
                    _check_xlsx_budget(deadline, cancelled)
                    copy_block = dict(_xlsx_target_window(
                        index, block, (start_col, start, end_col, end), limit_chars,
                        deadline, cancelled,
                    ))
                    copy_block["text"] = _render_xlsx_region(
                        copy_block,
                        start_col=start_col,
                        start_row=start,
                        end_col=end_col,
                        end_row=end,
                        limit_chars=max(0, limit_chars - len(f"[{copy_block['label']}]\n")),
                    )
                    blocks.append(copy_block)
    if not blocks:
        available = ", ".join(
            str(block.get("label") or block.get("name")) for block in index.blocks[:20]
        )
        raise ValueError(f"没有找到“{target}”；可用范围：{available or '无'}")
    view = DocumentIndex(kind=index.kind, blocks=blocks, meta=index.meta)
    return view.render(limit_chars=limit_chars)

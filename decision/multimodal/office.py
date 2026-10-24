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


_MAX_INDEX_ROWS = 100_000


def _xlsx(path: Path, limit_chars: int, *, max_rows: int | None = None) -> DocumentIndex:
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
            merged = [str(rng) for rng in sheet.merged_cells.ranges]
            formula_cells = []
            # Keep a sparse physical-coordinate index for exact target reads.
            # ``rows_data`` remains the human-readable overview, while this
            # map avoids reconstructing a target from compressed list indexes.
            cells_data: dict[str, dict[str, dict[str, str]]] = {}
            hidden_rows: list[int] = []
            row_limit = min(
                sheet.max_row,
                max_rows if max_rows is not None else _MAX_INDEX_ROWS,
            )
            for row_number, row in enumerate(
                sheet.iter_rows(min_row=1, max_row=row_limit), 1
            ):
                cells = [_cell_text(cell.value) for cell in row]
                if not header:
                    header = list(cells)
                coordinate = f"第 {row_number} 行"
                hidden_flag = bool(sheet.row_dimensions[row_number].hidden)
                hidden = " [隐藏]" if hidden_flag else ""
                if hidden_flag:
                    hidden_rows.append(row_number)
                # Keep one record per physical Excel row, including blank rows.
                # The coordinate is part of the record so every targeted read
                # remains correct even with sparse sheets.
                rows.append(f"{coordinate}{hidden}: " + " | ".join(cells))
                source_row = None
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
                for cell in row:
                    formula = ""
                    if source_row is not None and cell.column <= len(source_row):
                        source_value = source_row[cell.column - 1].value
                        if isinstance(source_value, str) and source_value.startswith("="):
                            formula = source_value
                    cached = _cell_text(cell.value)
                    if cached or formula:
                        cells_data.setdefault(str(row_number), {})[str(cell.column)] = {
                            "coordinate": cell.coordinate,
                            "value": cached,
                            "formula": formula,
                        }
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
                    "rows_data": rows,
                    "cells_data": cells_data,
                    "hidden_rows": hidden_rows,
                    "merged_ranges": merged,
                    "truncated": row_limit < sheet.max_row,
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
        return sheet_name, 0, int(match.group(1)), 0, int(match.group(2))
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
    return sheet_name, start_col, start_row, end_col, end_row


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

    max_columns = max(1, int(block.get("columns", 1) or 1))
    start_col = start_col or 1
    end_col = end_col or max_columns
    start_col = max(1, min(start_col, max_columns))
    end_col = max(start_col, min(end_col, max_columns))
    cells_data = block.get("cells_data") or {}
    hidden = {int(row) for row in block.get("hidden_rows", [])}
    merged_ranges = [str(value) for value in (block.get("merged_ranges") or [])]
    lines: list[str] = []
    used = 0
    for row_number in range(max(1, start_row), max(start_row, end_row) + 1):
        row_data = cells_data.get(str(row_number), {})
        values: list[str] = []
        for column_number in range(start_col, end_col + 1):
            coordinate = f"{get_column_letter(column_number)}{row_number}"
            cell = row_data.get(str(column_number)) or {}
            formula = str(cell.get("formula") or "")
            value = str(cell.get("value") or "")
            if formula:
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
            values.append(rendered)
        marker = " [隐藏]" if row_number in hidden else ""
        line = f"第 {row_number} 行{marker}: " + " | ".join(values)
        if used + len(line) > limit_chars:
            lines.append(f"…（已达到 {limit_chars} 字上限）")
            break
        lines.append(line)
        used += len(line)
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


def read_target(index: DocumentIndex, target: str, *, limit_chars: int = 12_000) -> str:
    """Render only the requested part of the document."""

    blocks = index.select(target) if target else index.blocks
    # Spreadsheet precision reads: ``Sheet!A401:C450``, ``Sheet 401-450`` or
    # ``第 401-450 行`` select a bounded row window without reparsing the file.
    if target and index.kind == "xlsx":
        parsed = _parse_xlsx_target(target)
        if parsed is not None:
            sheet_name, start_col, start, end_col, end = parsed
            candidates = [
                block
                for block in index.blocks
                if not sheet_name
                or str(block.get("name", "")).casefold() == sheet_name.casefold()
            ]
            if candidates:
                blocks = []
                for block in candidates:
                    # Exact physical-coordinate rendering: no compressed row
                    # or column list is used for a target request.
                    copy_block = dict(block)
                    copy_block["text"] = _render_xlsx_region(
                        block,
                        start_col=start_col,
                        start_row=start,
                        end_col=end_col,
                        end_row=end,
                        limit_chars=limit_chars,
                    )
                    blocks.append(copy_block)
    if not blocks:
        available = ", ".join(
            str(block.get("label") or block.get("name")) for block in index.blocks[:20]
        )
        raise ValueError(f"没有找到“{target}”；可用范围：{available or '无'}")
    view = DocumentIndex(kind=index.kind, blocks=blocks, meta=index.meta)
    return view.render(limit_chars=limit_chars)

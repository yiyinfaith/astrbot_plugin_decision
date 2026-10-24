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
            merged = [str(rng) for rng in sheet.merged_cells.ranges][:40]
            formula_cells = []
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
                hidden = " [隐藏]" if sheet.row_dimensions[row_number].hidden else ""
                # Keep one record per physical Excel row, including blank rows.
                # The coordinate is part of the record so every targeted read
                # remains correct even with sparse sheets.
                rows.append(f"{coordinate}{hidden}: " + " | ".join(cells))
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
                    "rows_data": rows,
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
        import re

        match = re.search(
            r"(?:!|\s|第\s*)([A-Za-z]*)(\d+)\s*(?:[:\-~到]\s*([A-Za-z]*)(\d+))?\s*行?$",
            target,
        )
        if match:
            from openpyxl.utils.cell import column_index_from_string

            start_col = column_index_from_string(match.group(1)) if match.group(1) else 1
            start = int(match.group(2))
            end_col = (
                column_index_from_string(match.group(3))
                if match.group(3)
                else start_col
            )
            end = int(match.group(4) or start)
            end = max(start, end)
            end_col = max(start_col, end_col)
            sheet_name = target.split("!", 1)[0].strip() if "!" in target else ""
            candidates = [
                block
                for block in index.blocks
                if not sheet_name or str(block.get("name", "")) == sheet_name
            ]
            if candidates:
                blocks = []
                for block in candidates:
                    # rows_data contains physical row coordinates, so an empty
                    # row never shifts a requested Excel range.
                    rows = []
                    for row in block.get("rows_data", []):
                        row_match = re.match(r"第\s*(\d+)\s*行(?:\s*\[隐藏\])?:\s*(.*)$", row)
                        if not row_match:
                            continue
                        row_number = int(row_match.group(1))
                        if start <= row_number <= end:
                            rows.append(row)
                    if start_col > 1 or end_col > 1:
                        import re as _re

                        sliced = []
                        for row in rows:
                            match_row = _re.match(
                                r"第\s*\d+\s*行(?:\s*\[隐藏\])?:\s*(.*)$", row
                            )
                            if not match_row:
                                sliced.append(row)
                                continue
                            cells = match_row.group(1).split(" | ")
                            sliced.append(
                                row.split(":", 1)[0]
                                + ": "
                                + " | ".join(cells[start_col - 1 : end_col])
                            )
                        rows = sliced
                    copy_block = dict(block)
                    copy_block["text"] = "\n".join(rows)
                    blocks.append(copy_block)
    if not blocks:
        available = ", ".join(
            str(block.get("label") or block.get("name")) for block in index.blocks[:20]
        )
        raise ValueError(f"没有找到“{target}”；可用范围：{available or '无'}")
    view = DocumentIndex(kind=index.kind, blocks=blocks, meta=index.meta)
    return view.render(limit_chars=limit_chars)

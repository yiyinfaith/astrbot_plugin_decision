"""Turn any supported media into something a chat model can actually read.

The provider layer only understands text, images and (sometimes) audio, so every
format is normalised here first. Animated images become a timestamped frame
sequence, video becomes sampled frames plus an optional audio track, audio is
segmented, and documents are reduced to structured text.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot import logger

# Files larger than this are refused before any decoding happens.
HARD_MAX_BYTES = 512 * 1024 * 1024

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".apng"}
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".log", ".ini", ".toml", ".py", ".js", ".ts", ".html", ".css", ".xml", ".sql",
}
AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus", ".amr", ".silk"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".m4v", ".ts"}

MAX_FRAME_PIXELS = 2_000_000
DEFAULT_EDGE = 1568


@dataclass(slots=True)
class Prepared:
    """Provider-ready material plus the notes that explain how it was produced."""

    images: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)
    text: str = ""
    notes: list[str] = field(default_factory=list)
    temp_paths: list[str] = field(default_factory=list)

    def cleanup(self) -> None:
        for path in self.temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        self.temp_paths.clear()

    def note_text(self) -> str:
        return "\n".join(self.notes)


def _guard_size(path: Path, limit_mb: int) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"无法读取媒体文件：{exc}") from exc
    if size > HARD_MAX_BYTES:
        raise ValueError("媒体体积超过安全上限")
    if limit_mb and size > limit_mb * 1024 * 1024:
        raise ValueError(f"媒体体积超过配置上限（{limit_mb} MB）")


async def run_command(cmd: list[str], *, timeout: float = 120.0) -> tuple[int, bytes, bytes]:
    # Use an async child process so cancellation of a media job also stops the
    # expensive ffmpeg/ffprobe process. A to_thread(subprocess.run) wrapper keeps
    # running after its awaiting task is cancelled and can accumulate orphaned
    # encoders during plugin reloads.
    if cmd and Path(cmd[0]).stem.lower() in {"ffmpeg", "ffprobe"}:
        # A downloaded "video" can actually be a playlist referencing an
        # internal HTTP service. Decoders must not open secondary network URLs
        # after the safe downloader has validated only the outer attachment.
        cmd = [cmd[0], "-protocol_whitelist", "file,pipe,crypto,data", *cmd[1:]]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return int(process.returncode or 0), stdout, stderr
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise


# --------------------------------------------------------------------- images
def _sample_indices(total: int, maximum: int) -> list[int]:
    if total <= maximum:
        return list(range(total))
    # Always keep the first and last frame so the motion is anchored.
    step = (total - 1) / (maximum - 1)
    picked = {round(index * step) for index in range(maximum)}
    return sorted(picked)


def _flatten_alpha(frame: Any) -> Any:
    from PIL import Image

    if frame.mode in {"RGBA", "LA", "P"}:
        rgba = frame.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    if frame.mode != "RGB":
        return frame.convert("RGB")
    return frame


def _resize_to_edge(frame: Any, edge: int) -> Any:
    from PIL import Image

    width, height = frame.size
    if max(width, height) <= edge:
        return frame
    scale = edge / float(max(width, height))
    return frame.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS
    )


def prepare_image(path: Path, settings: Any, workdir: Path) -> Prepared:
    """Decode a still image or an animation into provider-ready images."""

    from PIL import Image

    prepared = Prepared()
    _guard_size(path, int(settings.int("mm_max_file_mb")))
    max_frames = int(settings.int("mm_gif_max_frames"))
    max_pixels = int(settings.int("mm_gif_max_pixels"))
    edge = int(settings.int("mm_image_max_edge")) or DEFAULT_EDGE

    with Image.open(path) as image:
        animated = bool(getattr(image, "is_animated", False))
        frames_total = int(getattr(image, "n_frames", 1) or 1)
        width, height = image.size
        if width * height > max_pixels:
            raise ValueError(f"图片像素过大（{width}x{height}），已拒绝解码")
        duration_ms = 0
        try:
            duration_ms = int(image.info.get("duration", 0) or 0)
        except (TypeError, ValueError):
            duration_ms = 0

        if not animated or frames_total <= 1:
            # Still image: keep the original bytes when the provider accepts the
            # format, otherwise re-encode to PNG.
            if image.format in {"PNG", "JPEG", "WEBP"}:
                prepared.images.append(str(path))
                prepared.notes.append(
                    f"静态图片 {image.format} {width}x{height}"
                )
                return prepared
            target = workdir / f"{path.stem}-still.png"
            _flatten_alpha(image).save(target, "PNG")
            prepared.temp_paths.append(str(target))
            prepared.images.append(str(target))
            prepared.notes.append(f"图片已转换为 PNG {width}x{height}")
            return prepared

        # Animation (GIF / APNG / animated WebP): several frames, in order.
        indices = _sample_indices(frames_total, max(2, max_frames))
        total_ms = duration_ms * frames_total if duration_ms else 0
        for order, index in enumerate(indices):
            image.seek(index)
            frame = _flatten_alpha(image.copy())
            frame = _resize_to_edge(frame, edge)
            target = workdir / f"{path.stem}-f{order:02d}.png"
            frame.save(target, "PNG")
            prepared.temp_paths.append(str(target))
            prepared.images.append(str(target))
            if total_ms:
                seconds = (index * duration_ms) / 1000.0
                prepared.notes.append(f"第 {order + 1} 帧 ≈ {seconds:.1f}s")
            else:
                prepared.notes.append(f"第 {order + 1} 帧")
        total_seconds = total_ms / 1000.0
        header = (
            f"这是一段动图（{frames_total} 帧，共约 {total_seconds:.1f}s），"
            f"下面按时间顺序给出 {len(prepared.images)} 帧。"
            "请描述画面随时间的变化，而不是只描述某一帧。"
        )
        prepared.notes.insert(0, header)
        logger.debug(
            "Decision Engine 动图解码：%s 帧取 %s 帧", frames_total, len(prepared.images)
        )
    return prepared


def animated_montage(path: Path, settings: Any, workdir: Path, *, columns: int = 3) -> Prepared:
    """Tile evenly spaced animation frames into one image.

    A provider request with a single animated file either fails or shows only the
    first frame. Tiling gives the model several frames at once in the same shape
    AstrBot itself uses for animation previews, so nothing has to be understood
    twice and the request stays a single image.
    """

    from PIL import Image

    prepared = prepare_image(path, settings, workdir)
    if len(prepared.images) <= 1:
        return prepared
    frames = []
    for frame_path in prepared.images:
        with Image.open(frame_path) as frame:
            frames.append(frame.convert("RGB").copy())
    if not frames:
        return prepared
    tile_w = max(frame.width for frame in frames)
    tile_h = max(frame.height for frame in frames)
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), (255, 255, 255))
    for index, frame in enumerate(frames):
        left = (index % columns) * tile_w + (tile_w - frame.width) // 2
        top = (index // columns) * tile_h + (tile_h - frame.height) // 2
        sheet.paste(frame, (left, top))
    target = workdir / f"{path.stem}-montage.png"
    sheet.save(target, "PNG")
    prepared.temp_paths.append(str(target))
    prepared.images = [str(target)]
    prepared.notes.insert(
        0,
        f"这是一段动图的 {len(frames)} 帧拼图（从左到右、从上到下按时间顺序）。"
        "请描述画面随时间的连续变化，而不是只描述其中一格。",
    )
    return prepared


# ---------------------------------------------------------------------- video
def tooling_status() -> dict[str, bool]:
    """Which external binaries the deep media paths need."""

    import shutil

    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
    }


def require_ffmpeg() -> None:
    """Raise a clear, actionable error when ffmpeg is missing."""

    status = tooling_status()
    missing = [name for name, ok in status.items() if not ok]
    if missing:
        raise ValueError(
            "缺少 "
            + " / ".join(missing)
            + "，无法处理视频或转换音频；请先安装 FFmpeg 后重试"
        )


async def probe_video(path: Path) -> dict[str, Any]:
    require_ffmpeg()
    code, out, _ = await run_command(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        timeout=60,
    )
    if code != 0:
        raise ValueError("无法读取视频元数据（ffprobe 失败）")
    try:
        payload = json.loads(out.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise ValueError("视频元数据解析失败") from exc
    duration = 0.0
    try:
        duration = float(payload.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    fps = 0.0
    width = height = 0
    has_audio = False
    for stream in payload.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and not width:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            rate = str(stream.get("avg_frame_rate") or "0/1")
            try:
                num, _, den = rate.partition("/")
                fps = float(num) / float(den or 1)
            except (TypeError, ValueError, ZeroDivisionError):
                fps = 0.0
        elif kind == "audio":
            has_audio = True
    return {
        "duration": max(0.0, duration),
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "has_audio": has_audio,
    }


async def extract_frames(
    path: Path, workdir: Path, timestamps: list[float], *, edge: int = 1024
) -> list[str]:
    frames: list[str] = []
    for order, moment in enumerate(timestamps):
        target = workdir / f"v-{order:03d}.png"
        scale = (
            f"scale='min({edge},iw)':-2"
            if edge
            else "scale=iw:-2"
        )
        code, _, _ = await run_command(
            [
                "ffmpeg", "-v", "error", "-y",
                "-ss", f"{max(0.0, moment):.3f}",
                "-i", str(path),
                "-frames:v", "1",
                "-vf", scale,
                str(target),
            ],
            timeout=90,
        )
        if code == 0 and target.is_file():
            frames.append(str(target))
    return frames


async def extract_audio_track(path: Path, workdir: Path) -> str:
    target = workdir / "v-audio.mp3"
    code, _, _ = await run_command(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(path),
            "-vn", "-ac", "1", "-ar", "16000", str(target),
        ],
        timeout=180,
    )
    return str(target) if code == 0 and target.is_file() else ""


async def audio_is_silent(path: Path) -> bool | None:
    """Detect digital silence; undecodable/unmeasured audio remains unknown.

    -80 dB is below quiet speech and close to the 16-bit quantization floor.
    A silent codec stream must not invite the model to invent sounds.
    """
    try:
        code, _, error = await run_command([
            "ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-vn",
            "-af", "volumedetect", "-f", "null", "-",
        ], timeout=60)
    except (OSError, RuntimeError, TimeoutError):
        return None
    match = re.search(r"max_volume:\s*(-?inf|[-\d.]+)\s*dB", error.decode("utf-8", "replace"))
    if code or not match:
        return None
    return float(match[1]) <= -80.0


def plan_timestamps(duration: float, max_frames: int) -> list[float]:
    """Evenly spaced sample points that always include the very beginning."""

    if duration <= 0 or max_frames <= 0:
        return [0.0]
    if max_frames == 1:
        return [0.0]
    step = duration / float(max_frames)
    return [round(step * index, 3) for index in range(max_frames)]


def sample_video(
    info: dict[str, Any], settings: Any, *, window: tuple[float, float] | None = None
) -> list[float]:
    duration = float(info.get("duration") or 0)
    max_frames = int(settings.int("mm_video_max_frames"))
    if window:
        start, end = window
        span = max(0.0, min(end, duration) - max(0.0, start))
        count = max(2, min(max_frames, int(max_frames / 2) or 2))
        return [round(start + span * index / max(1, count - 1), 3) for index in range(count)]
    return plan_timestamps(duration, max_frames)


# ---------------------------------------------------------------------- audio
async def probe_audio(path: Path) -> dict[str, Any]:
    require_ffmpeg()
    code, out, _ = await run_command(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", str(path)],
        timeout=60,
    )
    duration = 0.0
    if code == 0:
        try:
            payload = json.loads(out.decode("utf-8", "replace") or "{}")
            duration = float(payload.get("format", {}).get("duration") or 0)
        except (ValueError, TypeError):
            duration = 0.0
    return {"duration": max(0.0, duration)}


async def prepare_audio(
    path: Path, settings: Any, workdir: Path
) -> Prepared:
    prepared = Prepared()
    _guard_size(path, int(settings.int("mm_max_file_mb")))
    info = await probe_audio(path)
    duration = float(info.get("duration") or 0)
    limit = float(settings.int("mm_audio_max_seconds"))
    if limit and duration > limit:
        raise ValueError(f"音频时长 {duration:.0f}s 超过上限 {limit:.0f}s")
    suffix = path.suffix.lower()
    segment = int(settings.int("mm_audio_segment_seconds")) or 0
    source = path
    if suffix not in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}:
        converted = workdir / "audio.mp3"
        code, _, _ = await run_command(
            [
                "ffmpeg", "-v", "error", "-y", "-i", str(path),
                "-vn", "-ac", "1", "-ar", "16000", str(converted),
            ],
            timeout=180,
        )
        if code != 0 or not converted.is_file():
            raise ValueError(f"无法转换音频格式 {suffix or '未知'}")
        prepared.temp_paths.append(str(converted))
        source = converted
        prepared.notes.append(f"音频已由 {suffix or '未知格式'} 转换为 mp3")
    if segment and duration > segment:
        chunks: list[str] = []
        start = 0.0
        index = 0
        while start < duration and index < 24:
            target = workdir / f"a-{index:02d}.mp3"
            code, _, _ = await run_command(
                [
                    "ffmpeg", "-v", "error", "-y", "-ss", f"{start:.1f}",
                    "-t", f"{segment}", "-i", str(source),
                    "-ac", "1", "-ar", "16000", str(target),
                ],
                timeout=180,
            )
            if code == 0 and target.is_file():
                chunks.append(str(target))
                prepared.temp_paths.append(str(target))
            start += segment
            index += 1
        if chunks:
            prepared.audio.extend(chunks)
            prepared.notes.append(
                f"音频约 {duration:.0f}s，按 {segment}s 分成 {len(chunks)} 段依次发送"
            )
            return prepared
    prepared.audio.append(str(source))
    prepared.notes.append(f"音频约 {duration:.0f}s")
    return prepared


# ---------------------------------------------------------------------- files
def _read_text_file(path: Path, limit: int) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding, errors="strict")[:limit]
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError as exc:
            raise ValueError(f"无法读取文件：{exc}") from exc
    return path.read_text(encoding="utf-8", errors="replace")[:limit]


def _pdf_text(path: Path, limit: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks = []
    for number, page in enumerate(reader.pages, 1):
        if sum(len(chunk) for chunk in chunks) > limit:
            break
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - malformed pdf
            text = f"(第 {number} 页解析失败：{exc})"
        if text.strip():
            chunks.append(f"[第 {number} 页]\n{text.strip()}")
    return "\n\n".join(chunks)[:limit]


def pdf_text_pages(path: Path, limit: int, pages: list[int] | None = None) -> str:
    """Extract only selected zero-based PDF pages, preserving page labels."""

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    wanted = pages if pages is not None else range(len(reader.pages))
    if isinstance(wanted, range):
        wanted = range(max(0, wanted.start), min(wanted.stop, len(reader.pages)), wanted.step)
    if pages is not None and not wanted:
        raise ValueError("指定的 PDF 页码超出文档范围")
    chunks: list[str] = []
    used = 0
    for number in wanted:
        if number < 0 or number >= len(reader.pages):
            continue
        try:
            text = reader.pages[number].extract_text() or ""
        except Exception as exc:  # pragma: no cover - malformed pdf
            text = f"(第 {number + 1} 页解析失败：{exc})"
        if not text.strip():
            continue
        chunk = f"[第 {number + 1} 页]\n{text.strip()}"
        chunks.append(chunk[: max(0, limit - used)])
        used += len(chunks[-1])
        if used >= limit:
            break
    return "\n\n".join(chunks)[:limit]


def _docx_text(path: Path, limit: int) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)[:limit]


def _pptx_text(path: Path, limit: int) -> str:
    from pptx import Presentation

    presentation = Presentation(str(path))
    parts = []
    for number, slide in enumerate(presentation.slides, 1):
        texts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                texts.append(shape.text_frame.text.strip())
        if texts:
            parts.append(f"[第 {number} 张幻灯片]\n" + "\n".join(texts))
    return "\n\n".join(parts)[:limit]


def _xlsx_text(path: Path, limit: int) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    try:
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if value is None else str(value) for value in row]
                if any(cell.strip() for cell in cells):
                    rows.append(" | ".join(cells))
                if len(rows) >= 200:
                    break
            if rows:
                parts.append(f"[工作表 {sheet.title}]\n" + "\n".join(rows))
    finally:
        workbook.close()
    return "\n\n".join(parts)[:limit]


def pdf_page_images(
    path: Path, settings: Any, workdir: Path, pages: list[int] | None = None
) -> list[str]:
    """Render PDF pages to PNG for vision-based OCR of scanned documents."""

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValueError("缺少 PyMuPDF，无法对扫描版 PDF 做 OCR") from exc
    limit = int(settings.int("mm_file_ocr_max_pages"))
    edge = int(settings.int("mm_image_max_edge")) or DEFAULT_EDGE
    rendered: list[str] = []
    with fitz.open(str(path)) as document:
        wanted = pages if pages is not None else range(min(limit, document.page_count))
        if isinstance(wanted, range):
            wanted = range(max(0, wanted.start), min(wanted.stop, document.page_count), wanted.step)
        for number in wanted[:limit]:
            if number < 0 or number >= document.page_count:
                continue
            page = document.load_page(number)
            zoom = min(2.0, edge / max(1.0, float(page.rect.width), float(page.rect.height)))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            target = workdir / f"page-{number + 1:02d}.png"
            pixmap.save(str(target))
            rendered.append(str(target))
    return rendered


def extract_file_text(path: Path, settings: Any) -> Prepared:
    """Reduce a document to structured text (never executes anything)."""

    prepared = Prepared()
    _guard_size(path, int(settings.int("mm_max_file_mb")))
    limit = int(settings.int("mm_file_max_chars"))
    suffix = path.suffix.lower()
    name = path.name
    if suffix == ".pdf":
        text = _pdf_text(path, limit)
    elif suffix == ".docx":
        text = _docx_text(path, limit)
    elif suffix == ".pptx":
        text = _pptx_text(path, limit)
    elif suffix in {".xlsx", ".xlsm"}:
        text = _xlsx_text(path, limit)
    elif suffix in TEXT_SUFFIXES:
        text = _read_text_file(path, limit)
    elif suffix in {".doc", ".xls", ".ppt"}:
        raise ValueError("旧版 Office 二进制格式不受支持，请另存为 docx/xlsx/pptx")
    else:
        raise ValueError(f"不支持的文件类型：{suffix or '未知'}")
    text = text.strip()
    if not text:
        raise ValueError("文件中没有可提取的文本（可能是扫描件，请改用图片理解）")
    prepared.text = text
    prepared.notes.append(f"文件 {name}，提取文本 {len(text)} 字")
    return prepared


def new_workdir(prefix: str = "mm-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_dir(directory: Path) -> None:
    for child in sorted(directory.rglob("*"), reverse=True):
        try:
            child.unlink() if child.is_file() else child.rmdir()
        except OSError:
            pass
    try:
        directory.rmdir()
    except OSError:
        pass

"""Optional local OCR backends.

Neither Tesseract nor PaddleOCR is a hard dependency: each adapter reports whether
it is usable, and the caller falls back to the vision model when none is. Nothing
is downloaded or installed automatically.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot import logger

ENGINE_LABELS = {"tesseract": "Tesseract", "paddleocr": "PaddleOCR"}


@dataclass(slots=True)
class OcrResult:
    text: str
    engine: str
    error: str = ""

    @property
    def ok(self) -> bool:
        # ``error`` may carry a non-fatal fallback note (for example, the
        # requested PaddleOCR backend was unavailable and Tesseract was used).
        # Text is usable whenever the selected engine returned it.
        return bool(self.text.strip())


def _tesseract_ready() -> tuple[bool, str]:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False, "未安装 pytesseract（pip install pytesseract）"
    if shutil.which("tesseract") is None:
        return False, "未安装 tesseract 可执行程序（apt install tesseract-ocr tesseract-ocr-chi-sim）"
    return True, ""


def _paddle_ready() -> tuple[bool, str]:
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False, "未安装 paddleocr（pip install paddleocr paddlepaddle）"
    return True, ""


def engine_status() -> dict[str, Any]:
    """Availability of each local backend, with an install hint."""

    tesseract_ok, tesseract_reason = _tesseract_ready()
    paddle_ok, paddle_reason = _paddle_ready()
    return {
        "tesseract": {"available": tesseract_ok, "reason": tesseract_reason},
        "paddleocr": {"available": paddle_ok, "reason": paddle_reason},
        "any": tesseract_ok or paddle_ok,
    }


def resolve_engine(requested: str) -> tuple[str, str]:
    """Pick a usable backend: the requested one when possible, else any."""

    status = engine_status()
    requested = (requested or "auto").strip().lower()
    if requested in ENGINE_LABELS and status[requested]["available"]:
        return requested, ""
    if requested in ENGINE_LABELS and not status[requested]["available"]:
        reason = status[requested]["reason"]
        fallback = next(
            (name for name in ENGINE_LABELS if status[name]["available"]), ""
        )
        if fallback:
            return fallback, f"{ENGINE_LABELS[requested]}不可用（{reason}），已改用{ENGINE_LABELS[fallback]}"
        return "", f"{ENGINE_LABELS[requested]}不可用：{reason}"
    for name in ("tesseract", "paddleocr"):
        if status[name]["available"]:
            return name, ""
    return "", "本地未安装 OCR 引擎：" + "；".join(
        status[name]["reason"] for name in ENGINE_LABELS if status[name]["reason"]
    )


def _ocr_tesseract(path: Path, lang: str) -> str:
    import pytesseract
    from PIL import Image

    with Image.open(path) as image:
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        return pytesseract.image_to_string(image, lang=lang or "chi_sim+eng")


def _ocr_paddle(path: Path) -> str:
    from paddleocr import PaddleOCR

    engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    result = engine.ocr(str(path), cls=True)
    lines: list[str] = []
    for page in result or []:
        for entry in page or []:
            try:
                lines.append(str(entry[1][0]))
            except (IndexError, TypeError):
                continue
    return "\n".join(lines)


async def ocr_image(
    path: Path, *, engine: str = "auto", lang: str = "chi_sim+eng", timeout: float = 120.0
) -> OcrResult:
    """OCR one image with a local backend, or report why it could not."""

    chosen, note = resolve_engine(engine)
    if not chosen:
        return OcrResult(text="", engine="", error=note)
    try:
        if chosen == "tesseract":
            text = await asyncio.wait_for(
                asyncio.to_thread(_ocr_tesseract, path, lang), timeout=timeout
            )
        else:
            text = await asyncio.wait_for(
                asyncio.to_thread(_ocr_paddle, path), timeout=timeout
            )
    except TimeoutError:
        return OcrResult(text="", engine=chosen, error=f"{ENGINE_LABELS[chosen]}识别超时")
    except Exception as exc:
        logger.info("Decision Engine 本地 OCR 失败：%s", str(exc)[:160])
        return OcrResult(text="", engine=chosen, error=str(exc)[:200])
    cleaned = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
    if not cleaned:
        return OcrResult(text="", engine=chosen, error="本地 OCR 没有识别出文字")
    return OcrResult(text=cleaned, engine=chosen, error=note)

"""Opt-in Gemini gateway streaming over its OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .inline_video import inline_credentials, video_mime
from .submissions import mark_submitted

MAX_MEDIA_BYTES = 40 * 1024 * 1024


def credentials(provider: Any, model: str) -> tuple[str, str]:
    key, base = inline_credentials(provider, model)
    host = urlsplit(base).hostname or ""
    if host == "googleapis.com" or host.endswith(".googleapis.com"):
        return "", ""
    return key, base


async def data_url(reference: str, *, max_bytes: int = MAX_MEDIA_BYTES, mime_type: str = "") -> str:
    if reference.startswith(("https://", "http://", "data:")):
        return reference

    def read() -> str:
        path = Path(reference)
        with path.open("rb") as source:
            raw = source.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("理解媒体超过 OpenAI 兼容传输大小预算")
        mime = mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

    return await asyncio.to_thread(read)


async def generate(
    provider: Any,
    *,
    model: str,
    prompt: str,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    video: str = "",
    timeout: float = 180,
) -> str:
    from openai import AsyncOpenAI

    key, base = credentials(provider, model)
    if not key:
        raise ValueError("所选 Provider 未配置可用的 OpenAI 兼容网关")
    content: list[dict] = [{"type": "text", "text": prompt}]
    for image in images or []:
        content.append({"type": "image_url", "image_url": {"url": await data_url(image)}})
    for source in audio or []:
        url = await data_url(source)
        if not url.startswith("data:") or ";base64," not in url:
            raise ValueError("音频兼容传输需要本地音频或 data URL")
        mime, data = url.split(";base64,", 1)
        fmt = (
            "mp3"
            if mime.endswith(("mpeg", "mp3"))
            else "wav"
            if mime.endswith(("wav", "wave", "x-wav"))
            else ""
        )
        if not fmt:
            raise ValueError("OpenAI 兼容音频仅接受预处理后的 WAV/MP3")
        content.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
    if video:
        # New API's actual Chat schema accepts a string here. A {url: ...}
        # object is silently omitted by rc.46's ParseContent implementation.
        content.append(
            {
                "type": "video_url",
                "video_url": await data_url(
                    video, max_bytes=15 * 1024 * 1024 - 1024, mime_type=video_mime(Path(video))
                ),
            }
        )
    answer: list[str] = []
    completed = False
    # Disable SDK retries as well as the runtime's whole-pipeline replay.
    async with AsyncOpenAI(
        api_key=key, base_url=base + "/v1", max_retries=0, timeout=timeout
    ) as client:
        mark_submitted()
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            stream=True,
        )
        try:
            async for chunk in stream:
                if getattr(chunk, "error", None):
                    raise RuntimeError("OpenAI 兼容流返回错误，未自动重新提交")
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                # reasoning_content/tool calls never become a media answer.
                if isinstance(choice.delta.content, str):
                    answer.append(choice.delta.content)
                if choice.finish_reason:
                    if choice.finish_reason != "stop":
                        raise RuntimeError("OpenAI 兼容理解未正常完成，未自动重新提交")
                    completed = True
        finally:
            await stream.close()
    result = "".join(answer).strip()
    if not completed or not result:
        raise RuntimeError("OpenAI 兼容流缺少完整最终结果，未自动重新提交")
    return result

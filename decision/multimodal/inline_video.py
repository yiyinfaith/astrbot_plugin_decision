"""Bounded whole-clip Gemini input, without Files or Interactions."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from .native_video import DEFAULT_BASE, _generate_content_text, provider_api_key
from .submissions import mark_submitted

# Conservative total JSON request budget; base64 overhead and prompt count too.
MAX_REQUEST_BYTES = 20 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def video_mime(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or ""
    if mime.startswith("video/"):
        return mime
    # Platform downloads often have a signed URL with no file extension.
    with path.open("rb") as source:
        header = source.read(512)
    if header[4:8] == b"ftyp":
        return "video/quicktime" if header[8:12] == b"qt  " else "video/mp4"
    if header.startswith(b"\x1aE\xdf\xa3") and b"webm" in header:
        return "video/webm"
    if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
        return "video/x-msvideo"
    return "video/mp4"


def inline_credentials(provider: Any, model: str = "") -> tuple[str, str]:
    config = getattr(provider, "provider_config", None) or {}
    kind = str(config.get("type") or "").lower()
    native = "gemini" in kind or "google" in kind
    if not native and not (
        kind == "openai_chat_completion"
        and model.removeprefix("models/").lower().startswith("gemini-")
    ):
        return "", ""
    # Never send an arbitrary OpenAI key to Google's default host.
    base = str(config.get("api_base") or (DEFAULT_BASE if native else "")).rstrip("/")
    for suffix in ("/v1beta", "/v1alpha", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    try:
        parsed = urlsplit(base)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.hostname.endswith("aiplatform.googleapis.com")
        ):
            return "", ""
        _ = parsed.port
    except ValueError:
        return "", ""
    return provider_api_key(provider), base


def request_size(path: Path, prompt: str) -> int:
    return (
        4 * ((path.stat().st_size + 2) // 3)
        + len(json.dumps(prompt, ensure_ascii=False).encode())
        + 512
    )


async def _read_json(response: Any) -> dict:
    data = bytearray()
    async for chunk in response.content.iter_any():
        if len(data) + len(chunk) > MAX_RESPONSE_BYTES:
            raise ValueError("Gemini 视频理解响应超过大小预算")
        data.extend(chunk)
    return json.loads(data)


async def _read_sse(response: Any) -> str:
    # Parse SSE event boundaries, including multi-line data and comments.
    pending = bytearray()
    event: list[str] = []
    answers: list[str] = []
    total = 0
    finished = False

    def dispatch() -> None:
        nonlocal finished
        if not event:
            return
        raw = "\n".join(event)
        event.clear()
        if raw == "[DONE]":
            return
        payload = json.loads(raw)
        if payload.get("error"):
            raise RuntimeError("Gemini 视频流返回错误；请求可能已处理，未自动重试")
        candidates = payload.get("candidates") or []
        if not candidates:
            return
        answer = _generate_content_text(payload)
        if answer:
            answers.append(answer)
        reason = candidates[0].get("finishReason") or candidates[0].get("finish_reason")
        if reason:
            if reason != "STOP":
                raise RuntimeError("Gemini 视频理解未正常完成，未自动重新提交")
            finished = True

    async for chunk in response.content.iter_any():
        pending.extend(chunk)
        while b"\n" in pending:
            line, _, remaining = pending.partition(b"\n")
            pending = bytearray(remaining)
            line = line.rstrip(b"\r")
            if not line:
                dispatch()
            elif line.startswith(b"data:"):
                total += len(line)
                if total > MAX_RESPONSE_BYTES:
                    raise ValueError("Gemini 视频流超过大小预算")
                event.append(line[5:].removeprefix(b" ").decode("utf-8"))
            # Comments/heartbeats never become answers or consume the budget.
        if len(pending) > MAX_RESPONSE_BYTES:
            raise ValueError("Gemini 视频流单行超过大小预算")
    if pending.startswith(b"data:"):
        event.append(bytes(pending[5:]).strip().decode("utf-8"))
    dispatch()
    text = "".join(answers).strip()
    if not finished or not text:
        raise RuntimeError("Gemini 视频流缺少完整最终结果，未自动重新提交")
    return text


async def generate(
    path: Path,
    prompt: str,
    *,
    provider: Any,
    model: str,
    timeout: float,
    stream: bool = True,
    gateway_openai: bool = False,
) -> str:
    """One paid submission; actual response bytes supply stream keepalive."""
    import aiohttp

    key, base = inline_credentials(provider, model)
    if not key:
        raise ValueError("所选模型没有可用的 Gemini 完整短视频接口")
    if request_size(path, prompt) > MAX_REQUEST_BYTES:
        raise ValueError("完整视频请求超过 inline 大小预算，请使用抽帧处理")
    if stream and gateway_openai:
        from . import openai_transport

        if openai_transport.credentials(provider, model)[0]:
            return await asyncio.wait_for(
                openai_transport.generate(
                    provider,
                    model=model,
                    prompt=prompt,
                    video=str(path),
                    timeout=timeout,
                ),
                timeout=timeout,
            )

    # Read at most the bounded amount even if the platform file changes size.
    def read() -> bytes:
        with path.open("rb") as source:
            data = source.read(MAX_REQUEST_BYTES + 1)
        if len(data) > MAX_REQUEST_BYTES:
            raise ValueError("视频文件在读取时超过大小预算")
        return data

    raw = await asyncio.to_thread(read)
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": video_mime(path),
                            "data": base64.b64encode(raw).decode("ascii"),
                        }
                    },
                    {"text": prompt},
                ],
            }
        ]
    }
    encoded = json.dumps(body, ensure_ascii=False).encode()
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("完整视频 JSON 超过 inline 大小预算")
    method = "streamGenerateContent?alt=sse" if stream else "generateContent"
    url = f"{base}/v1beta/models/{quote(model.removeprefix('models/'), safe='')}:{method}"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        mark_submitted()
        async with session.post(
            url, data=encoded, headers=headers, allow_redirects=False
        ) as response:
            if response.status != 200:
                # No response body/URL in the error: they may contain secrets.
                raise RuntimeError(f"Gemini 完整视频理解失败（HTTP {response.status}），未自动重试")
            if stream and "text/event-stream" in response.headers.get("Content-Type", ""):
                return await _read_sse(response)
            payload = await _read_json(response)
            text = _generate_content_text(payload)
            reason = ((payload.get("candidates") or [{}])[0]).get("finishReason")
            if not text or reason not in (None, "STOP") or payload.get("error"):
                raise RuntimeError("Gemini 完整视频理解未返回有效最终结果")
            return text

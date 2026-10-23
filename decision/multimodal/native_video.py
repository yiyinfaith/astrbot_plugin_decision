"""Native video understanding through the Gemini Files API.

Only real, verified endpoints are used here: upload, poll until ACTIVE, call
generateContent with a ``fileData`` part, then delete the uploaded file. Any other
provider keeps the frame-extraction path, so this module never affects them.
"""

from __future__ import annotations

import asyncio
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot import logger

DEFAULT_BASE = "https://generativelanguage.googleapis.com"
KIND_LABELS = {"native": "原生视频", "frames": "抽帧精读"}


@dataclass(slots=True)
class NativeResult:
    text: str
    route: str
    file_name: str = ""
    error: str = ""
    agentic: bool = False


def gemini_credentials(provider: Any) -> tuple[str, str]:
    """Reuse the provider's own key and base URL; never ask the user twice."""

    config = getattr(provider, "provider_config", None) or {}
    provider_type = str(config.get("type") or "").lower()
    getter = getattr(provider, "get_current_key", None)
    key = ""
    if callable(getter):
        try:
            key = str(getter() or "")
        except Exception:  # pragma: no cover - defensive
            key = ""
    if not key:
        keys_getter = getattr(provider, "get_keys", None)
        if callable(keys_getter):
            try:
                keys = keys_getter() or []
                key = str(keys[0]) if keys else ""
            except Exception:  # pragma: no cover
                key = ""
    if not key:
        key = str(config.get("key") or "")
    base = str(config.get("api_base") or DEFAULT_BASE).rstrip("/")
    if "gemini" not in provider_type and "google" not in provider_type:
        return "", ""
    if not key:
        return "", ""
    return key, base or DEFAULT_BASE


def native_available(provider: Any) -> tuple[bool, str]:
    key, base = gemini_credentials(provider)
    if not key:
        return False, "该 Provider 不是 Gemini 或缺少可复用的 API Key，无法使用原生视频上传"
    return True, ""


async def _request(
    session: Any,
    method: str,
    url: str,
    *,
    timeout: float,
    **kwargs: Any,
) -> Any:
    return await asyncio.wait_for(session.request(method, url, **kwargs), timeout=timeout)


async def upload_file(
    path: Path,
    *,
    api_key: str,
    base: str = DEFAULT_BASE,
    timeout: float = 300.0,
) -> tuple[str, str]:
    """Upload a file with the resumable protocol and return ``(name, uri)``."""

    import aiohttp

    size = path.stat().st_size
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    if not mime.startswith("video/"):
        mime = "video/mp4"
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(size),
        "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json",
    }
    start_body = {"file": {"display_name": path.name}}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base}/upload/v1beta/files?key={api_key}",
            headers=headers,
            json=start_body,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"原生视频上传初始化失败：HTTP {response.status}")
            upload_url = response.headers.get("X-Goog-Upload-URL") or response.headers.get(
                "x-goog-upload-url"
            )
            if not upload_url:
                raise RuntimeError("原生视频上传未返回上传地址")
        data = await asyncio.to_thread(path.read_bytes)
        upload_headers = {
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        async with session.post(
            upload_url,
            headers=upload_headers,
            data=data,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status >= 400:
                raise RuntimeError(f"原生视频上传失败：HTTP {response.status}")
            payload = await response.json(content_type=None)
    info = (payload or {}).get("file", {}) or {}
    name = str(info.get("name") or "")
    uri = str(info.get("uri") or "")
    if not name or not uri:
        raise RuntimeError("原生视频上传返回内容不完整")
    return name, uri


async def wait_active(
    name: str, *, api_key: str, base: str = DEFAULT_BASE, timeout: float = 300.0
) -> None:
    """Poll the file until Gemini reports ACTIVE."""

    import aiohttp

    deadline = asyncio.get_running_loop().time() + timeout
    async with aiohttp.ClientSession() as session:
        while asyncio.get_running_loop().time() < deadline:
            async with session.get(
                f"{base}/v1beta/{name}?key={api_key}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(f"查询原生视频状态失败：HTTP {response.status}")
                payload = await response.json(content_type=None)
            state = str((payload or {}).get("state") or "")
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise RuntimeError("Gemini 处理视频失败（state=FAILED）")
            await asyncio.sleep(2.0)
    raise RuntimeError("原生视频处理超时（等待 ACTIVE 超时）")


async def generate(
    *,
    api_key: str,
    base: str,
    model: str,
    file_uri: str,
    prompt: str,
    mime_type: str = "video/mp4",
    timeout: float = 300.0,
) -> str:
    """Ask the Gemini GenerateContent endpoint about the uploaded video."""

    import aiohttp

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                    {"text": prompt},
                ],
            }
        ]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base}/v1beta/models/{model}:generateContent?key={api_key}",
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status >= 400:
                detail = (await response.text())[:200]
                raise RuntimeError(f"原生视频理解失败：HTTP {response.status} {detail}")
            payload = await response.json(content_type=None)
    candidates = (payload or {}).get("candidates") or []
    if not candidates:
        raise RuntimeError("原生视频理解没有返回候选结果")
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    text = "".join(str(part.get("text") or "") for part in parts).strip()
    if not text:
        raise RuntimeError("原生视频理解返回空结果")
    return text


async def agentic_generate(
    *,
    api_key: str,
    base: str,
    model: str,
    file_uri: str,
    prompt: str,
    mime_type: str = "video/mp4",
    timeout: float = 900.0,
) -> str:
    """Use the official Interactions API for server-side agentic video analysis."""

    import aiohttp

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    body = {
        "model": model,
        "input": [
            {
                "type": "video",
                "uri": file_uri,
                "mime_type": mime_type,
                "processing": "agentic",
            },
            {"type": "text", "text": prompt},
        ],
        "background": True,
    }
    deadline = asyncio.get_running_loop().time() + timeout
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base}/v1beta/interactions", headers=headers, json=body,
            timeout=aiohttp.ClientTimeout(total=min(timeout, 120.0)),
        ) as response:
            if response.status >= 400:
                detail = (await response.text())[:200]
                raise RuntimeError(f"Gemini Agentic Video 启动失败：HTTP {response.status} {detail}")
            payload = await response.json(content_type=None)
        interaction_id = str((payload or {}).get("id") or "")
        if not interaction_id:
            raise RuntimeError("Gemini Agentic Video 未返回 interaction id")
        while asyncio.get_running_loop().time() < deadline:
            status = str((payload or {}).get("status") or "")
            if status in {"completed", "succeeded", "success"}:
                break
            if status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(f"Gemini Agentic Video 失败：{status}")
            await asyncio.sleep(min(2.0, max(0.0, deadline - asyncio.get_running_loop().time())))
            async with session.get(
                f"{base}/v1beta/interactions/{interaction_id}", headers=headers,
                timeout=aiohttp.ClientTimeout(total=min(timeout, 60.0)),
            ) as response:
                if response.status >= 400:
                    detail = (await response.text())[:200]
                    raise RuntimeError(f"Gemini Agentic Video 状态查询失败：HTTP {response.status} {detail}")
                payload = await response.json(content_type=None)
        else:
            raise RuntimeError("Gemini Agentic Video 等待超时")
    output = str((payload or {}).get("output_text") or "").strip()
    if not output:
        for step in (payload or {}).get("steps", []) or []:
            content = step.get("content") or []
            for part in content:
                output += str(part.get("text") or "")
        output = output.strip()
    if not output:
        raise RuntimeError("Gemini Agentic Video 返回空结果")
    return output


async def delete_file(name: str, *, api_key: str, base: str = DEFAULT_BASE) -> None:
    """Best-effort cleanup so we do not leave files on the provider."""

    if not name:
        return
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{base}/v1beta/{name}?key={api_key}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status >= 400:
                    logger.debug(
                        "Decision Engine 删除 Gemini 上传文件失败：HTTP %s", response.status
                    )
    except Exception as exc:  # pragma: no cover - cleanup must never raise
        logger.debug("Decision Engine 删除 Gemini 上传文件异常：%s", str(exc)[:120])


async def analyze(
    path: Path,
    prompt: str,
    *,
    provider: Any,
    model: str,
    timeout: float = 300.0,
    workdir: Path | None = None,
    agentic: bool = False,
) -> NativeResult:
    """Full native video flow with upload, polling, analysis and cleanup."""

    key, base = gemini_credentials(provider)
    if not key:
        return NativeResult(text="", route="frames", error=native_available(provider)[1])
    name = ""
    try:
        name, uri = await upload_file(path, api_key=key, base=base, timeout=timeout)
        await wait_active(name, api_key=key, base=base, timeout=timeout)
        audio_note = (
            "请同时分析视频画面和原始音轨。结合听到的语音、音乐与环境声音"
            "解释画面中人物、事件和时间关系；若没有音轨或听不清，请明确说明。\n\n"
        )
        if agentic:
            text = await agentic_generate(
                api_key=key,
                base=base,
                model=model or "gemini-2.5-flash",
                file_uri=uri,
                prompt=audio_note + prompt,
                mime_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
                timeout=timeout,
            )
        else:
            text = await generate(
                api_key=key,
                base=base,
                model=model or "gemini-2.5-flash",
                file_uri=uri,
                prompt=audio_note + prompt,
                mime_type=mimetypes.guess_type(path.name)[0] or "video/mp4",
                timeout=timeout,
            )
        return NativeResult(text=text, route="native", file_name=name, agentic=agentic)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info("Decision Engine 原生视频理解失败，将回退抽帧：%s", str(exc)[:160])
        return NativeResult(text="", route="frames", error=str(exc)[:200])
    finally:
        if name:
            await delete_file(name, api_key=key, base=base)

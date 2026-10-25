"""Provider discovery, capability checks and the multi-modal call itself.

Nothing here is Gemini specific: every configured AstrBot chat provider is a
candidate. Capability data comes from the provider's own ``modalities`` setting,
and every media type is reduced to something the provider accepts through the
pre-processing layer, so a text-only or image-only model still has a usable path.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from astrbot import logger

# Which provider modality a media kind ultimately needs.
# Video is understood as a frame sequence (image) plus optional audio; a file is
# reduced to text locally, so it needs no provider modality at all.
KIND_MODALITIES: dict[str, tuple[str, ...]] = {
    "image": ("image",),
    "video": ("image",),
    "audio": ("audio",),
    "file": (),
}

MODALITY_LABELS = {"image": "图像", "audio": "音频", "tool_use": "工具"}
KIND_LABELS = {"image": "图片", "video": "视频", "audio": "音频", "file": "文件"}


@dataclass(slots=True)
class ProviderChoice:
    provider: Any
    provider_id: str
    model: str
    needed: tuple[str, ...]

    def missing_modalities(self) -> list[str]:
        return [name for name in self.needed if not supports_modality(self.provider, name)]


def _modalities(provider: Any) -> list[str] | None:
    config = getattr(provider, "provider_config", None) or {}
    raw = config.get("modalities", None)
    if raw == []:
        # An empty list is AstrBot's "unconfigured" marker, meaning all supported.
        return None
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return None


def supports_modality(provider: Any, modality: str) -> bool:
    declared = _modalities(provider)
    if declared is None:
        return True
    return modality in declared


def kind_supported(provider: Any, kind: str) -> bool:
    return all(supports_modality(provider, name) for name in KIND_MODALITIES.get(kind, ()))


def classify_kind(provider: Any, kind: str) -> str:
    """How a media kind can be handled: native, preprocessed or unsupported.

    ``native`` means the provider accepts the modality itself. ``preprocessed``
    means the plugin reduces the media to something the provider does take (video
    to frames, a document to text). ``unsupported`` means there is no usable path.
    """

    declared = _modalities(provider)
    if kind == "video":
        from .native_video import native_available

        if native_available(provider)[0]:
            return "native"
        if declared is None:
            return "preprocessed"
        return "preprocessed" if "image" in declared else "unsupported"
    if declared is None:
        return "native"
    if kind == "file":
        return "preprocessed"
    if kind == "video":
        # A native label is only valid when the native upload route is really
        # available for this provider, never from a modality declaration alone.
        from .native_video import native_available

        if native_available(provider)[0]:
            return "native"
        return "preprocessed" if "image" in declared else "unsupported"
    if kind == "image":
        return "native" if "image" in declared else "unsupported"
    if kind == "audio":
        return "native" if "audio" in declared else "unsupported"
    return "unsupported"


def describe_provider(provider: Any) -> dict[str, Any]:
    meta = None
    try:
        meta = provider.meta()
    except Exception:  # pragma: no cover - metadata is provider specific
        meta = None
    provider_id = str(getattr(meta, "id", "") or "")
    model = str(getattr(meta, "model", "") or "")
    type_name = str(getattr(meta, "type", "") or "")
    declared = _modalities(provider)
    caps = ["text"] if declared is None else declared
    return {
        "id": provider_id,
        "model": model,
        "type": type_name,
        "label": f"{provider_id} · {model or type_name}".strip(" ·"),
        "modalities": caps,
        "modalities_label": "、".join(
            MODALITY_LABELS.get(name, name) for name in caps if name != "tool_use"
        ),
        "supports": {
            kind: kind_supported(provider, kind) for kind in KIND_LABELS
        },
        "capability": {kind: classify_kind(provider, kind) for kind in KIND_LABELS},
    }


def provider_catalog(context: Any) -> list[dict[str, Any]]:
    """Every chat provider AstrBot currently has configured."""

    catalog: list[dict[str, Any]] = []
    getter = getattr(context, "get_all_providers", None)
    if not callable(getter):
        return catalog
    try:
        providers = getter()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Decision Engine 读取模型列表失败：%s", str(exc)[:160])
        return catalog
    for provider in providers or ():
        try:
            catalog.append(describe_provider(provider))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Decision Engine 跳过不可用模型：%s", str(exc)[:120])
    return catalog


def resolve_provider(context: Any, provider_id: str, kind: str) -> ProviderChoice | None:
    getter = getattr(context, "get_provider_by_id", None)
    if not callable(getter) or not provider_id:
        return None
    try:
        provider = getter(provider_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Decision Engine 获取多模态模型 %s 失败：%s", provider_id, str(exc)[:160]
        )
        return None
    if provider is None:
        return None
    meta = None
    try:
        meta = provider.meta()
    except Exception:  # pragma: no cover
        meta = None
    return ProviderChoice(
        provider=provider,
        provider_id=provider_id,
        model=str(getattr(meta, "model", "") or ""),
        needed=KIND_MODALITIES.get(kind, ()),
    )


def pick_provider(context: Any, settings: Any, kind: str) -> tuple[ProviderChoice | None, str]:
    """Choose the configured provider for ``kind``.

    Returns ``(choice, error)``. The error explains why nothing usable was found so
    the caller can surface it instead of failing silently.
    """

    preferred = ""
    try:
        preferred = settings.model_for(kind)
    except AttributeError:
        preferred = str(settings.get(f"mm_model_{kind}") or settings.get("mm_default_model") or "")
    if not preferred:
        return None, f"未选择{KIND_LABELS.get(kind, kind)}理解模型，请在 WebUI 中配置"
    choice = resolve_provider(context, preferred, kind)
    if choice is None:
        return None, f"模型 {preferred} 不存在或已禁用"
    missing = choice.missing_modalities()
    if missing:
        labels = "、".join(MODALITY_LABELS.get(name, name) for name in missing)
        return None, f"模型 {preferred} 未声明支持{labels}能力"
    return choice, ""


async def call_provider(
    provider: Any,
    *,
    prompt: str,
    images: list[str] | None = None,
    audio: list[str] | None = None,
    model: str | None = None,
    timeout: float = 180.0,
    stream: bool = True,
) -> str:
    """Run one multi-modal completion and return the plain text answer."""

    chat = getattr(provider, "text_chat", None)
    if not callable(chat):
        raise RuntimeError("模型不支持文本对话接口")
    stream_chat = getattr(provider, "text_chat_stream", None)
    # Receive the host's stream contract (deltas + final snapshot) to avoid a
    # gateway's buffered-response deadline. Providers without that interface
    # keep the ordinary completion path; an enabled stream is never retried
    # through a second paid request just because its transport failed.
    use_stream = stream and inspect.isasyncgenfunction(stream_chat)
    if use_stream:
        chat = stream_chat

    kwargs: dict[str, Any] = {"prompt": prompt}
    if images:
        kwargs["image_urls"] = list(images)
    if audio:
        kwargs["audio_urls"] = list(audio)
    if model:
        kwargs["model"] = model
    # Decision owns mm_retries. A host provider's default five-attempt retry
    # loop otherwise multiplies each configured attempt and can bill twice
    # after a response timeout. Use its explicit per-call override when present.
    try:
        if "request_max_retries" in inspect.signature(chat).parameters:
            kwargs["request_max_retries"] = 1
    except (TypeError, ValueError):
        pass

    if use_stream:
        return await asyncio.wait_for(_stream_final_text(chat(**kwargs)), timeout=timeout)
    response = await asyncio.wait_for(chat(**kwargs), timeout=timeout)
    return _extract_text(response)


async def _stream_final_text(stream: Any) -> str:
    """Consume the host's complete snapshot; never cache partial/thought text."""

    final = ""
    async with aclosing(stream):
        async for response in stream:
            is_chunk = response.get("is_chunk", False) if isinstance(response, dict) else getattr(response, "is_chunk", False)
            if not is_chunk:
                final = _extract_text(response)
    if not final:
        raise RuntimeError("流式理解未返回完整结果")
    return final


def safe_provider_error(provider: Any, exc: BaseException) -> str:
    """Redact known provider credentials before truncating any SDK error."""
    message = str(exc) or type(exc).__name__
    config = getattr(provider, "provider_config", None) or {}
    secrets = []
    for name in ("key", "api_key", "api_keys", "token", "access_token"):
        value = config.get(name)
        secrets.extend(value if isinstance(value, list) else [value])
    for method in ("get_current_key", "get_keys"):
        getter = getattr(provider, method, None)
        if callable(getter):
            try:
                value = getter()
                secrets.extend(value if isinstance(value, list) else [value])
            except Exception:
                pass
    for secret in secrets:
        if isinstance(secret, str) and secret:
            message = message.replace(secret, "<redacted>")
    return message[:200]


def _extract_text(response: Any) -> str:
    if response is None:
        return ""
    text = getattr(response, "completion_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    if isinstance(response, str):
        return response.strip()
    result = getattr(response, "result", None)
    if isinstance(result, str):
        return result.strip()
    if isinstance(response, dict):
        for name in ("completion_text", "result"):
            value = response.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    # Object representations can contain thoughts, tool calls, credentials or
    # SDK metadata. They are never a substitute for a final model answer.
    return ""

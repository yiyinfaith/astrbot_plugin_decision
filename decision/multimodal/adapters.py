"""Provider discovery, capability checks and the multi-modal call itself.

Nothing here is Gemini specific: every configured AstrBot chat provider is a
candidate. Capability data comes from the provider's own ``modalities`` setting,
and every media type is reduced to something the provider accepts through the
pre-processing layer, so a text-only or image-only model still has a usable path.
"""

from __future__ import annotations

import asyncio
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
) -> str:
    """Run one multi-modal completion and return the plain text answer."""

    chat = getattr(provider, "text_chat", None)
    if not callable(chat):
        raise RuntimeError("模型不支持文本对话接口")

    kwargs: dict[str, Any] = {"prompt": prompt}
    if images:
        kwargs["image_urls"] = list(images)
    if audio:
        kwargs["audio_urls"] = list(audio)
    if model:
        kwargs["model"] = model

    response = await asyncio.wait_for(chat(**kwargs), timeout=timeout)
    return _extract_text(response)


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
    try:
        return str(response).strip()
    except Exception:  # pragma: no cover
        return ""

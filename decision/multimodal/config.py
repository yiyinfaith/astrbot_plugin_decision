"""Multi-modal understanding: configuration defaults, validation and page payload.

The module is deliberately independent from ``conversation_flow_analysis_enabled``:
media understanding and dialogue participant inference are separate sub-features
that only share the native "对话流增强" master switch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MEDIA_KINDS = ("image", "video", "audio", "file")
KIND_LABELS = {
    "image": "图片",
    "video": "视频",
    "audio": "音频",
    "file": "文件",
}

# Conservative defaults: nothing is understood or charged until the operator turns
# it on explicitly.
MULTIMODAL_DEFAULTS: dict[str, Any] = {
    # --- master switches ---------------------------------------------------
    "mm_enabled": False,
    "mm_tools_enabled": False,
    "mm_stream_enabled": True,
    "mm_gateway_openai_stream": False,
    # --- per-kind switches -------------------------------------------------
    "mm_auto_image": False,
    "mm_auto_video": False,
    "mm_auto_audio": False,
    "mm_auto_file": False,
    "mm_tool_image": True,
    "mm_tool_video": True,
    "mm_tool_audio": True,
    "mm_tool_file": True,
    # --- auto understanding scope (whitelist by default) ------------------
    "mm_auto_scope": [],
    "mm_auto_scope_blacklist": False,
    # --- model selection ---------------------------------------------------
    "mm_default_model": "",
    "mm_model_image": "",
    "mm_model_video": "",
    "mm_model_audio": "",
    "mm_model_file": "",
    # --- size and budget limits -------------------------------------------
    "mm_max_file_mb": 40,
    "mm_max_auto_per_message": 4,
    "mm_max_auto_per_minute": 12,
    "mm_inject_max_chars": 1200,
    # How many earlier media summaries a text-only question may recall.
    "mm_recall_items": 3,
    # --- image / animation -------------------------------------------------
    "mm_gif_max_frames": 16,
    "mm_gif_max_pixels": 40_000_000,
    "mm_image_max_edge": 1568,
    "mm_sanitize_animated_images": True,
    # --- video -------------------------------------------------------------
    "mm_video_short_seconds": 60,
    "mm_video_max_seconds": 600,
    "mm_video_max_frames": 24,
    "mm_video_agentic_rounds": 2,
    "mm_video_agentic_segments": 6,
    # --- audio -------------------------------------------------------------
    "mm_audio_max_seconds": 600,
    "mm_audio_segment_seconds": 120,
    # --- file --------------------------------------------------------------
    "mm_file_max_chars": 12_000,
    # auto: render PPTX pages and only explicitly targeted DOCX pages;
    # always: render every DOCX/PPTX request; never: embedded assets/text only.
    "mm_office_render": "auto",
    "mm_office_render_max_pages": 12,
    "mm_office_render_timeout_seconds": 90,
    # Scanned pages are rendered and read by the vision model. Keeping this
    # explicit means a plain text PDF never costs an image request.
    "mm_file_ocr": True,
    # auto / local_first / vision_first / local_only
    "mm_ocr_strategy": "auto",
    # auto / tesseract / paddleocr
    "mm_ocr_engine": "auto",
    # auto / native / frames
    "mm_video_strategy": "auto",
    "mm_ocr_lang": "chi_sim+eng",
    "mm_ocr_timeout_seconds": 120,
    "mm_file_ocr_max_pages": 5,
    "mm_file_chunk_chars": 6_000,
    # --- runtime -----------------------------------------------------------
    "mm_timeout_seconds": 180,
    "mm_concurrency": 2,
    "mm_retries": 1,
    "mm_cache_ttl_hours": 12,
    "mm_cache_max_entries": 500,
    # How long a reply decision may wait for an in-flight understanding.
    "mm_auto_wait_seconds": 0,
}

BOOL_KEYS = frozenset(key for key, value in MULTIMODAL_DEFAULTS.items() if isinstance(value, bool))
LIST_KEYS = frozenset(key for key, value in MULTIMODAL_DEFAULTS.items() if isinstance(value, list))
STR_KEYS = frozenset(
    {key for key, value in MULTIMODAL_DEFAULTS.items() if isinstance(value, str)}
)

# Inclusive bounds for the numeric settings, used both for validation and to keep
# hand-edited config files inside a safe range.
INT_BOUNDS: dict[str, tuple[int, int]] = {
    "mm_max_file_mb": (1, 500),
    "mm_max_auto_per_message": (1, 16),
    "mm_max_auto_per_minute": (0, 600),
    "mm_inject_max_chars": (120, 8000),
    "mm_recall_items": (0, 10),
    "mm_gif_max_frames": (2, 64),
    "mm_gif_max_pixels": (1_000_000, 200_000_000),
    "mm_image_max_edge": (256, 4096),
    "mm_video_short_seconds": (5, 600),
    "mm_video_max_seconds": (10, 3600),
    "mm_video_max_frames": (2, 96),
    "mm_video_agentic_rounds": (1, 5),
    "mm_video_agentic_segments": (2, 24),
    "mm_audio_max_seconds": (10, 3600),
    "mm_audio_segment_seconds": (20, 600),
    "mm_file_max_chars": (500, 200_000),
    "mm_file_ocr_max_pages": (1, 30),
    "mm_office_render_max_pages": (1, 30),
    "mm_office_render_timeout_seconds": (10, 300),
    "mm_ocr_timeout_seconds": (10, 600),
    "mm_file_chunk_chars": (500, 100_000),
    "mm_timeout_seconds": (10, 900),
    "mm_concurrency": (1, 8),
    "mm_retries": (0, 3),
    "mm_cache_ttl_hours": (0, 720),
    "mm_cache_max_entries": (10, 10_000),
    "mm_auto_wait_seconds": (0, 30),
}


@dataclass(slots=True)
class MultimodalSettings:
    """Validated view of the multi-modal configuration."""

    values: dict[str, Any] = field(default_factory=lambda: dict(MULTIMODAL_DEFAULTS))

    def get(self, key: str) -> Any:
        return self.values.get(key, MULTIMODAL_DEFAULTS.get(key))

    def bool(self, key: str) -> bool:
        return bool(self.values.get(key, MULTIMODAL_DEFAULTS.get(key)))

    def int(self, key: str) -> int:
        try:
            return int(self.values.get(key, MULTIMODAL_DEFAULTS.get(key)) or 0)
        except (TypeError, ValueError):
            return int(MULTIMODAL_DEFAULTS.get(key, 0) or 0)

    def text(self, key: str) -> str:
        return str(self.values.get(key, MULTIMODAL_DEFAULTS.get(key)) or "").strip()

    def items(self) -> list[str]:
        raw = self.values.get("mm_auto_scope")
        return [str(item).strip() for item in (raw or []) if str(item).strip()]

    # --- derived helpers ---------------------------------------------------
    @property
    def auto_kinds(self) -> tuple[str, ...]:
        if not self.bool("mm_enabled"):
            return ()
        return tuple(kind for kind in MEDIA_KINDS if self.bool(f"mm_auto_{kind}"))

    @property
    def tool_kinds(self) -> tuple[str, ...]:
        if not self.bool("mm_enabled") or not self.bool("mm_tools_enabled"):
            return ()
        return tuple(kind for kind in MEDIA_KINDS if self.bool(f"mm_tool_{kind}"))

    def model_for(self, kind: str) -> str:
        override = self.text(f"mm_model_{kind}")
        return override or self.text("mm_default_model")

    def scope_allows(self, umo: str, *, matcher=None) -> bool:
        """Apply the auto-understanding black/white list.

        Whitelist (default) with an empty list allows nobody, which keeps a fresh
        install from spending money on every media message in every group.
        """

        entries = self.items()
        matched = bool(matcher(umo, entries)) if matcher else umo in entries
        if self.bool("mm_auto_scope_blacklist"):
            return not matched
        return matched


def normalize_settings(raw: Any) -> dict[str, Any]:
    """Merge stored values over the defaults, coercing types defensively."""

    merged = dict(MULTIMODAL_DEFAULTS)
    if not isinstance(raw, Mapping):
        return merged
    for key, default in MULTIMODAL_DEFAULTS.items():
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(default, bool):
            merged[key] = bool(value)
        elif isinstance(default, list):
            if isinstance(value, (list, tuple)):
                merged[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(default, int):
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            low, high = INT_BOUNDS.get(key, (number, number))
            merged[key] = max(low, min(high, number))
        elif isinstance(default, str):
            merged[key] = str(value or "").strip()
    return merged


def validate_page_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate the WebUI payload for the multi-modal section.

    Returns ``(cleaned, error)`` where ``error`` is an empty string on success.
    """

    cleaned: dict[str, Any] = {}
    for key, default in MULTIMODAL_DEFAULTS.items():
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                return {}, f"{key} must be a boolean"
            cleaned[key] = value
        elif isinstance(default, list):
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                return {}, f"{key} must be a string list"
            cleaned[key] = [item.strip() for item in value if item.strip()]
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int):
                return {}, f"{key} must be an integer"
            low, high = INT_BOUNDS.get(key, (value, value))
            if not low <= value <= high:
                return {}, f"{key} must be between {low} and {high}"
            cleaned[key] = value
        else:
            if not isinstance(value, str):
                return {}, f"{key} must be a string"
            cleaned[key] = value.strip()
    # Cross-field sanity: the short/long boundary must stay inside the hard limit.
    short = cleaned.get(
        "mm_video_short_seconds", MULTIMODAL_DEFAULTS["mm_video_short_seconds"]
    )
    longest = cleaned.get("mm_video_max_seconds", MULTIMODAL_DEFAULTS["mm_video_max_seconds"])
    if short > longest:
        return {}, "mm_video_short_seconds must not exceed mm_video_max_seconds"
    return cleaned, ""


def page_settings(settings: MultimodalSettings) -> dict[str, Any]:
    """The editable section the WebUI renders under 输入增强."""

    payload = dict(settings.values)
    from .ocr import engine_status
    from .preprocess import tooling_status

    payload["mm_tooling"] = {**tooling_status(), "ocr": engine_status()}
    payload["mm_auto_kinds_label"] = ", ".join(
        KIND_LABELS[kind] for kind in settings.auto_kinds
    )
    payload["mm_tool_kinds_label"] = ", ".join(
        KIND_LABELS[kind] for kind in settings.tool_kinds
    )
    return payload

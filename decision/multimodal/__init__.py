"""Multi-modal understanding package.

Public surface used by ``main.py``:

* :class:`~.runtime.MultimodalRuntime` – lifecycle, auto understanding, injection
* :func:`~.config.normalize_settings` / :func:`~.config.page_settings` – config
* :func:`~.adapters.provider_catalog` – the WebUI model dropdown
* :func:`~.tools.build_tools` – the four ``understand_*`` LLM tools
"""

from __future__ import annotations

from .config import (
    KIND_LABELS,
    MEDIA_KINDS,
    MULTIMODAL_DEFAULTS,
    MultimodalSettings,
    normalize_settings,
    page_settings,
    validate_page_payload,
)

__all__ = [
    "KIND_LABELS",
    "MEDIA_KINDS",
    "MULTIMODAL_DEFAULTS",
    "MultimodalRuntime",
    "MultimodalSettings",
    "normalize_settings",
    "page_settings",
    "validate_page_payload",
]


def __getattr__(name: str):
    # Imported lazily so config-only consumers (tests, WebUI helpers) do not need
    # the full AstrBot provider stack.
    if name == "MultimodalRuntime":
        from .runtime import MultimodalRuntime

        return MultimodalRuntime
    raise AttributeError(name)

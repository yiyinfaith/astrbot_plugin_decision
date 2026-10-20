"""Minimal AstrBot doubles so the service tests run without an AstrBot runtime.

The doubles are installed into :data:`sys.modules` only when a real ``astrbot``
distribution is unavailable, so the same tests exercise the genuine framework on a
host that has AstrBot installed.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from types import ModuleType
from typing import Any


class TextPart:
    """Stand-in for ``astrbot.core.agent.message.TextPart``."""

    def __init__(self, text: str = "", **kwargs: Any):
        self.text = text
        self.type = "text"
        self.__dict__.update(kwargs)

    def mark_as_temp(self) -> TextPart:
        self.temporary = True
        return self


class PluginUploadFile:
    """Stand-in for ``astrbot.api.web.PluginUploadFile``."""

    def __init__(self, data: bytes = b"", content_length: int | None = None):
        self._data = data
        self.content_length = len(data) if content_length is None else content_length

    async def read(self, size: int = -1) -> bytes:
        return self._data if size < 0 else self._data[:size]


class RequestStub:
    """Mutable stand-in for the active Page request."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> RequestStub:
        self.method = "GET"
        self.query: dict[str, Any] = {}
        self.json_body: Any = {}
        self.files_map: dict[str, Any] = {}
        return self

    async def json(self, default: Any = None) -> Any:
        return self.json_body if self.json_body is not None else default

    async def files(self) -> dict[str, Any]:
        return self.files_map


def json_response(data: Any, **kwargs: Any) -> Any:
    return data


def error_response(message: str, status_code: int = 500, **kwargs: Any) -> dict[str, Any]:
    return {"error": message, "status_code": status_code}


def file_response(path: Any, **kwargs: Any) -> dict[str, Any]:
    return {"path": str(path), **kwargs}


class Plain:
    """Stand-in for ``astrbot.api.message_components.Plain``."""

    def __init__(self, text: str = "", **kwargs: Any):
        self.text = text
        self.type = "plain"
        self.__dict__.update(kwargs)


class Image:
    """Stand-in for ``astrbot.api.message_components.Image``."""

    def __init__(self, file: str = "", **kwargs: Any):
        self.file = file
        self.type = "image"
        self.__dict__.update(kwargs)

    @classmethod
    def fromFileSystem(cls, path: Any) -> Image:
        return cls(file=str(path))


def _real_astrbot_available() -> bool:
    """Return whether a genuine AstrBot distribution is importable.

    ``find_spec`` raises when another test module has already installed a bare
    ``ModuleType`` named ``astrbot`` (such a stub has no ``__spec__``), so a
    missing spec is treated as "already stubbed" rather than as an error.
    """

    try:
        return importlib.util.find_spec("astrbot") is not None
    except (ImportError, ValueError):
        return False


def install() -> None:
    """Install AstrBot doubles when the real runtime is not importable.

    Idempotent and additive: existing modules are reused and only missing
    attributes are filled in. Recreating them would replace ``web.request``,
    and ``PersonalNetworkService`` holds that object by value, so a second
    install from another test module would silently break it.
    """

    if _real_astrbot_available():
        return

    def module(name: str) -> ModuleType:
        existing = sys.modules.get(name)
        if existing is None:
            existing = ModuleType(name)
            sys.modules[name] = existing
        return existing

    astrbot = module("astrbot")
    if not hasattr(astrbot, "logger"):
        astrbot.logger = logging.getLogger("tests.personal_network.astrbot")
    api = module("astrbot.api")
    api.logger = astrbot.logger
    web = module("astrbot.api.web")
    web.PluginUploadFile = PluginUploadFile
    web.json_response = json_response
    web.error_response = error_response
    web.file_response = file_response
    if not isinstance(getattr(web, "request", None), RequestStub):
        web.request = RequestStub()
    components = module("astrbot.api.message_components")
    components.Plain = Plain
    components.Image = Image
    core = module("astrbot.core")
    agent = module("astrbot.core.agent")
    message = module("astrbot.core.agent.message")
    message.TextPart = TextPart
    api.web = web
    api.message_components = components
    core.agent = agent
    agent.message = message
    astrbot.api = api
    astrbot.core = core


def reset_request() -> RequestStub:
    """Reset and return the shared request double.

    The service imports ``request`` by value, so the existing object is mutated in
    place instead of being replaced.
    """
    return sys.modules["astrbot.api.web"].request.reset()

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


def install() -> None:
    """Install AstrBot doubles when the real runtime is not importable."""
    if importlib.util.find_spec("astrbot") is not None:
        return
    astrbot = ModuleType("astrbot")
    astrbot.logger = logging.getLogger("tests.personal_network.astrbot")
    api = ModuleType("astrbot.api")
    api.logger = astrbot.logger
    web = ModuleType("astrbot.api.web")
    web.PluginUploadFile = PluginUploadFile
    web.json_response = json_response
    web.error_response = error_response
    web.file_response = file_response
    web.request = RequestStub()
    core = ModuleType("astrbot.core")
    agent = ModuleType("astrbot.core.agent")
    message = ModuleType("astrbot.core.agent.message")
    message.TextPart = TextPart
    api.web = web
    core.agent = agent
    agent.message = message
    astrbot.api = api
    astrbot.core = core
    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.web": web,
            "astrbot.core": core,
            "astrbot.core.agent": agent,
            "astrbot.core.agent.message": message,
        }
    )


def reset_request() -> RequestStub:
    """Reset and return the shared request double.

    The service imports ``request`` by value, so the existing object is mutated in
    place instead of being replaced.
    """
    return sys.modules["astrbot.api.web"].request.reset()

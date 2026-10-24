"""Install minimal AstrBot doubles before the service module is imported."""

from __future__ import annotations

import importlib

import pytest

from ._astrbot_stub import RequestStub, error_response, file_response, install, json_response

install()


@pytest.fixture(autouse=True)
def isolated_web_request(monkeypatch):
    """Service unit tests always use an explicit request/response boundary.

    Installing AstrBot must not make direct service calls depend on a live
    FastAPI ContextVar or change their return type. Native runner integration
    remains in test_life_native.py and uses the genuine framework separately.
    """

    service = importlib.import_module("decision.personal_network.service")
    web = importlib.import_module("astrbot.api.web")
    request = RequestStub()
    monkeypatch.setattr(web, "request", request)
    monkeypatch.setattr(service, "request", request)
    monkeypatch.setattr(service, "json_response", json_response)
    monkeypatch.setattr(service, "error_response", error_response)
    monkeypatch.setattr(service, "file_response", file_response)

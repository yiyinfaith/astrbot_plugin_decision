"""The plan generator must see the per-persona appearance description."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.personal_network._astrbot_stub import install  # noqa: E402

install()

from decision.virtual_life.core.generator import DailyPlanGenerator  # noqa: E402

FALLBACK = "无（未填写外貌补充描述；若有外貌参考图，请以参考图为准）"


def generator(config=None, resolver=None):
    instance = DailyPlanGenerator.__new__(DailyPlanGenerator)
    instance.context = None
    instance.config = config or {}
    instance.generating = set()
    if resolver is not None:
        instance.appearance_resolver = resolver
    return instance


@pytest.mark.asyncio
async def test_per_persona_description_wins():
    async def resolver(persona_id):
        return "银色长发，红色眼睛" if persona_id == "alice" else ""

    instance = generator(
        {"appearance_settings": {"description": "全局补充"}}, resolver
    )
    assert await instance._appearance_context("alice") == "银色长发，红色眼睛"


@pytest.mark.asyncio
async def test_global_description_is_the_fallback():
    async def resolver(persona_id):
        return ""

    instance = generator(
        {"appearance_settings": {"description": "全局补充"}}, resolver
    )
    assert await instance._appearance_context("bob") == "全局补充"


@pytest.mark.asyncio
async def test_missing_description_returns_the_documented_placeholder():
    instance = generator({"appearance_settings": {"description": ""}}, None)
    assert await instance._appearance_context("alice") == FALLBACK


@pytest.mark.asyncio
async def test_resolver_failure_falls_back_to_the_global_value():
    def resolver(persona_id):
        raise RuntimeError("store offline")

    instance = generator(
        {"appearance_settings": {"description": "全局补充"}}, resolver
    )
    assert await instance._appearance_context("alice") == "全局补充"


@pytest.mark.asyncio
async def test_sync_resolver_is_accepted():
    instance = generator({"appearance_settings": {"description": "全局"}}, lambda pid: "同步结果")
    assert await instance._appearance_context("alice") == "同步结果"


@pytest.mark.asyncio
async def test_no_persona_id_uses_the_global_value():
    async def resolver(persona_id):
        return "不该被使用"

    instance = generator({"appearance_settings": {"description": "全局"}}, resolver)
    assert await instance._appearance_context(None) == "全局"

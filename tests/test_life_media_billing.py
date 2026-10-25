"""The proactive media billing account must be the configured one, not the session."""

from __future__ import annotations

import base64
import io
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.personal_network._astrbot_stub import install  # noqa: E402

install()

from decision.life_media import BillingEvent, LifeMedia  # noqa: E402


class FakeAigen:
    def __init__(self):
        self.image_calls: list[dict] = []

    async def generate_image(
        self, event=None, prompt: str = "", aspect_ratio: str = "", n=None, **_ignored
    ):
        self.image_calls.append({"event": event, "prompt": prompt, "n": n})
        yield "image-ok"


class FakeEvent:
    """A CronMessageEvent-like event: sender is the session id, platform is real."""

    def __init__(self, session_user="1046209937", platform="aiocqhttp", extras=None):
        self.unified_msg_origin = f"{platform}:GroupMessage:{session_user}"
        self._sender = session_user
        self._platform = platform
        self.sent = []
        self._extras = dict(extras or {})
        self.message_obj = SimpleNamespace(message=[], message_str="", raw_message=None)

    def get_platform_name(self):
        return self._platform

    def get_sender_id(self):
        return self._sender

    def get_sender_name(self):
        return "Scheduler"

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    async def send(self, value):
        self.sent.append(value)


def media_settings(**overrides):
    base = {
        "enable": True,
        "photo_enable": True,
        "video_enable": True,
        "audio_enable": False,
        "image_aspect_ratio": "3:4",
        "video_aspect_ratio": "9:16",
        "video_duration": 5,
        "cooldown_seconds": 0,
        "daily_limit": 10,
        "photo_count_min": 1,
        "photo_count_max": 1,
        "video_count_min": 1,
        "video_count_max": 1,
        "audio_count_min": 1,
        "audio_count_max": 1,
        "audio_mode": "朗读",
        "audio_voice": "",
        "photo_prompt": "照片 {subject_desc} 数量{count} 穿搭：{outfit}",
        "video_prompt": "视频 {subject_desc} 数量{count} 穿搭：{outfit}",
        "audio_prompt": "音频 {style} {text} 数量{count}",
        "proactive_billing_mode": "fixed",
        "proactive_billing_platform": "aiocqhttp",
        "proactive_billing_user_id": "2980555633",
    }
    base.update(overrides)
    return base


async def make_media(tmp_path, *, settings=None, aigen=None):
    settings = settings or media_settings()
    aigen = aigen if aigen is not None else FakeAigen()
    plan = SimpleNamespace(has_outfit=True, outfit=None, date="2026-10-21", timeline=(),
                           theme="t", mood="m")

    async def resolve(_umo):
        return SimpleNamespace(id="alice")

    async def ensure_plan(_umo):
        return SimpleNamespace(id="alice"), plan

    runtime = SimpleNamespace(
        config={"media_settings": settings,
                "appearance_settings": {"description": "银色长发", "use_reference_outfit": True}},
        closed=False,
        personas=SimpleNamespace(resolve=resolve),
        _ensure_plan_for_umo=ensure_plan,
        _now=lambda: datetime(2026, 10, 21, 12, 0),
        can_commit=lambda umo, generation: True,
    )

    async def get_runtime():
        return runtime

    owner = SimpleNamespace(
        context=SimpleNamespace(get_all_stars=lambda: [
            SimpleNamespace(name="astrbot_plugin_aigen", activated=True, star_cls=aigen)]),
        _get_life_runtime=get_runtime,
    )
    media = LifeMedia(owner, tmp_path)
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")
    await media.appearance.upload(
        "alice", "ref.png",
        "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
    )
    return media, aigen


def proactive_event(**extra):
    base = {"decision_life_proactive": True, "decision_life_generation": 0,
            "decision_life_media_kind": "photo", "decision_life_media_max": 10}
    base.update(extra)
    return FakeEvent(extras=base)


@pytest.mark.asyncio
async def test_proactive_bills_the_configured_account(tmp_path):
    media, aigen = await make_media(tmp_path)
    await media.capture(proactive_event(), kind="photo", count=1)

    billed = aigen.image_calls[0]["event"]
    assert billed.get_platform_name() == "aiocqhttp"
    assert billed.get_sender_id() == "2980555633"


@pytest.mark.asyncio
async def test_proactive_requires_explicit_platform_and_user(tmp_path):
    media, aigen = await make_media(
        tmp_path, settings=media_settings(proactive_billing_platform="")
    )
    result = await media.capture(proactive_event(), kind="photo", count=1)
    assert "明确填写" in result
    assert aigen.image_calls == []


@pytest.mark.asyncio
async def test_explicit_platform_overrides_the_current_one(tmp_path):
    settings = media_settings(proactive_billing_platform="telegram")
    media, aigen = await make_media(tmp_path, settings=settings)
    await media.capture(proactive_event(), kind="photo", count=1)
    assert aigen.image_calls[0]["event"].get_platform_name() == "telegram"


@pytest.mark.asyncio
async def test_ordinary_chat_is_unaffected(tmp_path):
    """A normal chat must still bill the person who spoke."""

    media, aigen = await make_media(tmp_path)
    event = FakeEvent(session_user="2980555633")
    await media.capture(event, kind="photo", count=1)

    billed = aigen.image_calls[0]["event"]
    # Not wrapped: the real event answers with its own identity.
    assert not isinstance(billed, BillingEvent)
    assert billed.get_sender_id() == "2980555633"
    assert billed.get_platform_name() == "aiocqhttp"


@pytest.mark.asyncio
async def test_session_mode_rejects_proactive_paid_media(tmp_path):
    settings = media_settings(proactive_billing_mode="session")
    media, aigen = await make_media(tmp_path, settings=settings)
    await media.capture(proactive_event(), kind="photo", count=1)

    assert aigen.image_calls == []


@pytest.mark.asyncio
async def test_blank_user_id_rejects_proactive_paid_media(tmp_path):
    settings = media_settings(proactive_billing_user_id="")
    media, aigen = await make_media(tmp_path, settings=settings)
    result = await media.capture(proactive_event(), kind="photo", count=1)
    assert "明确填写" in result
    assert aigen.image_calls == []


def test_billing_event_proxies_everything_else():
    real = FakeEvent(session_user="1046209937")
    wrapped = BillingEvent(real, "aiocqhttp", "2980555633")

    assert wrapped.get_sender_id() == "2980555633"
    assert wrapped.get_platform_name() == "aiocqhttp"
    # Media extraction and sending still reach the real event.
    assert wrapped.unified_msg_origin == real.unified_msg_origin
    assert wrapped.message_obj is real.message_obj

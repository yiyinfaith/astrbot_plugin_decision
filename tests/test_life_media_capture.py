"""Exercise ``LifeMedia.capture`` call counts and guardrails without AstrBot.

The AstrBot doubles from ``tests.personal_network._astrbot_stub`` are reused so
``decision.life_media`` imports on a host without an AstrBot runtime.
"""

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


@pytest.fixture(scope="module", autouse=True)
def _astrbot_doubles():
    """Install the doubles before ``decision.life_media`` is imported.

    Installing at import time would win the race against the personal-network
    conftest and leave that suite with a half-configured stub, so the import is
    deferred into this fixture.
    """

    install()


from decision.life_media import LifeMedia  # noqa: E402


class FakeAigen:
    """Records every generation call and yields a plain status string.

    The parameter names mirror the real aigen API because ``LifeMedia.capture``
    filters its kwargs through ``inspect.signature`` of the bound method.
    """

    def __init__(self):
        self.image_calls: list[dict] = []
        self.video_calls: list[dict] = []
        self.audio_calls: list[dict] = []

    async def generate_image(
        self,
        event=None,
        prompt: str = "",
        resolution: str = "",
        aspect_ratio: str = "",
        n: int | None = None,
        use_message_media: bool = True,
        **_ignored,
    ):
        self.image_calls.append(
            {"event": event, "prompt": prompt, "aspect_ratio": aspect_ratio, "n": n}
        )
        yield "image-ok"

    async def generate_video(
        self,
        event=None,
        prompt: str = "",
        duration: float = 0,
        aspect_ratio: str = "",
        reference_mode: str = "auto",
        use_message_media: bool = True,
        **_ignored,
    ):
        self.video_calls.append(
            {"event": event, "prompt": prompt, "duration": duration,
             "aspect_ratio": aspect_ratio, "reference_mode": reference_mode}
        )
        yield "video-ok"

    async def generate_audio(
        self,
        event=None,
        text: str = "",
        mode: str = "朗读",
        voice: str = "",
        **_ignored,
    ):
        self.audio_calls.append({"event": event, "text": text, "mode": mode, "voice": voice})
        yield "audio-ok"


class FakeEvent:
    unified_msg_origin = "aiocqhttp:FriendMessage:1"

    def __init__(self, extras=None):
        self.sent = []
        self._extras = dict(extras or {})
        # ``ReferenceEvent`` copies this and replaces its message list.
        self.message_obj = SimpleNamespace(message=[], message_str="", raw_message=None)

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
        "audio_enable": True,
        "image_aspect_ratio": "3:4",
        "video_aspect_ratio": "9:16",
        "video_duration": 5,
        "cooldown_seconds": 0,
        "daily_limit": 10,
        "photo_count_min": 1,
        "photo_count_max": 3,
        "video_count_min": 1,
        "video_count_max": 3,
        "audio_count_min": 1,
        "audio_count_max": 3,
        "audio_mode": "朗读",
        "audio_voice": "",
        # Proactive paid media must use an operator-selected account.
        "proactive_billing_mode": "fixed",
        "proactive_billing_platform": "aiocqhttp",
        "proactive_billing_user_id": "1",
        "photo_prompt": "照片 {subject_desc} 数量{count} 穿搭：{outfit}",
        "video_prompt": "视频 {subject_desc} 数量{count} 穿搭：{outfit}",
        "audio_prompt": "音频 {style} {text} 数量{count}",
    }
    base.update(overrides)
    return base


async def make_media(tmp_path, *, settings=None, aigen=None, outfit=True, with_ref=True):
    """Build a LifeMedia wired to fakes plus a stub runtime."""

    settings = settings or media_settings()
    aigen = aigen if aigen is not None else FakeAigen()
    plan = SimpleNamespace(
        has_outfit=outfit,
        outfit=None,
        date="2026-10-21",
        timeline=(),
        theme="t",
        mood="m",
    )

    async def resolve(_umo):
        return SimpleNamespace(id="alice")

    async def ensure_plan(_umo):
        return SimpleNamespace(id="alice"), plan

    runtime = SimpleNamespace(
        config={"media_settings": settings, "appearance_settings": {"description": "银色长发"}},
        closed=False,
        personas=SimpleNamespace(resolve=resolve),
        _ensure_plan_for_umo=ensure_plan,
        _now=lambda: datetime(2026, 10, 21, 12, 0),
        can_commit=lambda umo, generation: True,
    )

    async def get_runtime():
        return runtime

    owner = SimpleNamespace(
        context=SimpleNamespace(
            get_all_stars=lambda: [
                SimpleNamespace(name="astrbot_plugin_aigen", activated=True, star_cls=aigen)
            ]
        ),
        _get_life_runtime=get_runtime,
    )
    media = LifeMedia(owner, tmp_path)
    if with_ref:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        await media.appearance.upload("alice", "ref.png", data_url)
    return media, aigen


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["photo", "video", "audio"])
async def test_no_proactive_media_draw_submits_zero_aigen_requests(tmp_path, kind):
    media, aigen = await make_media(tmp_path)
    event = FakeEvent({"decision_life_proactive": True})
    result = await media.capture(event, kind=kind, audio_text="你好")
    assert "未抽中" in result
    assert not aigen.image_calls and not aigen.video_calls and not aigen.audio_calls


@pytest.mark.asyncio
async def test_proactive_audio_spends_its_budget_across_calls(tmp_path):
    """The audio range caps the whole proactive run, not a single call."""

    media, aigen = await make_media(tmp_path)
    event = FakeEvent(
        {
            "decision_life_proactive": True,
            "decision_life_generation": 0,
            "decision_life_media_kind": "audio",
            "decision_life_media_min": 1,
            "decision_life_media_max": 2,
            "decision_life_media_used": 0,
        }
    )

    await media.capture(event, kind="audio", audio_text="第一段", count=1)
    await media.capture(event, kind="audio", audio_text="第二段", count=1)
    blocked = await media.capture(event, kind="audio", audio_text="第三段", count=1)

    assert len(aigen.audio_calls) == 2
    assert [c["text"] for c in aigen.audio_calls] == ["第一段", "第二段"]
    assert "上限" in blocked


@pytest.mark.asyncio
async def test_reload_cancel_before_backend_call_refunds_reserved_quota(tmp_path):
    media, aigen = await make_media(tmp_path)
    runtime = await media.owner._get_life_runtime()
    original_call_kwargs = media._call_kwargs

    def close_after_prepare(*args, **kwargs):
        result = original_call_kwargs(*args, **kwargs)
        runtime.closed = True
        return result

    media._call_kwargs = close_after_prepare

    result = await media.capture(FakeEvent(), kind="photo", count=1)

    assert "已取消" in result
    assert aigen.image_calls == []
    assert await media.usage.load() == {"usage": {}}


@pytest.mark.asyncio
async def test_daily_limit_counts_generated_media_units(tmp_path):
    media, aigen = await make_media(tmp_path)

    await media.capture(FakeEvent(), kind="photo", count=3)

    usage = await media.usage.load()
    key = "2026-10-21::aiocqhttp:FriendMessage:1"
    assert usage["usage"][key] == 3
    assert aigen.image_calls[0]["n"] == 3


@pytest.mark.asyncio
async def test_backend_actual_batch_units_override_requested_count(tmp_path):
    class PartialBatchAigen(FakeAigen):
        async def generate_image(self, event=None, n=None, **kwargs):
            self.image_calls.append({"event": event, "n": n, **kwargs})
            event.set_extra("aigen_completed_media_units", 2)
            yield "image-ok"

    media, aigen = await make_media(
        tmp_path,
        settings=media_settings(photo_count_min=3, photo_count_max=3),
        aigen=PartialBatchAigen(),
    )

    await media.capture(FakeEvent(), kind="photo", count=3)

    usage = await media.usage.load()
    assert usage["usage"]["2026-10-21::aiocqhttp:FriendMessage:1"] == 2


@pytest.mark.asyncio
async def test_request_cannot_reserve_more_than_daily_limit(tmp_path):
    media, aigen = await make_media(tmp_path, settings=media_settings(daily_limit=2))

    result = await media.capture(FakeEvent(), kind="photo", count=3)

    assert "仅剩 2 个额度" in result
    assert aigen.image_calls == []
    assert await media.usage.load() == {"usage": {}}


@pytest.mark.asyncio
async def test_partial_video_failure_reconciles_reserved_units(tmp_path):
    class PartialAigen(FakeAigen):
        async def generate_video(self, *args, **kwargs):
            self.video_calls.append({"event": kwargs.get("event")})
            if len(self.video_calls) == 2:
                raise RuntimeError("second segment failed")
            yield "video-ok"

    media, aigen = await make_media(
        tmp_path,
        settings=media_settings(video_count_min=2, video_count_max=2),
        aigen=PartialAigen(),
    )

    result = await media.capture(FakeEvent(), kind="video", count=2)

    assert "部分生成失败" in result
    usage = await media.usage.load()
    assert usage["usage"]["2026-10-21::aiocqhttp:FriendMessage:1"] == 1


@pytest.mark.asyncio
async def test_proactive_rejects_a_second_media_kind(tmp_path):
    """Only the preselected kind may be sent during one proactive delivery."""

    media, aigen = await make_media(tmp_path)
    event = FakeEvent(
        {
            "decision_life_proactive": True,
            "decision_life_generation": 0,
            "decision_life_media_kind": "photo",
            "decision_life_media_min": 1,
            "decision_life_media_max": 1,
            "decision_life_media_used": 0,
        }
    )

    result = await media.capture(event, kind="video", scene="x", count=1)
    assert "只允许发送" in result
    assert not aigen.video_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [1, 3, 10])
async def test_audio_generates_one_clip_per_call(tmp_path, requested):
    """A single call speaks one text, so extra counts must not duplicate it.

    The configured range is a per-run budget the model spends by calling the
    tool again with new content.
    """

    settings = media_settings(audio_count_min=1, audio_count_max=3)
    media, aigen = await make_media(tmp_path, settings=settings)

    result = await media.capture(
        FakeEvent(), kind="audio", audio_text="你好呀", count=requested
    )
    assert len(aigen.audio_calls) == 1
    assert aigen.audio_calls[0]["text"] == "你好呀"
    assert aigen.audio_calls[0]["mode"] == "朗读"
    assert "audio-ok" in result


@pytest.mark.asyncio
async def test_audio_does_not_require_a_reference_image(tmp_path):
    media, aigen = await make_media(tmp_path, with_ref=False)
    result = await media.capture(FakeEvent(), kind="audio", audio_text="喂", count=1)
    assert len(aigen.audio_calls) == 1
    assert "未上传外貌参考图" not in result


@pytest.mark.asyncio
async def test_audio_requires_text(tmp_path):
    media, aigen = await make_media(tmp_path)
    result = await media.capture(FakeEvent(), kind="audio", audio_text="   ", count=1)
    assert "需要提供" in result
    assert not aigen.audio_calls


@pytest.mark.asyncio
async def test_photo_uses_single_batched_call(tmp_path):
    settings = media_settings(photo_count_min=2, photo_count_max=4)
    media, aigen = await make_media(tmp_path, settings=settings)

    await media.capture(FakeEvent(), kind="photo", scene="窗边", subject="other", count=3)
    assert len(aigen.image_calls) == 1
    assert aigen.image_calls[0]["n"] == 3
    assert "别人" in aigen.image_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_photo_count_is_clamped_to_the_configured_range(tmp_path):
    settings = media_settings(photo_count_min=2, photo_count_max=2)
    media, aigen = await make_media(tmp_path, settings=settings)

    await media.capture(FakeEvent(), kind="photo", count=99)
    assert aigen.image_calls[0]["n"] == 2


@pytest.mark.asyncio
async def test_video_issues_one_call_per_item(tmp_path):
    settings = media_settings(video_count_min=2, video_count_max=4)
    media, aigen = await make_media(tmp_path, settings=settings)

    await media.capture(FakeEvent(), kind="video", scene="散步", count=2)
    assert len(aigen.video_calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,switch",
    [("photo", "photo_enable"), ("video", "video_enable"), ("audio", "audio_enable")],
)
async def test_disabled_kind_is_refused_without_calling_aigen(tmp_path, kind, switch):
    settings = media_settings(**{switch: False})
    media, aigen = await make_media(tmp_path, settings=settings)

    result = await media.capture(FakeEvent(), kind=kind, audio_text="hi", count=1)
    assert "已关闭" in result
    assert not (aigen.image_calls or aigen.video_calls or aigen.audio_calls)


@pytest.mark.asyncio
async def test_master_media_switch_is_refused(tmp_path):
    settings = media_settings(enable=False)
    media, aigen = await make_media(tmp_path, settings=settings)

    result = await media.capture(FakeEvent(), kind="photo", count=1)
    assert "已关闭" in result
    assert not aigen.image_calls


@pytest.mark.asyncio
async def test_daily_limit_blocks_generation(tmp_path):
    settings = media_settings(daily_limit=0)
    media, aigen = await make_media(tmp_path, settings=settings)

    result = await media.capture(FakeEvent(), kind="photo", count=1)
    assert "额度已用完" in result
    assert not aigen.image_calls


@pytest.mark.asyncio
async def test_photo_requires_a_reference_image(tmp_path):
    media, aigen = await make_media(tmp_path, with_ref=False)

    result = await media.capture(FakeEvent(), kind="photo", count=1)
    assert "未上传外貌参考图" in result
    assert not aigen.image_calls


@pytest.mark.asyncio
async def test_photo_requires_a_daily_outfit_when_the_fallback_is_off(tmp_path):
    media, aigen = await make_media(tmp_path, outfit=False)
    media.owner._get_life_runtime = _runtime_with_appearance(
        media.owner._get_life_runtime, {"use_reference_outfit": False}
    )

    result = await media.capture(FakeEvent(), kind="photo", count=1)
    assert "今日穿搭尚未生成" in result
    assert not aigen.image_calls


@pytest.mark.asyncio
async def test_reference_outfit_fallback_allows_generation_without_outfit(tmp_path):
    """The shipped default falls back to the reference outfit."""

    media, aigen = await make_media(tmp_path, outfit=False)

    await media.capture(FakeEvent(), kind="photo", count=1)
    assert len(aigen.image_calls) == 1
    assert "参考图" in aigen.image_calls[0]["prompt"]


def _runtime_with_appearance(original, appearance):
    """Wrap the async runtime getter so appearance_settings is overridden."""

    async def getter():
        runtime = await original()
        runtime.config["appearance_settings"] = {
            **runtime.config.get("appearance_settings", {}),
            **appearance,
        }
        return runtime

    return getter

"""Runtime behaviour: auto understanding, caching, injection and GIF safety."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.personal_network._astrbot_stub import install  # noqa: E402

install()

from decision.multimodal.models import MediaRef  # noqa: E402
from decision.multimodal.runtime import MultimodalRuntime  # noqa: E402


class FakeProvider:
    def __init__(self, modalities=("text", "image", "audio"), fail: str = ""):
        self.provider_config = {"modalities": list(modalities), "id": "prov-1"}
        self.calls: list[dict] = []
        self._fail = fail

    def meta(self):
        return SimpleNamespace(id="prov-1", model="model-1", type="openai")

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError(self._fail)
        return SimpleNamespace(completion_text=f"描述{len(self.calls)}")


def make_context(provider):
    return SimpleNamespace(
        get_all_providers=lambda: [provider],
        get_provider_by_id=lambda pid: provider if pid == "prov-1" else None,
    )


class FakeEvent:
    def __init__(self, components, umo="aiocqhttp:GroupMessage:1", msg_id="m1"):
        self.unified_msg_origin = umo
        self.message_obj = SimpleNamespace(message_id=msg_id, message=components)
        self._components = components

    def get_messages(self):
        return list(self._components)

    def get_sender_id(self):
        return "user-1"


class FakeImage:
    type = "image"

    def __init__(self, path):
        self.path = str(path)

    async def convert_to_file_path(self):
        return str(self.path)


def build(tmp_path, provider, **settings):
    base = {
        "mm_enabled": True,
        "mm_auto_enabled": True,
        "mm_auto_image": True,
        "mm_auto_scope": ["aiocqhttp:GroupMessage:1"],
        "mm_auto_scope_blacklist": False,
        "mm_default_model": "prov-1",
        "mm_concurrency": 2,
        "mm_timeout_seconds": 30,
        "mm_retries": 0,
    }
    base.update(settings)
    owner = SimpleNamespace(context=make_context(provider))
    return MultimodalRuntime(owner, data_dir=tmp_path, settings=base)


def png(tmp_path, name="a.png"):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (32, 32), (1, 2, 3)).save(path, "PNG")
    return path


def gif(tmp_path, name="a.gif"):
    from PIL import Image

    path = tmp_path / name
    frames = [Image.new("RGB", (24, 24), (i * 60, 10, 10)) for i in range(4)]
    frames[0].save(path, "GIF", save_all=True, append_images=frames[1:], duration=100)
    return path


# ----------------------------------------------------------------- observing
@pytest.mark.asyncio
async def test_auto_understanding_runs_and_caches(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=True)
    assert len(refs) == 1
    await asyncio.sleep(0.4)
    item = await runtime.cache.get_for_media(refs[0].umo, refs[0].content_key)
    assert item is not None and item.ok
    assert item.summary == "描述1"
    assert provider.calls, "provider must have been called"
    await runtime.terminate()


@pytest.mark.asyncio
async def test_auto_understanding_skipped_when_scope_disallows(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    event = FakeEvent([FakeImage(png(tmp_path))], umo="aiocqhttp:GroupMessage:2")
    refs = await runtime.observe(event, auto_allowed=False)
    assert len(refs) == 1, "media is still indexed for the tools"
    await asyncio.sleep(0.3)
    assert provider.calls == []
    assert await runtime.cache.get(refs[0].content_key) is None
    await runtime.terminate()


@pytest.mark.asyncio
async def test_auto_off_still_indexes_media_for_tool_call(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider, mm_auto_enabled=False)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    assert refs and provider.calls == []
    item, error = await runtime.understand_by_selector(
        refs[0].umo, kind="image", selector="latest", instruction="请描述画面"
    )
    assert error == "" and item is not None and provider.calls
    await runtime.terminate()


@pytest.mark.asyncio
async def test_generic_tool_instruction_reuses_auto_result(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=True)
    await asyncio.sleep(0.3)
    assert len(provider.calls) == 1
    item = await runtime.understand(refs[0], instruction="请描述这张图片的内容")
    assert item.ok and len(provider.calls) == 1
    await runtime.terminate()


@pytest.mark.asyncio
async def test_specific_followup_triggers_incremental_analysis(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=True)
    await asyncio.sleep(0.3)
    assert len(provider.calls) == 1
    item = await runtime.understand(refs[0], instruction="图中有多少个人？")
    assert item.ok and len(provider.calls) == 2
    await runtime.terminate()


@pytest.mark.asyncio
async def test_concurrent_generic_tool_joins_auto_inflight(tmp_path):
    provider = FakeProvider()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_chat(**kwargs):
        provider.calls.append(kwargs)
        started.set()
        await release.wait()
        return SimpleNamespace(completion_text="自动概览")

    provider.text_chat = slow_chat
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    automatic = asyncio.create_task(runtime.understand(refs[0]))
    await started.wait()
    tool = asyncio.create_task(runtime.understand(refs[0], instruction="请描述画面"))
    await asyncio.sleep(0)
    assert len(provider.calls) == 1
    release.set()
    first, second = await asyncio.gather(automatic, tool)
    assert first.summary == second.summary == "自动概览"
    await runtime.terminate()


@pytest.mark.asyncio
async def test_target_isolation_prevents_wrong_region_reuse(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    source = png(tmp_path, "target.png")
    first = MediaRef(
        kind="image", reference=str(source), urn="u::target::0",
        umo="aiocqhttp:GroupMessage:1", extra={"target": "page 1"}
    )
    second = MediaRef(
        kind="image", reference=str(source), urn="u::target::0",
        umo="aiocqhttp:GroupMessage:1", extra={"target": "page 2"}
    )
    await runtime.understand(first)
    await runtime.understand(second)
    assert len(provider.calls) == 2
    await runtime.terminate()


@pytest.mark.asyncio
async def test_file_processing_policy_changes_cache_route_signature(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    before = runtime._route_signature("file")
    runtime.refresh(
        {
            **runtime.settings.values,
            "mm_ocr_lang": "eng",
            "mm_file_chunk_chars": 9000,
            "mm_office_render_timeout_seconds": 120,
        }
    )
    after = runtime._route_signature("file")
    assert before != after
    assert "lang=eng" in after and "chunk=9000" in after
    await runtime.terminate()


def test_local_media_version_changes_cache_identity(tmp_path):
    path = png(tmp_path, "versioned.png")
    ref = MediaRef(kind="image", reference=str(path), urn="u::v::0", umo="u")
    first = ref.content_key
    path.write_bytes(path.read_bytes() + b"changed")
    assert ref.content_key != first


@pytest.mark.asyncio
async def test_disabled_kind_is_not_understood(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider, mm_auto_image=False)
    await runtime.initialize()
    await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=True)
    await asyncio.sleep(0.3)
    assert provider.calls == []
    await runtime.terminate()


@pytest.mark.asyncio
async def test_auto_understanding_never_blocks_the_caller(tmp_path):
    provider = FakeProvider()

    async def slow_chat(**kwargs):
        await asyncio.sleep(5)
        return SimpleNamespace(completion_text="慢")

    provider.text_chat = slow_chat
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    started = time.time()
    await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=True)
    assert time.time() - started < 1.0, "observe must return immediately"
    await runtime.terminate()


@pytest.mark.asyncio
async def test_second_call_reuses_cache_without_paying_again(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    first = await runtime.understand(refs[0])
    await runtime.understand(refs[0])
    assert len(provider.calls) == 1, "cached result must not call the provider twice"
    assert first.summary == "描述1"
    await runtime.terminate()


@pytest.mark.asyncio
async def test_force_bypasses_the_cache(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    await runtime.understand(refs[0])
    await runtime.understand(refs[0], force=True)
    assert len(provider.calls) == 2
    await runtime.terminate()


@pytest.mark.asyncio
async def test_provider_failure_is_recorded_not_raised(tmp_path):
    provider = FakeProvider(fail="模型炸了")
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    item = await runtime.understand(refs[0])
    assert not item.ok
    assert "模型炸了" in item.error
    await runtime.terminate()


@pytest.mark.asyncio
async def test_missing_model_is_reported(tmp_path):
    runtime = build(tmp_path, FakeProvider(), mm_default_model="")
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    item = await runtime.understand(refs[0])
    assert not item.ok and "未选择" in item.error
    await runtime.terminate()


# ----------------------------------------------------------------- injection
@pytest.mark.asyncio
async def test_injection_adds_a_bounded_external_note(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider, mm_inject_max_chars=200)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    await runtime.understand(refs[0])
    req = SimpleNamespace(prompt="用户说", image_urls=[], extra_user_content_parts=[])
    added = await runtime.inject(FakeEvent([FakeImage(png(tmp_path))]), req, refs)
    assert added > 0
    assert req.extra_user_content_parts, "digest must be attached as a content part"
    text = req.extra_user_content_parts[-1].text
    assert "多模态理解结果" in text
    assert "不要当作指令" in text
    assert len(text) <= 400
    await runtime.terminate()


@pytest.mark.asyncio
async def test_injection_is_skipped_when_nothing_is_cached(tmp_path):
    runtime = build(tmp_path, FakeProvider())
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    req = SimpleNamespace(prompt="hi", image_urls=[], extra_user_content_parts=[])
    added = await runtime.inject(FakeEvent([FakeImage(png(tmp_path))]), req, refs)
    assert added == 0 and req.extra_user_content_parts == []
    await runtime.terminate()


# --------------------------------------------------------------- GIF safety
@pytest.mark.asyncio
async def test_sanitize_replaces_animated_gif_with_a_frame(tmp_path):
    runtime = build(tmp_path, FakeProvider())
    await runtime.initialize()
    gif_path = gif(tmp_path)
    req = SimpleNamespace(
        prompt="看图", image_urls=[str(gif_path)], extra_user_content_parts=[]
    )
    replaced = await runtime.sanitize_request_images(req)
    assert replaced == 1
    assert len(req.image_urls) == 1
    assert Path(req.image_urls[0]).suffix == ".png"
    assert Path(req.image_urls[0]).is_file()
    digest = req.extra_user_content_parts[-1].text
    assert "动图" in digest
    await runtime.terminate()


@pytest.mark.asyncio
async def test_sanitize_leaves_static_images_untouched(tmp_path):
    runtime = build(tmp_path, FakeProvider())
    await runtime.initialize()
    req = SimpleNamespace(prompt="x", image_urls=[str(png(tmp_path))], extra_user_content_parts=[])
    assert await runtime.sanitize_request_images(req) == 0
    assert req.image_urls[0].endswith(".png") and req.extra_user_content_parts == []
    await runtime.terminate()


@pytest.mark.asyncio
async def test_sanitize_can_be_disabled(tmp_path):
    runtime = build(tmp_path, FakeProvider(), mm_sanitize_animated_images=False)
    await runtime.initialize()
    req = SimpleNamespace(prompt="x", image_urls=[str(gif(tmp_path))], extra_user_content_parts=[])
    assert await runtime.sanitize_request_images(req) == 0
    assert req.image_urls[0].endswith(".gif")
    await runtime.terminate()


# --------------------------------------------------------------------- tools
@pytest.mark.asyncio
async def test_tool_selector_latest_and_index(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    await runtime.observe(
        FakeEvent([FakeImage(png(tmp_path, "one.png"))], msg_id="m1"), auto_allowed=False
    )
    await runtime.observe(
        FakeEvent([FakeImage(png(tmp_path, "two.png"))], msg_id="m2"), auto_allowed=False
    )
    item, error = await runtime.understand_by_selector("aiocqhttp:GroupMessage:1")
    assert error == "" and item is not None
    assert item.message_id == "m2", "latest means the newest media"

    item, error = await runtime.understand_by_selector(
        "aiocqhttp:GroupMessage:1", selector="2"
    )
    assert error == "" and item.message_id == "m1"
    await runtime.terminate()


@pytest.mark.asyncio
async def test_tool_cannot_reach_another_session(tmp_path):
    runtime = build(tmp_path, FakeProvider())
    await runtime.initialize()
    await runtime.observe(
        FakeEvent([FakeImage(png(tmp_path))], umo="aiocqhttp:GroupMessage:1"),
        auto_allowed=False,
    )
    item, error = await runtime.understand_by_selector("aiocqhttp:GroupMessage:999")
    assert item is None and "没有可用" in error
    await runtime.terminate()


@pytest.mark.asyncio
async def test_tool_reports_out_of_range_selector(tmp_path):
    runtime = build(tmp_path, FakeProvider())
    await runtime.initialize()
    await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    item, error = await runtime.understand_by_selector(
        "aiocqhttp:GroupMessage:1", selector="9"
    )
    assert item is None and "超出范围" in error
    await runtime.terminate()


# ------------------------------------------------------------- rate limiting
@pytest.mark.asyncio
async def test_auto_budget_limits_tasks_per_minute(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider, mm_max_auto_per_minute=1)
    await runtime.initialize()
    for index in range(3):
        await runtime.observe(
            FakeEvent([FakeImage(png(tmp_path, f"p{index}.png"))], msg_id=f"m{index}"),
            auto_allowed=True,
        )
    await asyncio.sleep(0.6)
    assert len(provider.calls) <= 1
    await runtime.terminate()


# ------------------------------------------------------------------ shutdown
@pytest.mark.asyncio
async def test_terminate_cancels_running_tasks(tmp_path):
    async def slow_chat(**kwargs):
        await asyncio.sleep(30)
        return SimpleNamespace(completion_text="never")

    provider = FakeProvider()
    provider.text_chat = slow_chat
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=True)
    await asyncio.sleep(0.2)
    await runtime.terminate()
    assert runtime._tasks == set()

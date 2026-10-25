"""Auto video routing must work through chat-only Gemini gateways."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.personal_network._astrbot_stub import install

install()

from decision.multimodal import adapters, native_video, preprocess  # noqa: E402
from decision.multimodal.models import MediaRef  # noqa: E402
from decision.multimodal.runtime import MultimodalRuntime  # noqa: E402


class RecordingProvider:
    def __init__(self, provider_type, base):
        self.provider_config = {
            "id": "route-provider", "type": provider_type, "api_base": base,
            "key": ["test-key"], "modalities": ["text", "image", "audio"],
        }
        self.calls = []

    def meta(self):
        return SimpleNamespace(id="route-provider", model="test-model")

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(completion_text="红色花朵。\n0-2s: 检查花苞变化")


@pytest.fixture
def video_pipeline(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"bounded video fixture")
    state = {"duration": 5.0, "native_calls": [], "frame_calls": []}

    async def probe(path):
        assert path == video
        return {"duration": state["duration"], "has_audio": False}

    async def extract(path, workdir, stamps, **kwargs):
        from PIL import Image

        assert path == video
        files = []
        for index, _stamp in enumerate(stamps):
            target = workdir / f"frame-{index}.png"
            Image.new("RGB", (16, 16), "red").save(target)
            files.append(str(target))
        state["frame_calls"].append((list(stamps), files))
        return files

    async def native(path, prompt, **kwargs):
        assert path == video
        state["native_calls"].append(kwargs)
        return native_video.NativeResult("官方原生结果", "native", agentic=kwargs["agentic"])

    monkeypatch.setattr(preprocess, "probe_video", probe)
    monkeypatch.setattr(preprocess, "extract_frames", extract)
    monkeypatch.setattr(native_video, "analyze", native)
    return video, state


def runtime_for(tmp_path, provider, strategy="auto"):
    context = SimpleNamespace(get_provider_by_id=lambda ident: provider)
    return MultimodalRuntime(
        SimpleNamespace(context=context), data_dir=tmp_path / "runtime",
        settings={"mm_enabled": True, "mm_default_model": "route-provider",
                  "mm_video_strategy": strategy, "mm_retries": 0,
                  "mm_video_agentic_rounds": 1, "mm_video_agentic_segments": 2},
    )


@pytest.mark.parametrize("base,expected", [
    (None, True),
    ("https://generativelanguage.googleapis.com", True),
    ("https://generativelanguage.googleapis.com:443/v1beta", True),
    ("https://gateway.example", False),
    ("https://generativelanguage.googleapis.com.gateway.example", False),
    ("https://aiplatform.googleapis.com", False),
    ("http://generativelanguage.googleapis.com", False),
    ("https://generativelanguage.googleapis.com/proxy", False),
    ("https://generativelanguage.googleapis.com?secret=test", False),
    ("https://generativelanguage.googleapis.com:invalid", False),
])
def test_auto_native_preference_only_uses_the_official_endpoint(base, expected):
    provider = RecordingProvider("googlegenai_chat_completion", base)
    assert native_video.auto_native_available(provider) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_type", ["openai_chat_completion", "googlegenai_chat_completion"])
@pytest.mark.parametrize("duration", [5.0, 120.0])
async def test_auto_gateway_uses_provider_frames_without_files(
    tmp_path, video_pipeline, provider_type, duration,
):
    video, state = video_pipeline
    state["duration"] = duration
    provider = RecordingProvider(provider_type, "https://gateway.example")
    runtime = runtime_for(tmp_path, provider)
    try:
        await runtime.initialize()
        ref = MediaRef(kind="video", reference=str(video), urn="v", umo="chat:1")
        item = await runtime.understand(ref, instruction="检查红色花朵")
        assert item.ok, item.error
        assert state["native_calls"] == [], "auto must not probe Files/Interactions on a gateway"
        assert item.meta["mode"] == ("frames" if duration == 5 else "agentic")
        assert len(provider.calls) == (1 if duration == 5 else 2)
        assert all(call["image_urls"] for call in provider.calls)
        assert "检查红色花朵" in provider.calls[0]["prompt"]
        assert "没有音轨" in provider.calls[-1]["prompt"]
        assert all(not Path(path).exists() for _, files in state["frame_calls"] for path in files)
    finally:
        await runtime.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy,base", [
    ("auto", "https://generativelanguage.googleapis.com/v1beta"),
    ("native", "https://gateway.example"),
])
async def test_official_auto_and_explicit_native_keep_native_route(
    tmp_path, video_pipeline, strategy, base,
):
    video, state = video_pipeline
    state["duration"] = 120.0
    provider = RecordingProvider("googlegenai_chat_completion", base)
    runtime = runtime_for(tmp_path, provider, strategy)
    try:
        await runtime.initialize()
        item = await runtime.understand(MediaRef(kind="video", reference=str(video), urn="v", umo="chat:1"))
        assert item.ok and item.meta["mode"] == "agentic_native"
        assert len(state["native_calls"]) == 1 and state["native_calls"][0]["agentic"]
        assert not provider.calls and not state["frame_calls"]
    finally:
        await runtime.terminate()


@pytest.mark.asyncio
async def test_auto_route_change_invalidates_video_cache(tmp_path, video_pipeline):
    video, state = video_pipeline
    provider = RecordingProvider("googlegenai_chat_completion", "https://gateway.example")
    runtime = runtime_for(tmp_path, provider)
    ref = MediaRef(kind="video", reference=str(video), urn="v", umo="chat:1")
    try:
        await runtime.initialize()
        generic = await runtime.understand(ref)
        assert generic.meta["mode"] == "frames"
        assert (await runtime.understand(ref)).key == generic.key
        assert len(provider.calls) == 1
        provider.provider_config["api_base"] = "https://generativelanguage.googleapis.com"
        official = await runtime.understand(ref)
        assert official.meta["mode"] == "native"
        assert official.key != generic.key
        assert official.meta["route_signature"] != generic.meta["route_signature"]
        assert len(state["native_calls"]) == 1
    finally:
        await runtime.terminate()


@pytest.mark.asyncio
async def test_gemini_multiframe_stream_uses_only_complete_snapshot():
    calls = []

    async def no_chat(**kwargs):
        raise AssertionError("multi-frame Gemini must not wait for a buffered response")

    async def stream(*, request_max_retries=None, **kwargs):
        calls.append((request_max_retries, kwargs))
        yield SimpleNamespace(is_chunk=True, completion_text="partial", reasoning_content="private")
        yield SimpleNamespace(is_chunk=False, completion_text="final answer 7429")

    provider = SimpleNamespace(
        provider_config={"type": "googlegenai_chat_completion"},
        text_chat=no_chat, text_chat_stream=stream,
    )
    text = await adapters.call_provider(provider, prompt="inspect", images=["a.png", "b.png"], model="m")
    assert text == "final answer 7429"
    assert calls == [(1, {"prompt": "inspect", "image_urls": ["a.png", "b.png"], "model": "m"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
async def test_gemini_stream_never_returns_partial_or_thought_only_output(terminal):
    async def stream(**kwargs):
        yield SimpleNamespace(is_chunk=True, completion_text="unfinished answer")
        if terminal:
            yield SimpleNamespace(is_chunk=False, completion_text="", reasoning_content="private")

    provider = SimpleNamespace(
        provider_config={"type": "googlegenai_chat_completion"},
        text_chat=lambda **kwargs: None, text_chat_stream=stream,
    )
    with pytest.raises(RuntimeError, match="完整"):
        await adapters.call_provider(provider, prompt="inspect", images=["a", "b"])


@pytest.mark.asyncio
async def test_gemini_stream_timeout_closes_generator_without_paid_fallback():
    closed = asyncio.Event()
    calls = []

    async def stream(**kwargs):
        calls.append("stream")
        try:
            yield SimpleNamespace(is_chunk=True, completion_text="partial")
            await asyncio.sleep(30)
        finally:
            closed.set()

    async def chat(**kwargs):
        calls.append("chat")

    provider = SimpleNamespace(
        provider_config={"type": "googlegenai_chat_completion"},
        text_chat=chat, text_chat_stream=stream,
    )
    with pytest.raises(TimeoutError):
        await adapters.call_provider(provider, prompt="inspect", images=["a", "b"], timeout=.02)
    assert closed.is_set() and calls == ["stream"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_type", ["openai_chat_completion", "googlegenai_chat_completion"])
async def test_stream_switch_selects_one_transport_per_request(provider_type):
    calls = []

    async def stream(**kwargs):
        calls.append("stream")
        yield SimpleNamespace(is_chunk=True, completion_text="partial")
        yield SimpleNamespace(is_chunk=False, completion_text="complete")

    async def chat(**kwargs):
        calls.append("chat")
        return SimpleNamespace(completion_text="buffered")

    provider = SimpleNamespace(provider_config={"type": provider_type}, text_chat=chat, text_chat_stream=stream)
    assert await adapters.call_provider(provider, prompt="hello") == "complete"
    assert await adapters.call_provider(provider, prompt="hello", stream=False) == "buffered"
    assert calls == ["stream", "chat"]


@pytest.mark.asyncio
async def test_stream_toggle_hot_refresh_reaches_runtime_calls(tmp_path):
    from PIL import Image

    calls = []

    async def stream(**kwargs):
        calls.append("stream")
        yield SimpleNamespace(is_chunk=False, completion_text="stream answer")

    provider = RecordingProvider("googlegenai_chat_completion", "https://gateway.example")
    provider.text_chat_stream = stream
    runtime = runtime_for(tmp_path, provider)
    image = tmp_path / "switch.png"
    Image.new("RGB", (16, 16), "red").save(image)
    ref = MediaRef(kind="image", reference=str(image), urn="i", umo="chat:1")
    try:
        await runtime.initialize()
        first = await runtime.understand(ref)
        assert first.summary == "stream answer" and calls == ["stream"]
        runtime.refresh({**runtime.settings.values, "mm_stream_enabled": False})
        # A completed understanding remains reusable across transport changes.
        assert (await runtime.understand(ref)).key == first.key and calls == ["stream"]
        second = await runtime.understand(ref, force=True)
        assert "红色花朵" in second.summary and len(provider.calls) == 1
    finally:
        await runtime.terminate()

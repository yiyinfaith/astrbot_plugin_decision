"""Exercise actual HTTP inputs, heartbeat streams and paid-submission counts."""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from aiohttp import web

from tests.personal_network._astrbot_stub import install

install()

from decision.multimodal import adapters, inline_video, preprocess  # noqa: E402
from decision.multimodal.models import MediaRef  # noqa: E402
from decision.multimodal.runtime import MultimodalRuntime  # noqa: E402


async def server(handler):
    app = web.Application()
    app.router.add_post("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


def runtime(tmp_path, provider, **settings):
    return MultimodalRuntime(
        SimpleNamespace(context=SimpleNamespace(get_provider_by_id=lambda _: provider)),
        data_dir=tmp_path / "cache",
        settings={
            "mm_default_model": "p",
            "mm_retries": 3,
            **settings,
        },
    )


def provider(base, kind="openai_chat_completion"):
    return SimpleNamespace(
        provider_config={"type": kind, "api_base": base, "key": ["secret"]},
        meta=lambda: SimpleNamespace(model="gemini-3.8-flash", id="p"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["openai_chat_completion", "googlegenai_chat_completion"])
@pytest.mark.parametrize("stream", [True, False])
async def test_short_clip_sends_complete_bytes_once_with_either_provider(
    tmp_path, monkeypatch, kind, stream
):
    raw = b"\x00\x00\x00\x18ftypisom-complete-video-7429"
    path = tmp_path / "clip.mp4"
    path.write_bytes(raw)
    calls = []

    async def handle(request):
        calls.append((request.path, request.query_string, await request.json()))
        assert request.headers["x-goog-api-key"] == "secret"
        if not stream:
            return web.json_response(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "完整视频7429"}]}, "finishReason": "STOP"}
                    ]
                }
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for data in [
            b": keep-alive\r\n\r\n",
            b": ping\n\n",
            (
                "data: "
                + json.dumps(
                    {"candidates": [{"content": {"parts": [{"text": "private", "thought": True}]}}]}
                )
                + "\n\n"
            ).encode(),
            (
                "data: "
                + json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "完整视频7429"}]},
                                "finishReason": "STOP",
                            }
                        ]
                    }
                )
                + "\n\n"
            ).encode(),
        ]:
            await response.write(data)
            await asyncio.sleep(0.01)
        await response.write_eof()
        return response

    async def probe(_):
        return {"duration": 5, "has_audio": True}

    async def no_frames(*args, **kwargs):
        raise AssertionError("complete short clip must not extract frames/audio")

    monkeypatch.setattr(preprocess, "probe_video", probe)
    monkeypatch.setattr(preprocess, "extract_frames", no_frames)
    monkeypatch.setattr(preprocess, "extract_audio_track", no_frames)
    runner, base = await server(handle)
    p = provider(base + "/v1" if kind.startswith("openai") else base, kind)
    rt = runtime(tmp_path, p, mm_stream_enabled=stream)
    try:
        await rt.initialize()
        ref = MediaRef(kind="video", reference=str(path), umo="chat:1", urn="clip")
        result = await rt.understand(ref)
        assert result.ok and result.summary == "完整视频7429", result.error
        assert result.meta["mode"] == "inline" and result.meta["frames"] == 0
        assert len(calls) == 1
        endpoint = "streamGenerateContent" if stream else "generateContent"
        assert calls[0][0] == f"/v1beta/models/gemini-3.8-flash:{endpoint}"
        parts = calls[0][2]["contents"][0]["parts"]
        assert len(parts) == 2 and base64.b64decode(parts[0]["inline_data"]["data"]) == raw
        assert "private" not in result.summary
        assert (await rt.understand(ref)).key == result.key and len(calls) == 1
    finally:
        await rt.terminate()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "partial", "http524", "empty"])
async def test_inline_uncertain_result_never_reposts_or_falls_back(tmp_path, monkeypatch, outcome):
    calls = []
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"bounded clip")

    async def handle(request):
        calls.append(await request.json())
        if outcome == "http524":
            return web.Response(status=524)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if outcome == "partial":
            await response.write(
                b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}\n\n'
            )
        if outcome == "timeout":
            await asyncio.sleep(1.0)
        await response.write_eof()
        return response

    original = inline_video.generate

    async def short_timeout(*args, **kwargs):
        kwargs["timeout"] = 0.5 if outcome == "timeout" else 5
        return await original(*args, **kwargs)

    async def probe(_):
        return {"duration": 5}

    monkeypatch.setattr(preprocess, "probe_video", probe)
    monkeypatch.setattr(inline_video, "generate", short_timeout)
    runner, base = await server(handle)
    rt = runtime(tmp_path, provider(base))
    try:
        await rt.initialize()
        result = await rt.understand(
            MediaRef(kind="video", reference=str(path), umo="chat:1", urn="clip")
        )
        assert not result.ok and result.summary == ""
        assert "未自动重新提交" in result.error
        assert len(calls) == 1, "uncertain upstream results must not generate another paid POST"
    finally:
        await rt.terminate()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_later_seeking_failure_does_not_replay_paid_overview(tmp_path, monkeypatch):
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise TimeoutError("focus timed out")
        return SimpleNamespace(completion_text="0-2s: inspect")

    p = provider("https://gateway.example")
    p.text_chat = chat
    path = tmp_path / "long.mp4"
    path.write_bytes(b"long")

    async def probe(_):
        return {"duration": 120}

    async def extract(*args, **kwargs):
        return ["frame.png"]

    monkeypatch.setattr(preprocess, "probe_video", probe)
    monkeypatch.setattr(preprocess, "extract_frames", extract)
    rt = runtime(tmp_path, p, mm_video_agentic_rounds=1)
    try:
        await rt.initialize()
        result = await rt.understand(
            MediaRef(kind="video", reference=str(path), umo="chat:1", urn="long")
        )
        assert not result.ok and len(calls) == 2
        assert "未自动重新提交" in result.error
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_retry_still_recovers_before_any_submission(tmp_path, monkeypatch):
    p = provider("https://gateway.example")
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(completion_text="recovered")

    p.text_chat = chat
    rt = runtime(tmp_path, p)
    attempts = []

    async def run(*args):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("local preprocess failure")
        return await adapters.call_provider(p, prompt="read"), {}

    monkeypatch.setattr(rt, "_run_kind", run)
    try:
        await rt.initialize()
        result = await rt.understand(
            MediaRef(kind="image", reference="unused", umo="chat:1", urn="image")
        )
        assert result.ok and result.summary == "recovered"
        assert len(attempts) == 2 and len(calls) == 1
    finally:
        await rt.terminate()


@pytest.mark.parametrize(
    "config,model",
    [
        ({"type": "openai_chat_completion", "key": ["secret"]}, "gemini-3.8-flash"),
        (
            {
                "type": "openai_chat_completion",
                "api_base": "https://example.com/v1",
                "key": ["secret"],
            },
            "other-model",
        ),
        (
            {
                "type": "googlegenai_chat_completion",
                "api_base": "https://user:secret@example.com",
                "key": ["secret"],
            },
            "gemini-x",
        ),
    ],
)
def test_inline_credentials_do_not_leak_arbitrary_keys(config, model):
    assert inline_video.inline_credentials(SimpleNamespace(provider_config=config), model) == (
        "",
        "",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("duration,budget", [(0, 100000), (120, 100000), (5, 1)])
async def test_unknown_long_and_oversized_clips_never_go_inline(
    tmp_path, monkeypatch, duration, budget
):
    calls = []
    p = provider("https://generativelanguage.googleapis.com", "googlegenai_chat_completion")

    async def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(completion_text="0-2s: focus")

    p.text_chat = chat
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"bounded video")

    async def probe(_):
        return {"duration": duration}

    async def frames(*args, **kwargs):
        return ["frame.png"]

    async def forbidden(*args, **kwargs):
        raise AssertionError("must not submit complete video")

    monkeypatch.setattr(preprocess, "probe_video", probe)
    monkeypatch.setattr(preprocess, "extract_frames", frames)
    monkeypatch.setattr(inline_video, "generate", forbidden)
    monkeypatch.setattr(inline_video, "MAX_REQUEST_BYTES", budget)
    rt = runtime(tmp_path, p, mm_video_agentic_rounds=1)
    try:
        await rt.initialize()
        result = await rt.understand(
            MediaRef(kind="video", reference=str(path), umo="chat:1", urn="clip")
        )
        assert result.ok, result.error
        assert result.meta["mode"] == ("agentic" if duration == 120 else "frames")
        assert len(calls) == (2 if duration == 120 else 1)
    finally:
        await rt.terminate()

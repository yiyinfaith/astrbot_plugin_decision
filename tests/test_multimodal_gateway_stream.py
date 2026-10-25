import asyncio
import base64

import pytest
from aiohttp import web

from tests.personal_network._astrbot_stub import install

install()

from decision.multimodal import preprocess  # noqa: E402
from decision.multimodal.models import MediaRef  # noqa: E402
from decision.multimodal.openai_transport import credentials  # noqa: E402
from tests.test_multimodal_inline_video import provider, runtime, server  # noqa: E402


@pytest.mark.asyncio
async def test_openai_compat_preserves_complete_video_and_excludes_thoughts(tmp_path, monkeypatch):
    calls = []
    path = tmp_path / "signed-download.bin"
    raw = b"\x00\x00\x00\x18ftypisom-complete original clip 7429"
    path.write_bytes(raw)

    async def handle(request):
        calls.append((request.path, await request.json()))
        assert request.headers["Authorization"] == "Bearer secret"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(
            b': keep-alive\n\ndata: {"choices":[{"delta":{"reasoning_content":"private"}}]}\n\n'
        )
        await asyncio.sleep(0.01)
        await response.write(
            b'data: {"choices":[{"delta":{"content":"answer7429"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        )
        await response.write_eof()
        return response

    async def probe(_):
        return {"duration": 5}

    monkeypatch.setattr(preprocess, "probe_video", probe)
    runner, base = await server(handle)
    rt = runtime(
        tmp_path, provider(base, "googlegenai_chat_completion"), mm_gateway_openai_stream=True
    )
    try:
        await rt.initialize()
        item = await rt.understand(
            MediaRef(kind="video", reference=str(path), umo="chat:1", urn="clip")
        )
        assert item.ok and item.summary == "answer7429", item.error
        assert len(calls) == 1 and calls[0][0] == "/v1/chat/completions"
        media = calls[0][1]["messages"][0]["content"][-1]
        assert media["type"] == "video_url" and isinstance(media["video_url"], str)
        assert media["video_url"].startswith("data:video/mp4;base64,")
        assert base64.b64decode(media["video_url"].split(",", 1)[1]) == raw
        assert item.meta["mode"] == "inline"
    finally:
        await rt.terminate()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_openai_compat_partial_failure_has_one_post_and_no_native_fallback(tmp_path):
    calls = []

    async def handle(request):
        calls.append(await request.json())
        return web.Response(
            text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            content_type="text/event-stream",
        )

    runner, base = await server(handle)
    p = provider(base, "googlegenai_chat_completion")

    async def forbidden(**kwargs):
        raise AssertionError("native fallback would double bill")

    p.text_chat = forbidden
    path = tmp_path / "image.png"
    from PIL import Image

    Image.new("RGB", (16, 16), "red").save(path)
    rt = runtime(tmp_path, p, mm_gateway_openai_stream=True)
    try:
        await rt.initialize()
        item = await rt.understand(
            MediaRef(kind="image", reference=str(path), umo="chat:1", urn="i")
        )
        assert not item.ok and len(calls) == 1
        assert "未自动重新提交" in item.error
    finally:
        await rt.terminate()
        await runner.cleanup()


def test_gateway_compat_never_redirects_official_google_credentials():
    assert credentials(
        provider("https://generativelanguage.googleapis.com", "googlegenai_chat_completion"),
        "gemini-3.8-flash",
    ) == ("", "")

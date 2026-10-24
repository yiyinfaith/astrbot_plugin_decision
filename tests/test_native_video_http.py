"""Exercise Gemini's REST state/answer boundary over an actual local socket."""
from __future__ import annotations

import pytest
from aiohttp import web

from tests.personal_network._astrbot_stub import install

install()

from decision.multimodal.native_video import agentic_generate  # noqa: E402


@pytest.mark.parametrize("status", ["completed", "incomplete", "requires_action", "success", "failed"])
async def test_only_completed_interactions_publish_a_final_answer(status):
    calls = {"create": 0, "cancel": 0}

    async def create(request):
        assert request.headers["x-goog-api-key"] == "private-test-key"
        body = await request.json()
        assert body["input"][0]["processing"] == "agentic"
        calls["create"] += 1
        return web.json_response({"id": "audit", "status": status, "steps": [
            {"type": "user_input", "content": [{"type": "text", "text": "USER PRIVATE"}]},
            {"type": "thought", "content": [{"type": "text", "text": "THOUGHT PRIVATE"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "FINAL ONLY"}]},
        ]})

    async def cancel(request):
        calls["cancel"] += 1
        return web.json_response({"status": "cancelled"})

    app = web.Application()
    app.router.add_post("/v1beta/interactions", create)
    app.router.add_post("/v1beta/interactions/audit/cancel", cancel)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        kwargs = dict(api_key="private-test-key", base=f"http://127.0.0.1:{port}", model="test", file_uri="files/test", prompt="describe", timeout=2)
        if status == "completed":
            assert await agentic_generate(**kwargs) == "FINAL ONLY"
        else:
            with pytest.raises(RuntimeError):
                await agentic_generate(**kwargs)
        assert calls == {"create": 1, "cancel": int(status != "completed")}
    finally:
        await runner.cleanup()

"""A downloaded playlist must not bypass the HTTP SSRF guard via FFmpeg."""
import shutil

import pytest
from aiohttp import web

from tests.personal_network._astrbot_stub import install

install()

from decision.multimodal.preprocess import run_command  # noqa: E402


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="actual ffprobe is required")
async def test_decoder_never_fetches_secondary_playlist_urls(tmp_path):
    received = []

    async def internal(request):
        received.append(request.path)
        return web.Response(body=b"private")

    app = web.Application()
    app.router.add_get("/secret.ts", internal)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    attachment = tmp_path / "downloaded.m3u8"
    attachment.write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:5\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:5,\nhttp://127.0.0.1:{port}/secret.ts\n#EXT-X-ENDLIST\n")
    try:
        code, _, error = await run_command(["ffprobe", "-v", "error", "-show_format", str(attachment)], timeout=5)
        assert code != 0 and b"whitelist" in error.lower()
        assert received == []
    finally:
        await runner.cleanup()

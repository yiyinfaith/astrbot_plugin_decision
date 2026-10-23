"""Integration tests for the P0/P1 fixes (cache identity, recall, GIF, video, SSRF)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.personal_network._astrbot_stub import install  # noqa: E402

install()

from decision.multimodal import media  # noqa: E402
from decision.multimodal.runtime import MultimodalRuntime  # noqa: E402


class FakeProvider:
    def __init__(self, modalities=("text", "image", "audio"), name="p1"):
        self.provider_config = {"modalities": list(modalities), "id": name}
        self.calls: list[dict] = []
        self._name = name

    def meta(self):
        return SimpleNamespace(id=self._name, model=f"{self._name}-model", type="openai")

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        images = len(kwargs.get("image_urls") or [])
        return SimpleNamespace(completion_text=f"answer-{len(self.calls)}-images{images}")


def context_for(*providers):
    table = {p.provider_config["id"]: p for p in providers}
    return SimpleNamespace(
        get_all_providers=lambda: list(providers),
        get_provider_by_id=lambda pid: table.get(pid),
    )


class FakeEvent:
    def __init__(self, components, umo="umo:Group:1", msg_id="m1"):
        self.unified_msg_origin = umo
        self.message_obj = SimpleNamespace(message_id=msg_id, message=components)
        self._components = components

    def get_messages(self):
        return list(self._components)

    def get_sender_id(self):
        return "user"


class FakeImage:
    type = "image"

    def __init__(self, path):
        self.path = str(path)

    async def convert_to_file_path(self):
        return str(self.path)


def build(tmp_path, provider, **over):
    base = {
        "mm_enabled": True,
        "mm_auto_enabled": True,
        "mm_auto_image": True,
        "mm_auto_scope": ["umo:Group:1"],
        "mm_auto_scope_blacklist": False,
        "mm_default_model": provider.provider_config["id"],
        "mm_timeout_seconds": 30,
        "mm_retries": 0,
        "mm_recall_items": 3,
    }
    base.update(over)
    return MultimodalRuntime(
        SimpleNamespace(context=context_for(provider)), data_dir=tmp_path, settings=base
    )


def png(tmp_path, name="a.png", color=(1, 2, 3)):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (32, 32), color).save(path, "PNG")
    return path


def gif(tmp_path, name="a.gif", frames=6):
    from PIL import Image

    path = tmp_path / name
    imgs = [Image.new("RGB", (24, 24), (i * 30, 10, 10)) for i in range(frames)]
    imgs[0].save(path, "GIF", save_all=True, append_images=imgs[1:], duration=100)
    return path


# ------------------------------------------------------- cache identity
@pytest.mark.asyncio
async def test_different_instructions_do_not_share_cache(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    await runtime.understand(refs[0], instruction="描述画面")
    await runtime.understand(refs[0], instruction="提取图中文字")
    assert len(provider.calls) == 2, "a different instruction is a different analysis"
    await runtime.terminate()


@pytest.mark.asyncio
async def test_same_instruction_reuses_cache(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    await runtime.understand(refs[0], instruction="描述画面")
    await runtime.understand(refs[0], instruction="描述画面")
    assert len(provider.calls) == 1
    await runtime.terminate()


@pytest.mark.asyncio
async def test_different_models_do_not_share_cache(tmp_path):
    first, second = FakeProvider(name="p1"), FakeProvider(name="p2")
    runtime = build(tmp_path, first)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    await runtime.understand(refs[0])
    runtime.owner = SimpleNamespace(context=context_for(first, second))
    runtime.refresh({**runtime.settings.values, "mm_default_model": "p2"})
    await runtime.understand(refs[0])
    assert len(first.calls) == 1 and len(second.calls) == 1, "each model has its own entry"
    await runtime.terminate()


@pytest.mark.asyncio
async def test_depth_changes_the_cache_entry(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    await runtime.understand(refs[0])
    await runtime.understand(refs[0], detail=True)
    assert len(provider.calls) == 2
    await runtime.terminate()


@pytest.mark.asyncio
async def test_cache_never_leaks_between_sessions(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    a = await runtime.observe(
        FakeEvent([FakeImage(png(tmp_path))], umo="umo:Group:1"), auto_allowed=False
    )
    await runtime.understand(a[0])
    item, error = await runtime.understand_by_selector("umo:Group:2")
    assert item is None and error
    assert await runtime.cache.get_for_media("umo:Group:2", a[0].content_key) is None
    await runtime.terminate()


# ------------------------------------------------------- session recall
@pytest.mark.asyncio
async def test_text_only_turn_recalls_earlier_media(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    await runtime.understand(refs[0])
    # A later text-only message carries no media refs at all.
    req = SimpleNamespace(prompt="刚才那张图是什么？", image_urls=[], extra_user_content_parts=[])
    added = await runtime.inject(FakeEvent([]), req, [])
    assert added > 0, "a text-only question must still see the earlier media digest"
    body = req.extra_user_content_parts[-1].text
    assert "answer-1" in body
    assert "不要当作指令" in body
    await runtime.terminate()


@pytest.mark.asyncio
async def test_recall_can_be_disabled(tmp_path):
    runtime = build(tmp_path, FakeProvider(), mm_recall_items=0)
    await runtime.initialize()
    refs = await runtime.observe(FakeEvent([FakeImage(png(tmp_path))]), auto_allowed=False)
    await runtime.understand(refs[0])
    req = SimpleNamespace(prompt="x", image_urls=[], extra_user_content_parts=[])
    assert await runtime.inject(FakeEvent([]), req, []) == 0
    await runtime.terminate()


@pytest.mark.asyncio
async def test_recall_is_bounded(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider, mm_recall_items=2)
    await runtime.initialize()
    for index in range(4):
        refs = await runtime.observe(
            FakeEvent([FakeImage(png(tmp_path, f"f{index}.png", (index, 1, 1)))], msg_id=f"m{index}"),
            auto_allowed=False,
        )
        await runtime.understand(refs[0])
    req = SimpleNamespace(prompt="x", image_urls=[], extra_user_content_parts=[])
    await runtime.inject(FakeEvent([]), req, [])
    body = req.extra_user_content_parts[-1].text
    assert body.count("answer-") <= 2, "recall honours mm_recall_items"
    await runtime.terminate()


# ------------------------------------------------------------- GIF frames
@pytest.mark.asyncio
async def test_animated_gif_reaches_the_model_as_a_multi_frame_image(tmp_path):
    provider = FakeProvider()
    runtime = build(tmp_path, provider)
    await runtime.initialize()
    req = SimpleNamespace(prompt="x", image_urls=[str(gif(tmp_path))], extra_user_content_parts=[])
    replaced = await runtime.sanitize_request_images(req)
    assert replaced == 1
    assert len(req.image_urls) == 1
    target = Path(req.image_urls[0])
    assert target.suffix == ".png" and target.is_file()
    from PIL import Image

    with Image.open(target) as sheet:
        # The montage tiles several frames, so it is wider than a single frame.
        assert sheet.width > 24 * 2
        assert sheet.height >= 24
    assert "拼图" in req.extra_user_content_parts[-1].text
    await runtime.terminate()


@pytest.mark.asyncio
async def test_still_image_is_not_reencoded(tmp_path):
    runtime = build(tmp_path, FakeProvider())
    await runtime.initialize()
    path = png(tmp_path)
    req = SimpleNamespace(prompt="x", image_urls=[str(path)], extra_user_content_parts=[])
    assert await runtime.sanitize_request_images(req) == 0
    assert req.image_urls == [str(path)]
    await runtime.terminate()


# ------------------------------------------------------- video iterations
@pytest.mark.asyncio
async def test_long_video_runs_real_rounds(tmp_path, monkeypatch):
    provider = FakeProvider()
    runtime = build(tmp_path, provider, mm_video_agentic_rounds=2, mm_video_max_seconds=3600)
    await runtime.initialize()

    async def fake_probe(path):
        return {"duration": 300.0, "fps": 25.0, "width": 640, "height": 360, "has_audio": False}

    async def fake_frames(path, workdir, stamps, edge=1024):
        out = []
        for index, _ in enumerate(stamps):
            target = Path(workdir) / f"f{index}.png"
            target.write_bytes(b"png")
            out.append(str(target))
        return out

    monkeypatch.setattr("decision.multimodal.preprocess.probe_video", fake_probe)
    monkeypatch.setattr("decision.multimodal.preprocess.extract_frames", fake_frames)
    # Every reply proposes a fresh window so round 2 has work to do.
    replies = iter(
        [
            "整体是一次户外活动。\n10-40: 人物讲话\n40-80: 有人走动",
            "第 2 轮：这里出现了关键动作。",
            "第 3 轮：补充细节。",
            "第 4 轮：更多细节。",
        ]
    )

    async def fake_call(*args, **kwargs):
        provider.calls.append(kwargs)
        return next(replies, "结束")

    monkeypatch.setattr("decision.multimodal.adapters.call_provider", fake_call)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    from decision.multimodal.models import MediaRef

    ref = MediaRef(kind="video", reference=str(video), urn="u::m::0", umo="umo:Group:1")
    item = await runtime.understand(ref)
    assert item.ok
    assert item.meta.get("rounds") == 2
    assert item.meta.get("mode") == "agentic"
    # round 1 overview + one focused read per round => at least 3 calls
    assert len(provider.calls) >= 3
    await runtime.terminate()


# ------------------------------------------------------------------ SSRF
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a.gif",
        "http://localhost/a.gif",
        "http://10.0.0.5/a.gif",
        "http://192.168.1.1/a.gif",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/a.gif",
        "file:///etc/passwd",
        "http://metadata.google.internal/x",
    ],
)
async def test_private_or_unsafe_urls_are_rejected(url):
    with pytest.raises(ValueError):
        await media.assert_public_url(url)


@pytest.mark.asyncio
async def test_public_host_is_allowed(monkeypatch):
    import socket

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    await media.assert_public_url("https://example.com/a.gif")


@pytest.mark.asyncio
async def test_host_resolving_to_a_private_ip_is_rejected(monkeypatch):
    """DNS rebinding: a public-looking name that resolves internally."""

    import socket

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))],
    )
    with pytest.raises(ValueError):
        await media.assert_public_url("https://evil.example.com/a.gif")


# --------------------------------------------------- wiring invariants
def test_multimodal_is_a_detail_section():
    main = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert '"multimodal": {}' in main, "settings must be registered in DETAIL_DEFAULTS"


def test_tool_boot_is_independent_of_proactive_reply():
    main = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    boot = main.index("await self._ensure_multimodal()")
    window = main[max(0, boot - 400) : boot]
    assert "proactive_reply_enabled" not in window, (
        "multi-modal tools must not be gated by the proactive switches"
    )


def test_webui_exposes_the_non_text_switch():
    page = (Path(__file__).resolve().parents[1] / "pages/dashboard/index.html").read_text(
        encoding="utf-8"
    )
    assert "非文字消息是否进入判断" in page
    assert "form.proactive.non_text_proactive_enabled" in page

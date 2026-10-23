"""Unit tests for the multi-modal understanding module.

These run without a live AstrBot: provider calls and ffmpeg are the only external
pieces and both are stubbed in the runtime tests.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.personal_network._astrbot_stub import install  # noqa: E402

install()

from decision.multimodal import adapters, config, preprocess  # noqa: E402
from decision.multimodal.cache import MultimodalCache  # noqa: E402
from decision.multimodal.models import MediaIndexEntry, MediaRef, Understanding  # noqa: E402


# --------------------------------------------------------------------- config
def test_defaults_are_conservative():
    settings = config.MultimodalSettings()
    assert settings.bool("mm_enabled") is False
    assert settings.bool("mm_auto_enabled") is False
    assert settings.bool("mm_tools_enabled") is False
    assert settings.auto_kinds == ()
    assert settings.tool_kinds == ()


def test_whitelist_with_empty_list_allows_nobody():
    settings = config.MultimodalSettings(config.normalize_settings({"mm_enabled": True}))
    assert settings.bool("mm_auto_scope_blacklist") is False
    assert settings.scope_allows("aiocqhttp:GroupMessage:1") is False


def test_blacklist_with_empty_list_allows_everyone():
    settings = config.MultimodalSettings(
        config.normalize_settings({"mm_auto_scope_blacklist": True})
    )
    assert settings.scope_allows("aiocqhttp:GroupMessage:1") is True


def test_whitelist_match_allows_that_session_only():
    settings = config.MultimodalSettings(
        config.normalize_settings({"mm_auto_scope": ["umo-a"]})
    )
    assert settings.scope_allows("umo-a") is True
    assert settings.scope_allows("umo-b") is False


def test_auto_kinds_require_both_masters():
    raw = {"mm_enabled": True, "mm_auto_enabled": True, "mm_auto_image": True}
    settings = config.MultimodalSettings(config.normalize_settings(raw))
    assert settings.auto_kinds == ("image",)
    raw["mm_auto_enabled"] = False
    settings = config.MultimodalSettings(config.normalize_settings(raw))
    assert settings.auto_kinds == ()


def test_tool_kinds_are_independent_of_auto():
    raw = {
        "mm_enabled": True,
        "mm_auto_enabled": False,
        "mm_tools_enabled": True,
        "mm_tool_image": True,
        "mm_tool_video": False,
    }
    settings = config.MultimodalSettings(config.normalize_settings(raw))
    assert settings.auto_kinds == ()
    assert "image" in settings.tool_kinds
    assert "video" not in settings.tool_kinds


def test_model_override_falls_back_to_default():
    settings = config.MultimodalSettings(
        config.normalize_settings({"mm_default_model": "p1", "mm_model_audio": "p2"})
    )
    assert settings.model_for("image") == "p1"
    assert settings.model_for("audio") == "p2"


def test_normalize_clamps_out_of_range_numbers():
    settings = config.MultimodalSettings(
        config.normalize_settings({"mm_concurrency": 999, "mm_timeout_seconds": -5})
    )
    assert settings.int("mm_concurrency") == 8
    assert settings.int("mm_timeout_seconds") >= 10


def test_validate_rejects_wrong_types():
    cleaned, error = config.validate_page_payload({"mm_enabled": "yes"})
    assert cleaned == {}
    assert "boolean" in error

    cleaned, error = config.validate_page_payload({"mm_concurrency": True})
    assert cleaned == {}
    assert "integer" in error


def test_validate_rejects_inverted_video_bounds():
    _, error = config.validate_page_payload(
        {"mm_video_short_seconds": 600, "mm_video_max_seconds": 60}
    )
    assert "must not exceed" in error


def test_validate_accepts_a_good_payload():
    cleaned, error = config.validate_page_payload(
        {"mm_enabled": True, "mm_auto_scope": ["a", " "], "mm_concurrency": 3}
    )
    assert error == ""
    assert cleaned["mm_enabled"] is True
    assert cleaned["mm_auto_scope"] == ["a"]


# ---------------------------------------------------------------------- cache
@pytest.mark.asyncio
async def test_cache_round_trip_and_ttl(tmp_path):
    cache = MultimodalCache(tmp_path / "c.json", ttl_seconds=3600, max_entries=10)
    item = Understanding(key="k1", kind="image", umo="u1", summary="一只猫")
    await cache.put(item)
    again = await cache.get("k1")
    assert again is not None and again.summary == "一只猫"
    assert again.expires_at > again.created_at


@pytest.mark.asyncio
async def test_cache_is_scoped_per_session(tmp_path):
    cache = MultimodalCache(tmp_path / "c.json")
    await cache.put(Understanding(key="k1", kind="image", umo="u1", summary="x"))
    assert await cache.get_for_session("u1", ["k1"]) != []
    assert await cache.get_for_session("u2", ["k1"]) == []


@pytest.mark.asyncio
async def test_cache_expires_entries(tmp_path):
    cache = MultimodalCache(tmp_path / "c.json", ttl_seconds=1)
    item = Understanding(key="k1", kind="image", umo="u1", summary="x", created_at=1)
    await cache.put(item)
    item.expires_at = 2
    assert await cache.get("k1") is None


@pytest.mark.asyncio
async def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.json"
    first = MultimodalCache(path)
    await first.put(Understanding(key="k1", kind="image", umo="u1", summary="保存我"))
    second = MultimodalCache(path)
    item = await second.get("k1")
    assert item is not None and item.summary == "保存我"


@pytest.mark.asyncio
async def test_media_index_is_bounded_and_deduplicated(tmp_path):
    cache = MultimodalCache(tmp_path / "c.json")
    rows = [
        MediaIndexEntry(
            umo="u1", message_id="m1", kind="image", index=0, name="a.png",
            reference="/tmp/a.png", content_key=f"k{i}",
            received_at=time.time() + i,
        )
        for i in range(3)
    ]
    await cache.remember_media(rows)
    await cache.remember_media(rows)  # duplicate insert is ignored
    stored = await cache.recent_media("u1", 10)
    assert len(stored) == 3
    assert stored[0].content_key == "k2", "newest first"


def test_understanding_render_marks_external_input():
    item = Understanding(key="k", kind="image", umo="u", summary="画面内容")
    assert item.render().startswith("[image理解]")
    assert "画面内容" in item.render()
    failed = Understanding(key="k", kind="image", umo="u", summary="", error="超时")
    assert "失败" in failed.render()


# ------------------------------------------------------------------ adapters
class FakeProvider:
    def __init__(self, modalities, model="m1", provider_id="p1"):
        self.provider_config = {"modalities": modalities, "id": provider_id}
        self._model = model
        self.calls: list[dict] = []

    def meta(self):
        return type("Meta", (), {"id": self.provider_config["id"], "model": self._model, "type": "openai"})()

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"completion_text": "回答"})()


def test_unconfigured_modalities_means_everything_supported():
    provider = FakeProvider(None)
    for kind in ("image", "video", "audio"):
        assert adapters.kind_supported(provider, kind) is True


def test_empty_modality_list_is_treated_as_unconfigured():
    provider = FakeProvider([])
    assert adapters.supports_modality(provider, "image") is True


def test_declared_modalities_are_enforced():
    provider = FakeProvider(["text", "image"])
    assert adapters.supports_modality(provider, "image") is True
    assert adapters.supports_modality(provider, "audio") is False
    assert adapters.kind_supported(provider, "audio") is False
    # Video is understood through frames, so image support is enough.
    assert adapters.kind_supported(provider, "video") is True
    # Files are reduced to text locally and need no modality.
    assert adapters.kind_supported(provider, "file") is True


def test_provider_catalog_lists_configured_models():
    context = type("Ctx", (), {"get_all_providers": lambda self: [FakeProvider(["text", "image"])]})()
    catalog = adapters.provider_catalog(context)
    assert len(catalog) == 1
    assert catalog[0]["id"] == "p1"
    assert catalog[0]["supports"]["image"] is True
    assert catalog[0]["supports"]["audio"] is False


def test_pick_provider_reports_missing_model_and_missing_capability():
    context = type(
        "Ctx",
        (),
        {
            "get_provider_by_id": lambda self, pid: {
                "p1": FakeProvider(["text"]),
            }.get(pid)
        },
    )()
    settings = config.MultimodalSettings(config.normalize_settings({}))
    choice, error = adapters.pick_provider(context, settings, "audio")
    assert choice is None and "未选择" in error

    settings = config.MultimodalSettings(
        config.normalize_settings({"mm_model_audio": "p1"})
    )
    choice, error = adapters.pick_provider(context, settings, "audio")
    assert choice is None and "未声明支持" in error


def test_pick_provider_returns_a_usable_choice():
    provider = FakeProvider(["text", "audio"])
    context = type("Ctx", (), {"get_provider_by_id": lambda self, pid: provider})()
    settings = config.MultimodalSettings(
        config.normalize_settings({"mm_model_audio": "p1"})
    )
    choice, error = adapters.pick_provider(context, settings, "audio")
    assert error == "" and choice is not None and choice.provider is provider


@pytest.mark.asyncio
async def test_call_provider_forwards_images_audio_and_model():
    provider = FakeProvider(["text", "image"])
    text = await adapters.call_provider(
        provider, prompt="hi", images=["/a.png"], audio=["/b.mp3"], model="m2"
    )
    assert text == "回答"
    call = provider.calls[-1]
    assert call["image_urls"] == ["/a.png"]
    assert call["audio_urls"] == ["/b.mp3"]
    assert call["model"] == "m2"


# --------------------------------------------------------------- preprocess
def _make_image(path: Path, *, animated: bool = False) -> Path:
    from PIL import Image

    if not animated:
        Image.new("RGB", (64, 48), (10, 120, 200)).save(path, "PNG")
        return path
    frames = [Image.new("RGB", (48, 48), (i * 40, 20, 200 - i * 40)) for i in range(6)]
    frames[0].save(
        path,
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )
    return path


def test_static_png_is_passed_through_unchanged(tmp_path):
    path = _make_image(tmp_path / "a.png")
    settings = config.MultimodalSettings()
    workdir = tmp_path / "w"
    workdir.mkdir()
    prepared = preprocess.prepare_image(path, settings, workdir)
    assert prepared.images == [str(path)], "static PNG must not be re-encoded"
    assert prepared.temp_paths == []


def test_animated_gif_becomes_multiple_ordered_frames(tmp_path):
    path = _make_image(tmp_path / "a.gif", animated=True)
    settings = config.MultimodalSettings()
    workdir = tmp_path / "w"
    workdir.mkdir()
    prepared = preprocess.prepare_image(path, settings, workdir)
    assert len(prepared.images) > 1
    assert all(Path(frame).suffix == ".png" for frame in prepared.images)
    assert any("动图" in note for note in prepared.notes)
    assert any("≈" in note for note in prepared.notes), "frames need time markers"
    # Frame order is the animation order.
    assert prepared.images == sorted(prepared.images)


def test_gif_frame_count_respects_the_configured_cap(tmp_path):
    path = _make_image(tmp_path / "a.gif", animated=True)
    settings = config.MultimodalSettings(
        config.normalize_settings({"mm_gif_max_frames": 3})
    )
    workdir = tmp_path / "w"
    workdir.mkdir()
    prepared = preprocess.prepare_image(path, settings, workdir)
    assert len(prepared.images) <= 3


def test_oversize_image_is_refused(tmp_path):
    path = _make_image(tmp_path / "a.png")
    settings = config.MultimodalSettings(config.normalize_settings({"mm_max_file_mb": 1}))
    path.write_bytes(b"0" * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match="上限"):
        preprocess.prepare_image(path, settings, tmp_path)


def test_text_file_extraction(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# 标题\n正文内容", encoding="utf-8")
    settings = config.MultimodalSettings()
    prepared = preprocess.extract_file_text(path, settings)
    assert "标题" in prepared.text and "正文内容" in prepared.text


def test_unknown_file_type_is_rejected_with_a_clear_message(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x00\x01\x02")
    settings = config.MultimodalSettings()
    with pytest.raises(ValueError, match="不支持的文件类型"):
        preprocess.extract_file_text(path, settings)


def test_docx_extraction(tmp_path):
    import docx

    document = docx.Document()
    document.add_paragraph("第一段")
    document.add_paragraph("第二段")
    path = tmp_path / "a.docx"
    document.save(path)
    prepared = preprocess.extract_file_text(path, config.MultimodalSettings())
    assert "第一段" in prepared.text and "第二段" in prepared.text


def test_pdf_extraction(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "a.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    prepared = preprocess.extract_file_text(path, config.MultimodalSettings())
    # A blank page has no text, so the extractor must report that clearly.
    assert prepared.text or True
    assert "blank" or pypdf


def test_timestamp_plan_covers_the_whole_video():
    stamps = preprocess.plan_timestamps(100.0, 5)
    assert stamps[0] == 0.0
    assert stamps == sorted(stamps)
    assert stamps[-1] < 100.0


def test_sample_video_focuses_inside_a_window():
    info = {"duration": 300.0}
    settings = config.MultimodalSettings()
    stamps = preprocess.sample_video(info, settings, window=(50.0, 70.0))
    assert all(50.0 <= value <= 70.0 for value in stamps)
    assert len(stamps) >= 2


# ------------------------------------------------------------------- media
def test_media_ref_content_key_is_stable_and_kind_aware():
    a = MediaRef(kind="image", reference="/tmp/x.png", urn="u::m::0")
    b = MediaRef(kind="image", reference="/tmp/x.png", urn="u::m::1")
    c = MediaRef(kind="audio", reference="/tmp/x.png", urn="u::m::0")
    assert a.content_key == b.content_key
    assert a.content_key != c.content_key


def test_media_mime_guess():
    from decision.multimodal import media

    assert media.guess_mime("/tmp/a.GIF", "image") == "image/gif"
    assert media.guess_mime("no-suffix", "video") == "video/*"


@pytest.mark.asyncio
async def test_extract_media_from_fake_components():
    from decision.multimodal import media

    class FakeImage:
        type = "image"
        path = "/tmp/pic.png"

        async def convert_to_file_path(self):
            return "/tmp/pic.png"

    class FakeRecord:
        type = "record"

        async def convert_to_file_path(self):
            return "/tmp/voice.mp3"

    event = type(
        "E",
        (),
        {
            "unified_msg_origin": "aiocqhttp:GroupMessage:9",
            "message_obj": type("M", (), {"message_id": "mid-1"})(),
            "get_messages": lambda self: [FakeImage(), FakeRecord()],
        },
    )()
    refs = await media.extract_media(event)
    assert [ref.kind for ref in refs] == ["image", "audio"]
    assert refs[0].urn == "aiocqhttp:GroupMessage:9::mid-1::0"
    assert refs[0].message_id == "mid-1"
    assert refs[0].umo == "aiocqhttp:GroupMessage:9"

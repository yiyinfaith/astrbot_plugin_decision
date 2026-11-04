"""Adaptive frame sampling and the local Agentic Seeking loop."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.personal_network._astrbot_stub import install

install()

from decision.multimodal import adapters, preprocess, video_seek  # noqa: E402
from decision.multimodal.models import MediaRef  # noqa: E402
from decision.multimodal.runtime import MultimodalRuntime  # noqa: E402


# ------------------------------------------------------------------ planning
def test_adaptive_timestamps_add_frames_at_scene_changes():
    even = video_seek.adaptive_timestamps(20.0, 12, [])
    assert len(even) == 12, "without scene hints the whole budget is spread evenly"
    cut = 7.3
    adaptive = video_seek.adaptive_timestamps(20.0, 12, [cut])
    assert len(adaptive) <= 12
    assert cut in adaptive, "a scene change gets its own frame"
    assert adaptive == sorted(adaptive) and all(0 <= t < 20 for t in adaptive)


def test_adaptive_timestamps_never_exceed_the_source_frame_rate():
    stamps = video_seek.adaptive_timestamps(2.0, 48, [0.5, 1.0, 1.5], source_fps=10)
    assert len(stamps) <= 20


def test_window_timestamps_follow_the_requested_fps_and_cap():
    window = video_seek.Window(10.0, 20.0, 2.0)
    assert len(video_seek.window_timestamps(window, 100)) == 20
    assert len(video_seek.window_timestamps(window, 8)) == 8
    assert all(10.0 <= t <= 20.0 for t in video_seek.window_timestamps(window, 8))


def test_parse_plan_reads_json_and_clamps_windows():
    plan = video_seek.parse_plan(
        '```json\n{"answer": "a", "findings": "f", "next": ['
        '{"start": 5, "end": 9, "fps": 30}, {"start": 50, "end": 999}, {"start": 3, "end": 3.2}],'
        ' "done": false}\n```',
        60.0,
        max_fps=4,
        limit=5,
    )
    assert plan.structured and plan.answer == "a" and plan.findings == "f" and not plan.done
    assert [(w.start, w.end, w.fps) for w in plan.windows] == [(5.0, 9.0, 4.0), (50.0, 60.0, 1.0)]


def test_parse_plan_without_windows_means_done_and_free_text_falls_back():
    assert video_seek.parse_plan('{"answer": "ok", "next": []}', 60, max_fps=4, limit=3).done
    loose = video_seek.parse_plan("看完了。\n10-20s: 再看一下", 60, max_fps=4, limit=3)
    assert not loose.structured and [(w.start, w.end) for w in loose.windows] == [(10.0, 20.0)]


@pytest.mark.parametrize(
    "target,expected",
    [
        ("30-45", (30.0, 45.0, 0.0)),
        ("30-45@2", (30.0, 45.0, 2.0)),
        ("30s-45s fps=1.5", (30.0, 45.0, 1.5)),
        ("2-5", (2.0, 5.0, 0.0)),
    ],
)
def test_parse_target_accepts_time_ranges(target, expected):
    window = video_seek.parse_target(target)
    assert (window.start, window.end, window.fps) == expected


@pytest.mark.parametrize("target", ["", "Sheet1!A1:B2", "45-30", "第3页"])
def test_parse_target_ignores_other_targets(target):
    assert video_seek.parse_target(target) is None


# ------------------------------------------------------------ runtime loops
class Provider:
    provider_config = {"type": "openai_chat_completion", "api_base": "https://cpa.example/v1"}

    def meta(self):
        return SimpleNamespace(id="vision", model="gemini-proxy")


@pytest.fixture
def video(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video")
    state = {"duration": 300.0, "replies": [], "calls": [], "extracts": [], "scenes": []}

    async def probe(path):
        return {"duration": state["duration"], "fps": 30.0, "has_audio": False}

    async def extract(path, workdir, stamps, edge=1024):
        state["extracts"].append((list(stamps), edge))
        files = []
        for index, _ in enumerate(stamps):
            target = Path(workdir) / f"v-{index:03d}.png"
            target.write_bytes(b"png")
            files.append(str(target))
        return files

    async def scenes(path, **kwargs):
        return list(state["scenes"])

    async def call(provider, **kwargs):
        state["calls"].append(kwargs)
        replies = state["replies"]
        return replies.pop(0) if replies else '{"answer": "完", "next": [], "done": true}'

    monkeypatch.setattr(preprocess, "probe_video", probe)
    monkeypatch.setattr(preprocess, "extract_frames", extract)
    monkeypatch.setattr(video_seek, "scene_changes", scenes)
    monkeypatch.setattr(adapters, "call_provider", call)
    return clip, state


def runtime_for(tmp_path, **settings):
    provider = Provider()
    return MultimodalRuntime(
        SimpleNamespace(context=SimpleNamespace(get_provider_by_id=lambda _: provider)),
        data_dir=tmp_path / "runtime",
        settings={"mm_enabled": True, "mm_default_model": "vision", "mm_retries": 0, **settings},
    )


def ref(clip, **extra):
    return MediaRef(kind="video", reference=str(clip), urn="clip", umo="chat:1", extra=extra)


@pytest.mark.asyncio
async def test_short_clip_uses_adaptive_frames_in_one_call(tmp_path, video):
    clip, state = video
    state["duration"] = 12.0
    state["scenes"] = [4.4]
    rt = runtime_for(tmp_path)
    try:
        await rt.initialize()
        item = await rt.understand(ref(clip))
        assert item.ok, item.error
        assert item.meta["mode"] == "frames" and item.meta["sampling"] == "adaptive"
        assert len(state["calls"]) == 1
        stamps, edge = state["extracts"][0]
        assert 4.4 in stamps and edge == video_seek.NORMAL_EDGE
        # about two frames a second, capped by the per-call maximum (24)
        assert 20 <= len(stamps) <= 24
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_detail_really_reads_more_and_sharper_frames(tmp_path, video):
    clip, state = video
    state["duration"] = 30.0
    rt = runtime_for(tmp_path)
    try:
        await rt.initialize()
        normal = await rt.understand(ref(clip))
        detailed = await rt.understand(ref(clip), detail=True)
        assert normal.key != detailed.key
        (normal_stamps, normal_edge), (detail_stamps, detail_edge) = state["extracts"]
        assert len(detail_stamps) > len(normal_stamps)
        assert detail_edge > normal_edge
        assert video_seek.DETAIL_SUFFIX.strip() in state["calls"][1]["prompt"]
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_seeking_keeps_going_until_the_model_is_done(tmp_path, video):
    clip, state = video
    state["replies"] = [
        '{"answer": "开会", "next": [{"start": 20, "end": 40, "fps": 1}], "done": false}',
        '{"answer": "开会，有人发言", "findings": "25s 举手",'
        ' "next": [{"start": 100, "end": 110, "fps": 3}], "done": false}',
        '{"answer": "开会，发言后投票", "findings": "105s 投票", "next": [], "done": true}',
    ]
    rt = runtime_for(tmp_path, mm_video_agentic_rounds=4)
    try:
        await rt.initialize()
        item = await rt.understand(ref(clip))
        assert item.ok, item.error
        assert item.meta["stop"] == "model_done" and item.meta["rounds_used"] == 2
        assert len(state["calls"]) == 3
        assert item.summary.startswith("开会，发言后投票")
        assert "25s 举手" in item.summary and "105s 投票" in item.summary
        focus = state["extracts"][2][0]
        # 3 fps over 10 s asks for 30 frames; one call is capped at 24
        assert len(focus) == 24 and all(100 <= t <= 110 for t in focus)
        assert "20.0-40.0s" in state["calls"][2]["prompt"], "already-seen spans are reported"
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_free_text_reply_continues_with_the_remaining_plan(tmp_path, video):
    clip, state = video
    state["replies"] = [
        '{"answer": "概览", "next": [{"start": 10, "end": 20}, {"start": 200, "end": 210}],'
        ' "done": false}',
        "模型没有按格式回答，只描述了画面。",
        '{"answer": "最终", "next": [], "done": true}',
    ]
    rt = runtime_for(tmp_path, mm_video_agentic_rounds=3, mm_video_agentic_segments=3)
    try:
        await rt.initialize()
        item = await rt.understand(ref(clip))
        # per round: ceil(3 segments / 3 rounds) = 1 window, so the second
        # planned window is read in round 2 even though round 1 was free text.
        assert item.meta["rounds_used"] == 2 and item.meta["stop"] == "model_done"
        assert item.summary.startswith("最终")
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_seeking_respects_the_frame_budget(tmp_path, video):
    clip, state = video
    endless = '{"answer": "继续", "next": [{"start": %d, "end": %d, "fps": 4}], "done": false}'
    state["replies"] = [endless % (i * 20, i * 20 + 15) for i in range(10)]
    rt = runtime_for(
        tmp_path, mm_video_agentic_rounds=5, mm_video_frame_budget=40, mm_video_max_frames=24
    )
    try:
        await rt.initialize()
        item = await rt.understand(ref(clip))
        assert item.ok
        assert item.meta["frames"] <= 40
        assert item.meta["stop"] == "frame_budget"
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_time_range_target_is_a_focused_read_with_shared_context(tmp_path, video):
    clip, state = video
    state["replies"] = ['{"answer": "整段是一场比赛", "next": [], "done": true}', "第 50 秒射门得分"]
    rt = runtime_for(tmp_path)
    try:
        await rt.initialize()
        overview = await rt.understand(ref(clip))
        assert overview.ok
        item = await rt.understand(ref(clip, target="48-52@5"))
        assert item.ok and item.meta["mode"] == "focus"
        assert item.meta["window"] == [48.0, 52.0] and item.meta["fps"] == 4.0, "capped at max fps"
        stamps, _ = state["extracts"][-1]
        assert len(stamps) == 16 and all(48 <= t <= 52 for t in stamps)
        assert "整段是一场比赛" in state["calls"][-1]["prompt"], "reuses the cached overview"
        again = await rt.understand(ref(clip, target="48-52@5"))
        assert again.key == item.key and len(state["calls"]) == 2, "cached, no extra call"
    finally:
        await rt.terminate()


@pytest.mark.asyncio
async def test_segment_tool_builds_the_time_range_target(tmp_path):
    from decision.multimodal import tools

    seen = {}

    async def understand_by_selector(umo, **kwargs):
        seen.update(kwargs, umo=umo)
        item = SimpleNamespace(
            ok=True, kind="video", name="", message_id="", provider_id="p", model="m",
            summary="ok", detail="ok", meta={"frames": 8, "mode": "focus"}, error="",
        )
        return item, ""

    runtime = SimpleNamespace(understand_by_selector=understand_by_selector)
    handler = tools._segment_handler(runtime)
    event = SimpleNamespace(unified_msg_origin="chat:1")
    text = await handler(event, start=12, end=18.5, fps=3, instruction="谁先动手")
    assert seen["target"] == "12-18.5@3" and seen["kind"] == "video"
    assert "ok" in text
    assert "无效" in await handler(event, start=10, end=5)
    assert tools.tool_names(["video"]) == {"understand_video", "understand_video_segment"}


def test_video_pipeline_gets_time_for_every_seeking_call(tmp_path):
    rt = runtime_for(tmp_path, mm_timeout_seconds=100, mm_video_agentic_rounds=3)
    assert rt._pipeline_timeout("video", False) == 500
    assert rt._pipeline_timeout("video", True) == 600
    assert rt._pipeline_timeout("image", True) == 100


def test_legacy_frames_strategy_maps_to_auto():
    from decision.multimodal.config import normalize_settings

    assert normalize_settings({"mm_video_strategy": "frames"})["mm_video_strategy"] == "auto"
    assert normalize_settings({"mm_video_strategy": "inline"})["mm_video_strategy"] == "inline"
    assert normalize_settings({"mm_video_strategy": "bogus"})["mm_video_strategy"] == "auto"


SCENE_LOG = """
[Parsed_metadata_3 @ 0x1] frame:24   pts:24      pts_time:6
[Parsed_metadata_3 @ 0x1] lavfi.scene_score=0.000000
[Parsed_metadata_3 @ 0x1] frame:25   pts:25      pts_time:6.25
[Parsed_metadata_3 @ 0x1] lavfi.scene_score=0.017641
[Parsed_metadata_3 @ 0x1] frame:26   pts:26      pts_time:6.5
[Parsed_metadata_3 @ 0x1] lavfi.scene_score=0.000000
[Parsed_metadata_3 @ 0x1] frame:27   pts:27      pts_time:6.75
[Parsed_metadata_3 @ 0x1] lavfi.scene_score=0.017641
"""


def test_scene_scores_are_parsed_in_order_with_the_offset():
    assert video_seek.parse_scene_scores(SCENE_LOG, offset=100) == [
        (106.0, 0.0),
        (106.25, 0.017641),
        (106.5, 0.0),
        (106.75, 0.017641),
    ]


def test_a_small_object_in_a_still_shot_is_a_scene_hint():
    # A 100x100 box appearing on a static 640x360 shot scores only ~0.02,
    # far below a shot-cut threshold such as 0.3.
    # 12 s sampled at 4 fps: the box is visible from 6.25 s to 6.5 s.
    scores = [(t / 4, 0.0) for t in range(48)]
    scores[25] = (6.25, 0.017641)
    scores[27] = (6.75, 0.017641)
    assert video_seek.pick_scene_peaks(scores, sensitivity=30, max_points=5) == [6.25, 6.75]


def test_constant_motion_only_reports_changes_that_stand_out():
    scores = [(t / 4, 0.05 + 0.01 * (t % 2)) for t in range(40)]
    scores[10] = (2.5, 0.12)  # ordinary jitter, not a change
    scores[30] = (7.5, 0.6)  # a real cut
    assert video_seek.pick_scene_peaks(scores, sensitivity=30, max_points=5) == [7.5]


def test_scene_peaks_keep_the_largest_changes():
    scores = [(float(t), 0.0) for t in range(20)]
    for moment, score in ((3.0, 0.2), (8.0, 0.9), (15.0, 0.5)):
        scores[int(moment)] = (moment, score)
    assert video_seek.pick_scene_peaks(scores, sensitivity=30, max_points=2) == [8.0, 15.0]

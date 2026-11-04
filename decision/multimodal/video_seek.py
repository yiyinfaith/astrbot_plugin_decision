"""Frame planning and the local Agentic Seeking loop for video understanding.

Everything here runs on FFmpeg frames plus an ordinary vision model, so it
works through any chat-only gateway (for example a CPA reverse proxy for
Gemini) without the Files or Interactions APIs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import preprocess

# Hard ceilings that no setting can raise (cost and request-size guards).
HARD_MAX_FRAMES_PER_CALL = 96
MIN_FOCUS_FPS = 0.2
SCENE_SAMPLE_FPS = 4.0
SCENE_DOWNSCALE = 160
NOTES_CHAR_LIMIT = 6000
DETAIL_FRAME_FACTOR = 1.5
DETAIL_EDGE = 1568
NORMAL_EDGE = 1024

OVERVIEW_PROMPT = (
    "你在分析一段视频。下面是覆盖整段视频的画面（均匀采样并在画面变化处加密）及其时间点。\n"
    "本次分析目标：{goal}\n\n"
    "请先概括你目前看到的内容，再决定是否需要细看某些时间段。可以细看的时间段最多 {limit} 段，"
    "每段给出起止秒数和希望的采样帧率（fps，{min_fps}~{max_fps}），快速动作用更高帧率。\n"
    "只输出一个 JSON 对象，不要输出其它文字：\n"
    '{{"answer": "就分析目标目前能给出的完整回答", '
    '"next": [{{"start": 秒, "end": 秒, "fps": 帧率, "reason": "为什么要细看"}}], '
    '"done": 是否已经足够回答（true/false）}}'
)

SEEK_PROMPT = (
    "继续分析同一段视频。本次分析目标：{goal}\n\n"
    "到目前为止的回答：\n{answer}\n\n"
    "已经细看过的时间段：{seen}\n"
    "下面是这一轮按你的要求细看的时间段画面（时间点已标注）：{windows}\n\n"
    "请结合新画面更新回答，并自行决定：信息已足够就设 done 为 true；"
    "还有不确定之处就在 next 中给出下一批要细看的时间段（最多 {limit} 段，fps {min_fps}~{max_fps}，"
    "不要重复已细看的时间段）。剩余预算：最多 {rounds_left} 轮、{frames_left} 帧。\n"
    "只输出一个 JSON 对象，不要输出其它文字：\n"
    '{{"answer": "更新后的完整回答", "findings": "这一轮新确认的细节（带时间点）", '
    '"next": [{{"start": 秒, "end": 秒, "fps": 帧率, "reason": "原因"}}], "done": true/false}}'
)

FOCUS_PROMPT = (
    "请精读这段视频 {start:.1f}s 到 {end:.1f}s 的画面（约每秒 {fps:g} 帧，时间点已标注），"
    "按时间顺序说明动作、变化和细节。\n本次分析目标：{goal}"
)

DETAIL_SUFFIX = (
    "\n\n请做细致分析：逐个时间点比较画面差异，留意快速动作、小物体、文字与表情；"
    "不确定之处直接说明，不要猜测。"
)


@dataclass(slots=True)
class Window:
    start: float
    end: float
    fps: float
    reason: str = ""

    @property
    def span(self) -> float:
        return max(0.0, self.end - self.start)

    def label(self) -> str:
        return f"{self.start:.1f}-{self.end:.1f}s"


@dataclass(slots=True)
class SeekPlan:
    answer: str
    findings: str = ""
    windows: list[Window] = field(default_factory=list)
    done: bool = False
    structured: bool = False


# --------------------------------------------------------------------- frames
async def scene_changes(
    path: Path,
    *,
    sensitivity: int = 30,
    max_points: int = 24,
    start: float = 0.0,
    end: float | None = None,
    timeout: float = 120.0,
) -> list[float]:
    """Timestamps where the picture changes the most (FFmpeg scene scores).

    A fixed cut threshold only finds shot changes: a small object appearing
    in a still shot scores about 0.02.  So every sampled frame is scored and
    the peaks that stand out from the clip's own level (``sensitivity``:
    smaller means more sensitive) are kept, largest first.  The clip is
    decoded small and at a bounded rate, so this stays cheap for long videos.
    Any failure just means "no scene hints".
    """

    span = (end - start) if end is not None and end > start else 0.0
    sample_fps = SCENE_SAMPLE_FPS if not span or span <= 300 else SCENE_SAMPLE_FPS / 2
    command = ["ffmpeg", "-hide_banner", "-nostdin"]
    if start > 0:
        command += ["-ss", f"{start:.3f}"]
    if span:
        command += ["-t", f"{span:.3f}"]
    command += [
        "-i",
        str(path),
        "-an",
        "-vf",
        (
            f"fps={sample_fps:g},scale={SCENE_DOWNSCALE}:-2,"
            "select='gte(scene,0)',metadata=print:key=lavfi.scene_score"
        ),
        "-f",
        "null",
        "-",
    ]
    try:
        code, _, error = await preprocess.run_command(command, timeout=timeout)
    except Exception:
        return []
    if code != 0:
        return []
    return pick_scene_peaks(
        parse_scene_scores(error.decode("utf-8", "replace"), offset=start),
        sensitivity=sensitivity,
        max_points=max_points,
    )


def parse_scene_scores(text: str, *, offset: float = 0.0) -> list[tuple[float, float]]:
    """``metadata=print`` output -> [(time, scene score)] in order."""

    scores: list[tuple[float, float]] = []
    moment: float | None = None
    for line in text.splitlines():
        time_match = re.search(r"pts_time:\s*([0-9.]+)", line)
        if time_match:
            moment = float(time_match.group(1))
            continue
        score_match = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if score_match and moment is not None:
            scores.append((round(offset + moment, 3), float(score_match.group(1))))
            moment = None
    return scores


def pick_scene_peaks(
    scores: list[tuple[float, float]], *, sensitivity: int, max_points: int
) -> list[float]:
    """Local maxima clearly above the clip's typical change, largest first."""

    if not scores or max_points <= 0:
        return []
    values = sorted(score for _, score in scores)
    median = values[len(values) // 2]
    sensitivity = max(1, int(sensitivity))
    # 30 (default): at least 0.005 and four times the median change.
    threshold = max(sensitivity / 6000, median * (1 + sensitivity / 10))
    peaks = []
    for index, (moment, score) in enumerate(scores):
        before = scores[index - 1][1] if index else -1.0
        after = scores[index + 1][1] if index + 1 < len(scores) else -1.0
        if score >= threshold and score >= before and score >= after:
            peaks.append((score, moment))
    peaks.sort(reverse=True)
    return sorted(moment for _, moment in peaks[:max_points])


def adaptive_timestamps(
    duration: float,
    budget: int,
    scenes: list[float],
    *,
    start: float = 0.0,
    end: float | None = None,
    source_fps: float = 0.0,
) -> list[float]:
    """Spread ``budget`` frames over a span: an even baseline plus scene cuts.

    Scene changes (where fast action and new shots happen) get up to half of
    the budget and the rest is spaced evenly, so slow stretches stay covered.
    A cut that falls right next to a baseline frame replaces it instead of
    being dropped.  The density never exceeds the source frame rate.
    """

    end = duration if end is None or end <= 0 else min(end, duration or end)
    span = max(0.0, end - start)
    budget = max(1, int(budget))
    if span <= 0:
        return [round(start, 3)]
    if source_fps > 0:
        budget = min(budget, max(1, int(span * source_fps)))
    inside = sorted(t for t in scenes if start <= t < end)
    scene_slots = min(len(inside), budget // 2)
    baseline = budget - scene_slots
    stamps = [start + span * (index + 0.5) / baseline for index in range(baseline)]
    if scene_slots:
        # Too many cuts: keep an evenly spread selection of them.
        chosen = (
            inside
            if len(inside) <= scene_slots
            else [
                inside[round(index * (len(inside) - 1) / max(1, scene_slots - 1))]
                for index in range(scene_slots)
            ]
        )
        gap = max(0.05, span / (budget * 2))
        cuts: set[float] = set()
        for cut in dict.fromkeys(chosen):
            nearest = min(range(len(stamps)), key=lambda k: abs(stamps[k] - cut))
            if abs(stamps[nearest] - cut) < gap and stamps[nearest] not in cuts:
                stamps[nearest] = cut
            else:
                stamps.append(cut)
            cuts.add(cut)
    return sorted({round(min(max(start, t), max(start, end - 0.05)), 3) for t in stamps})


def window_timestamps(window: Window, cap: int) -> list[float]:
    count = max(2, min(int(cap), int(math.ceil(window.span * window.fps)) or 2))
    step = window.span / count
    return [round(window.start + step * (index + 0.5), 3) for index in range(count)]


def stamp_notes(stamps: list[float]) -> str:
    return "画面时间点（秒）：" + "、".join(f"{s:.2f}" for s in stamps)


# ---------------------------------------------------------------- planning
def clamp_window(raw: Any, duration: float, *, max_fps: float) -> Window | None:
    try:
        start = float(raw.get("start"))
        end = float(raw.get("end"))
    except (AttributeError, TypeError, ValueError):
        return None
    if duration > 0:
        start, end = max(0.0, min(start, duration)), max(0.0, min(end, duration))
    if end - start < 0.5:
        return None
    try:
        fps = float(raw.get("fps") or 1.0)
    except (TypeError, ValueError):
        fps = 1.0
    fps = max(MIN_FOCUS_FPS, min(float(max_fps), fps))
    return Window(start, end, fps, str(raw.get("reason") or "")[:200])


def _json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_plan(text: str, duration: float, *, max_fps: float, limit: int) -> SeekPlan:
    """Read the model's JSON decision; fall back to "start-end" lines."""

    payload = _json_object(text)
    if payload is not None:
        windows = [
            window
            for raw in (payload.get("next") or [])
            if isinstance(raw, dict)
            and (window := clamp_window(raw, duration, max_fps=max_fps)) is not None
        ][: max(0, limit)]
        return SeekPlan(
            answer=str(payload.get("answer") or "").strip(),
            findings=str(payload.get("findings") or "").strip(),
            windows=windows,
            done=bool(payload.get("done")) or not windows,
            structured=True,
        )
    windows = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-~到]\s*(\d+(?:\.\d+)?)", text or ""):
        window = clamp_window(
            {"start": match.group(1), "end": match.group(2), "fps": 1.0}, duration, max_fps=max_fps
        )
        if window is not None:
            windows.append(window)
    return SeekPlan(answer=(text or "").strip(), windows=windows[: max(0, limit)])


def overlaps_seen(window: Window, seen: list[Window]) -> bool:
    """True when most of ``window`` has already been inspected."""

    if window.span <= 0:
        return True
    covered = sum(
        max(0.0, min(window.end, other.end) - max(window.start, other.start)) for other in seen
    )
    return covered >= 0.8 * window.span


def even_windows(duration: float, count: int, fps: float) -> list[Window]:
    """Fallback windows when the overview gave no usable plan."""

    count = max(1, count)
    step = (duration or count) / count
    return [Window(step * index, step * (index + 1), fps, "均匀补看") for index in range(count)]


def parse_target(target: str) -> Window | None:
    """``30-45`` / ``30-45@2`` / ``30s-45s fps=2`` -> a focus window (fps optional)."""

    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-~到]\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?"
        r"\s*(?:(?:@|fps\s*=?)\s*(\d+(?:\.\d+)?))?\s*",
        target or "",
        re.I,
    )
    if not match:
        return None
    start, end = float(match.group(1)), float(match.group(2))
    if end <= start:
        return None
    fps = float(match.group(3)) if match.group(3) else 0.0
    return Window(start, end, fps)


def truncate(text: str, limit: int = NOTES_CHAR_LIMIT) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"

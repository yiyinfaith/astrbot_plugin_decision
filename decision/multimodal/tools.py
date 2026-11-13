"""The LLM tools that let the model ask for media understanding on demand.

They are registered through the plugin's normal tool path, so the existing decision
tool filtering, scope lists and enable/disable switches all apply unchanged.
"""

from __future__ import annotations

from typing import Any

KIND_ARG = {
    "type": "string",
    "description": "媒体种类，可选 image / video / audio / file；留空表示任意类型",
    "enum": ["image", "video", "audio", "file", ""],
}

SELECTOR_ARG = {
    "type": "string",
    "description": (
        "定位哪一份媒体：latest 表示最近一条（默认），"
        "数字表示最近媒体列表中的第几个（1 为最近），"
        "其它字符串按文件名片段匹配"
    ),
}

TARGET_ARG = {
    "type": "string",
    "description": (
        "只分析其中一部分：PDF 页码（如 3 或 2-5）、PPTX 幻灯片页码、"
        "DOCX 章节标题、XLSX 工作表名或单元格区域（如 Sheet1!A1:D20）"
    ),
}

VIDEO_TARGET_ARG = {
    "type": "string",
    "description": "只精读一段时间（秒）：如 30-45，或 30-45@2 表示每秒 2 帧",
}

INSTRUCTION_ARG = {
    "type": "string",
    "description": "可选的自定义分析指令，例如“分析这段视频第 20 秒发生了什么”",
}

DETAIL_ARG = {
    "type": "boolean",
    "description": (
        "细致分析：视频会抽更多、更高清的画面并多给一轮精读，"
        "回答更细（会消耗更多 token）"
    ),
}

FORCE_ARG = {
    "type": "boolean",
    "description": "是否忽略已有缓存重新分析；缓存已有时默认复用以免重复计费",
}


# Every tool name a media kind owns; the plugin registers, toggles and lists
# understanding tools only through this table.
TOOL_NAMES = {
    "image": ("understand_image",),
    "video": ("understand_video", "understand_video_segment"),
    "audio": ("understand_audio",),
    "file": ("understand_file",),
}


def tool_names(kinds: tuple[str, ...] | list[str] | set[str]) -> set[str]:
    return {name for kind in kinds for name in TOOL_NAMES.get(kind, ())}


def all_tool_names() -> set[str]:
    return tool_names(tuple(TOOL_NAMES))


def _schema(kind: str) -> dict[str, Any]:
    properties = {
        "selector": SELECTOR_ARG,
        "instruction": INSTRUCTION_ARG,
        "target": VIDEO_TARGET_ARG if kind == "video" else TARGET_ARG,
        "detail": DETAIL_ARG,
        "force": FORCE_ARG,
    }
    if not kind:
        properties["kind"] = KIND_ARG
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


def _result(item: Any, error: str, *, detail: bool) -> str:
    if item is None:
        return f"无法理解该媒体：{error or '未知原因'}"
    if not item.ok:
        return f"无法理解该媒体：{item.error}"
    body = item.detail if detail and item.detail else item.summary
    lines = [
        f"类型：{item.kind}",
    ]
    if item.name:
        lines.append(f"来源：{item.name}")
    if item.message_id:
        lines.append(f"消息：{item.message_id}")
    if item.provider_id:
        lines.append(f"模型：{item.provider_id} / {item.model}")
    lines.append("内容：")
    lines.append(body)
    meta = item.meta or {}
    if meta.get("frames"):
        lines.append(f"（共 {meta['frames']} 帧，模式 {meta.get('mode', 'frames')}）")
    if meta.get("mode") == "agentic":
        lines.append(
            f"（精读 {meta.get('rounds_used', 0)} 轮、{meta.get('windows', 0)} 段，"
            "需要时可用 understand_video_segment 精读指定时间段）"
        )
    if meta.get("segments"):
        lines.append(f"（音频分 {meta['segments']} 段处理）")
    return "\n".join(lines)


def build_tools(runtime: Any, kinds: tuple[str, ...]) -> list[Any]:
    """Create the enabled understanding tools.

    ``kinds`` decides which of the four tools exist at all, so an operator can
    expose only the modalities they pay for.
    """

    from astrbot.core.agent.tool import FunctionTool

    specs = (
        ("understand_image", "image", "理解当前会话中的图片。动图会按帧序列理解动画过程。"),
        (
            "understand_video",
            "video",
            "理解当前会话中的视频。短视频按时间顺序抽帧，长视频先概括再精读关键片段。",
        ),
        ("understand_audio", "audio", "理解当前会话中的音频，包括语音转录、环境声音与音乐。"),
        ("understand_file", "file", "理解当前会话中发送的文件（PDF/DOCX/PPTX/XLSX/文本等）。"),
    )
    segment = (
        "understand_video_segment",
        "video",
        "精读视频中的一段时间：按指定起止秒数和每秒帧数重新抽帧分析，"
        "适合确认快速动作或某个时刻的细节。",
    )
    tools: list[Any] = []
    for name, kind, description in specs:
        if kind not in kinds:
            continue
        tools.append(
            FunctionTool(
                name=name,
                description=description,
                parameters=_schema(kind),
                handler=_handler(runtime, kind),
            )
        )
    if "video" in kinds:
        name, kind, description = segment
        tools.append(
            FunctionTool(
                name=name,
                description=description,
                parameters=SEGMENT_SCHEMA,
                handler=_segment_handler(runtime),
            )
        )
    return tools


SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "selector": SELECTOR_ARG,
        "start": {"type": "number", "description": "开始时间（秒）"},
        "end": {"type": "number", "description": "结束时间（秒）"},
        "fps": {
            "type": "number",
            "description": "每秒抽几帧；快速动作用更高的值，留空按时长自动决定",
        },
        "instruction": INSTRUCTION_ARG,
        "detail": DETAIL_ARG,
        "force": FORCE_ARG,
    },
    "required": ["start", "end"],
    "additionalProperties": False,
}


def _segment_handler(runtime: Any):
    async def handle(event: Any, **kwargs: Any) -> str:
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        if not umo:
            return "无法确定当前会话，已拒绝访问媒体。"
        try:
            start = float(kwargs.get("start"))
            end = float(kwargs.get("end"))
            fps = float(kwargs.get("fps") or 0)
        except (TypeError, ValueError):
            return "start、end、fps 必须是数字（秒 / 每秒帧数）。"
        if start < 0 or end <= start or fps < 0:
            return "时间范围无效：需要 0 ≤ start < end，fps 不能为负。"
        target = f"{start:g}-{end:g}" + (f"@{fps:g}" if fps else "")
        detail = bool(kwargs.get("detail"))
        item, error = await runtime.understand_by_selector(
            umo,
            kind="video",
            selector=str(kwargs.get("selector") or "latest"),
            instruction=str(kwargs.get("instruction") or ""),
            detail=detail,
            force=bool(kwargs.get("force")),
            target=target,
        )
        return _result(item, error, detail=detail)

    return handle


def _handler(runtime: Any, kind: str):
    async def handle(event: Any, **kwargs: Any) -> str:
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        if not umo:
            return "无法确定当前会话，已拒绝访问媒体。"
        selector = str(kwargs.get("selector") or "latest")
        target = str(kwargs.get("target") or "")
        instruction = str(kwargs.get("instruction") or "")
        detail = bool(kwargs.get("detail"))
        force = bool(kwargs.get("force"))
        item, error = await runtime.understand_by_selector(
            umo,
            kind=kind,
            selector=selector,
            instruction=instruction,
            detail=detail,
            force=force,
            target=target,
        )
        return _result(item, error, detail=detail)

    return handle

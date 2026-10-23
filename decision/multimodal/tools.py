"""The four LLM tools that let the model ask for media understanding on demand.

They are registered through the plugin's normal tool path, so the existing JEV
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

INSTRUCTION_ARG = {
    "type": "string",
    "description": "可选的自定义分析指令，例如“分析这段视频第 20 秒发生了什么”",
}

DETAIL_ARG = {
    "type": "boolean",
    "description": "是否返回更详细的分析结果（会消耗更多 token）",
}

FORCE_ARG = {
    "type": "boolean",
    "description": "是否忽略已有缓存重新分析；缓存已有时默认复用以免重复计费",
}


def _schema(kind: str) -> dict[str, Any]:
    properties = {
        "selector": SELECTOR_ARG,
        "instruction": INSTRUCTION_ARG,
        "target": TARGET_ARG,
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
        ("understand_video", "video", "理解当前会话中的视频。短视频按时间顺序抽帧，长视频先概括再精读关键片段。"),
        ("understand_audio", "audio", "理解当前会话中的音频，包括语音转录、环境声音与音乐。"),
        ("understand_file", "file", "理解当前会话中发送的文件（PDF/DOCX/PPTX/XLSX/文本等）。"),
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
    return tools


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

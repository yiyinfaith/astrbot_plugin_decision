"""Allowlisted administrator WebUI operations; no command registration."""

from __future__ import annotations

import base64
from pathlib import Path

# Human-readable explanations for the runtime's short reason codes, so the WebUI
# never shows a bare English token to the operator.
REASON_TEXT = {
    "disabled": "主动发起未启用（私聊主动与群聊主动都关闭了）。",
    "quiet hours": "当前处于免打扰时段。",
    "platform offline": "平台连续失败后暂停中，请稍后再试。",
    "schedule unavailable": "今天还没有生成日程，请先点“生成今日内容”。",
    "agent task active": "该会话正在处理另一条 Agent 任务，请稍后再试。",
    "conversation is not idle enough": "会话刚有互动，暂不打扰。",
    "request failed": "生成或发送时出错，详见运行结果。",
    "Jev 未通过或候选已失效": "Jev 判断此刻不适合发起，或候选已过期。",
}


class AdminView:
    def __init__(self, umo):
        self.unified_msg_origin = umo
        self.role = "admin"

    def is_admin(self):
        return True

    def plain_result(self, text):
        # Upstream command references are not exposed as a second command UI.
        import re

        text = re.sub(r"/大时间表\s*[^，。；\n]*", "WebUI → 主动发起 → 长期时间表", str(text))
        return {"type": "text", "text": text}

    def image_result(self, path):
        raw = Path(path).read_bytes()
        return {"type": "image", "src": "data:image/png;base64," + base64.b64encode(raw).decode()}


async def perform_life_action(runtime, umo, action, text="", date_range=""):
    event = AdminView(umo)
    persona = await runtime.personas.resolve(umo)
    if action == "inspect":
        plan = runtime.visible_plan(
            runtime.storage.get_plan(runtime._now().date().isoformat(), persona.id)
        )
        return {
            "persona_id": persona.id,
            "plan": plan.to_dict() if plan else None,
            "timeline": runtime.long_term.list_for_persona(persona.id),
            "draft": runtime.long_term.get_draft(persona.id),
            "requirements": [
                r.to_dict()
                for r in runtime.storage.schedule_requirements.values()
                if r.persona_id == persona.id
            ],
        }
    if action == "add_requirement":
        return await runtime._add_schedule_requirement(event, date_range, text)
    if action == "settle_delay":
        await runtime.reply_delay_coordinator.settle_now(umo)
        return "已结算当前会话等待批次"
    if action == "initiate":
        _, plan = await runtime._ensure_plan_for_umo(umo)
        sent, reason = await runtime._attempt_unsolicited(
            umo, persona, plan, text or "管理员请求此刻自然地发起一次聊天", "manual"
        )
        return {
            "sent": sent,
            "reason": reason,
            "message": "已主动发起一条消息。"
            if sent
            else f"本次没有发送：{REASON_TEXT.get(reason, reason)}",
        }
    dispatch = {
        "generate": (runtime.rewrite_schedule, (event, text)),
        "rewrite_schedule": (runtime.rewrite_daily_timeline, (event, text)),
        "rewrite_outfit": (runtime.rewrite_daily_outfit, (event, text)),
        "generate_timeline": (runtime.long_term_generate, (event, text)),
        "regenerate_timeline": (runtime.long_term_regenerate, (event, text)),
        "import_timeline": (runtime.long_term_import, (event, text)),
        "approve_timeline": (runtime.long_term_approve, (event,)),
        "reject_timeline": (runtime.long_term_reject, (event, text)),
        "render_schedule": (runtime.show_schedule, (event,)),
        "render_outfit": (runtime.show_outfit, (event,)),
        "render_timeline": (runtime.long_term_list, (event,)),
        "render_stage": (runtime.long_term_view, (event, text or None)),
    }
    method, args = dispatch[action]
    return [item async for item in method(*args)]

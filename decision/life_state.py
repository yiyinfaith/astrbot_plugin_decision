"""Persona-wide continuity derived from the saved plan, never external completion."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from .virtual_life.core.utils import timeline_item_at


class PersonaLifeState:
    def __init__(self, runtime, ledger):
        self.runtime, self.ledger = runtime, ledger

    def current(self, persona, plan, now):
        cfg = self.runtime.config["continuity_settings"]
        if not cfg["enable"] or not cfg["state_enabled"] or not plan:
            return {}
        item = timeline_item_at(plan, now)
        if not item:
            return {}
        start = datetime.combine(now.date(), datetime.strptime(item.start, "%H:%M").time(), now.tzinfo)
        end = (start.replace(hour=0, minute=0) + timedelta(days=1)) if item.end == "24:00" else datetime.combine(now.date(), datetime.strptime(item.end, "%H:%M").time(), now.tzinfo)
        progress = max(0, min(1, (now - start).total_seconds() / max(1, (end - start).total_seconds())))
        phase = "刚进入活动" if progress < 1 / 3 else "活动进行中" if progress < 2 / 3 else "活动接近计划结束"
        tone = {"sleep": "休息中", "busy": "忙碌", "focus": "专注", "transit": "在路上", "social": "社交中", "available": "可交流"}.get(item.state, "平稳")
        previous = next((entry for entry in reversed(plan.timeline) if entry.end == item.start and entry.id != item.id), None)
        recovery_end = min(end, start + timedelta(minutes=30))
        recovering = item.state == "available" and previous and previous.state in {"busy", "focus", "transit", "social"} and now < recovery_end
        affect = {"feeling": "逐渐放松" if recovering else (plan.mood if item.state == "available" else tone),
                  "cause": f"日程从{previous.activity}切换到{item.activity}" if recovering else f"当前日程活动：{item.activity}",
                  "since": start.timestamp(), "until": recovery_end.timestamp() if recovering else end.timestamp()}
        if item.state == "available" and previous and previous.state in {"busy", "focus", "transit", "social"} and not recovering:
            affect["since"] = recovery_end.timestamp()
        key = f"{plan.date}:{plan.revision}:{item.id}:{phase}"
        with self.ledger.transaction() as db:
            row = db.execute("SELECT data FROM persona_states WHERE persona=?", (persona,)).fetchone()
            old = json.loads(row[0]) if row else {}
            events = old.get("events", [])
            if old.get("affect", {}).get("feeling") != affect["feeling"] or old.get("affect", {}).get("cause") != affect["cause"]:
                events.append({"at": now.timestamp(), "type": "affect", "activity": item.activity, "fact": affect["feeling"], "cause": affect["cause"]})
            if old.get("key") != key:
                if old.get("activity") and old.get("item_id") != item.id and old.get("planned_end", 0) <= now.timestamp():
                    events.append({"at": old["planned_end"], "type": "planned_transition", "activity": old["activity"], "fact": "原日程活动的计划时段已结束；不是外部完成凭据"})
                events.append({"at": now.timestamp(), "type": "phase", "activity": item.activity, "fact": phase, "cause": tone})
            state = {"key": key, "persona": persona, "item_id": item.id, "activity": item.activity,
                     "location": item.location, "phase": phase, "progress": round(progress, 3),
                     "base_mood": plan.mood, "tone": tone, "planned_start": start.timestamp(),
                     "affect": affect,
                     "planned_end": end.timestamp(), "events": events[-12:],
                     "source": "当前已保存的虚拟日程；不能据此捏造真实见闻或外部任务完成"}
            db.execute("INSERT OR REPLACE INTO persona_states VALUES (?,?)", (persona, json.dumps(state, ensure_ascii=False)))
        return state

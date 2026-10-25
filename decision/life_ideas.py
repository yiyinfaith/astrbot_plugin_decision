"""Persona activity ideas with source, budget and per-session sharing state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re

from .virtual_life.core.utils import timeline_item_at


class LifeIdeas:
    def __init__(self, runtime, ledger):
        self.runtime, self.ledger = runtime, ledger
        self.contacts = runtime.contacts
        self.thought_lock = asyncio.Lock()

    async def model(self, *args, **kwargs):
        return await self.runtime.temporal.model(*args, **kwargs)

    async def save_thought(self, persona, text, source, *, node="", scope=None):
        settings, now = self.runtime.config["thought_settings"], self.runtime._now()
        if not settings["enable"] or not settings["share_tool_enable"]:
            raise ValueError("想法暂存已关闭")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1500 or not source:
            raise ValueError("想法须有简短内容和真实活动来源")
        ident = hashlib.sha256((persona + "\0" + node + "\0" + text.strip()).encode()).hexdigest()
        with self.ledger.transaction() as db:
            db.execute("DELETE FROM thoughts WHERE expires<?", (now.timestamp(),))
            count = db.execute("SELECT COUNT(*) FROM thoughts WHERE persona=?", (persona,)).fetchone()[0]
            if count >= settings["max_pending_per_persona"]:
                raise ValueError("人格想法池已满")
            db.execute("INSERT OR IGNORE INTO thoughts VALUES (?,?,?,?,?)", (ident, persona, now.timestamp(), now.timestamp() + settings["ttl_minutes"] * 60,
                json.dumps({"text": text.strip(), "source": source, "node": node, "scope": scope}, ensure_ascii=False)))
        return ident

    def thought(self, umo, persona):
        with self.ledger.transaction() as db:
            rows = db.execute("SELECT * FROM thoughts WHERE persona=? AND expires>? AND NOT EXISTS(SELECT 1 FROM shares WHERE thought=thoughts.id AND umo=? AND status IN ('sent','dismissed')) ORDER BY created DESC LIMIT 100", (persona, self.runtime._now().timestamp(), umo)).fetchall()
            for row in rows:
                data = json.loads(row["data"])
                if data.get("scope") in {None, umo}:
                    return {**dict(row), **data}
            return None

    def mark_shared(self, ident, umo):
        with self.ledger.transaction() as db:
            db.execute("INSERT OR REPLACE INTO shares VALUES (?,?,'sent')", (ident, umo))

    async def activity_thought(self, umo, persona, plan):
        settings, now = self.runtime.config["thought_settings"], self.runtime._now()
        if not settings["enable"] or not plan:
            return
        item = timeline_item_at(plan, now)
        if not item or item.state in {"sleep", "transit"}:
            return
        node = f"{plan.date}:{plan.revision}:{item.id}"
        async with self.thought_lock:
            with self.ledger.transaction() as db:
                rows = db.execute("SELECT at,node FROM idea_attempts WHERE persona=? AND at>=?", (persona.id, self.contacts.day_bounds(now)[0])).fetchall()
                last = db.execute("SELECT MAX(at) FROM idea_attempts WHERE persona=?", (persona.id,)).fetchone()[0]
                if len(rows) >= settings["max_generations_per_persona_day"] or any(r["node"] == node for r in rows) or (last and now.timestamp() - last < settings["min_generation_minutes"] * 60):
                    return
                # Reserve the node before the paid call, including failures.
                db.execute("INSERT INTO idea_attempts VALUES (?,?,?)", (persona.id, node, now.timestamp()))
            text = await self.model(umo, "根据当前真实日程活动，生成一个符合人格、可分享的小想法。只输出一两句观点或联想；不描述未记录的声音、人、物，不使用听见、看到、遇见、刚刚发生等见闻措辞，不捏造亲身经历、外部操作或承诺，不输出内部推理。\n"
                + json.dumps({"time": now.isoformat(), "activity": item.activity, "location": item.location,
                              "continuous_state": self.runtime.life_state.current(persona.id, plan, now) if getattr(self.runtime, "life_state", None) else {}}, ensure_ascii=False), system=persona.prompt, purpose="activity_idea", persona_id=persona.id)
            if re.search(r"听见|听到|看到|看见|刚刚|遇见|遇到|买到|收到|有人|朋友说", text):
                self.runtime.record_audit(umo, False, "活动想法含无来源见闻，已丢弃并保留本次调用预算")
                return
            await self.save_thought(persona.id, text, {"activity": item.activity, "location": item.location, "at": now.isoformat()}, node=node)

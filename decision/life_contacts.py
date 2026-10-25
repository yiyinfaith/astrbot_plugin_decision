"""Bounded contact frequency and authority, isolated by full UMO + persona."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, time, timedelta

from .life_ledger import LifeLedger


def group_authorized(event, settings):
    """Only platform metadata decides authority; never accept a tool role argument."""
    if ":GroupMessage:" not in str(event.unified_msg_origin):
        return True
    ident = str(event.get_sender_id())
    listed = ident in settings["group_adjust_users"]
    if settings["group_adjust_blacklist"]:
        return not listed
    if listed or getattr(event, "role", "member") == "admin":
        return True
    raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
    if not isinstance(raw, dict):
        return False
    sender = raw.get("sender", {})
    # Some adapters expose the member role on message_obj.sender instead.
    return isinstance(sender, dict) and sender.get("role") in {"owner", "admin"}


def explicit_feedback(text):
    text = str(text or "")[:4000]
    # Requests to resume take precedence over a quoted earlier complaint only
    # when they are an unambiguous current imperative. Complex feedback is left
    # to the model's preference tool, using the actual message as evidence.
    if re.search(r"(?:恢复|重新开始|继续)(?:主动)?(?:联系|找我|聊天|发消息)", text):
        return "resume"
    if re.search(r"(?:别再|不要再|停止|暂停|别)(?:主动)?(?:找我|联系我|给我发|发消息|主动)", text):
        return "pause"
    if re.search(r"(?:太吵|太烦|少(?:一点|点)?(?:主动)?(?:找我|联系|发消息)|主动.{0,8}太频繁|别.{0,4}这么频繁)", text):
        return "less"
    if re.search(r"(?:多(?:一点|点)?(?:主动)?(?:找我|联系|发消息)|主动.{0,8}太少|怎么.{0,6}不找我)", text):
        return "more"
    return ""


class ContactControl:
    def __init__(self, runtime, ledger):
        self.runtime = runtime
        self.ledger: LifeLedger = ledger

    @property
    def settings(self):
        return self.runtime.config["adaptive_settings"]

    def bounds(self, umo):
        key = "group_min_minutes" if ":GroupMessage:" in umo else "friend_min_minutes"
        return self.settings[key], self.settings["max_interval_minutes"]

    def profile(self, db, umo, persona, now):
        low, high = self.bounds(umo)
        data = self.ledger.profile(db, umo, persona, now.timestamp(), low, high)
        if data["awaiting"] and not data["negative_applied"] and now.timestamp() - data["last_sent"] >= self.settings["response_window_minutes"] * 60:
            data["interval"] = min(high, data["interval"] * self.settings["negative_factor"])
            data["negative_applied"] = True
        self.ledger.save_profile(db, umo, persona, data)
        return data

    def summary(self, umo, persona, now):
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            return {**data, "umo": umo, "persona": persona, "today": self.counts(db, umo, persona, now), "bounds_minutes": self.bounds(umo)}

    @staticmethod
    def day_bounds(now):
        start = datetime.combine(now.date(), time.min, now.tzinfo)
        return start.timestamp(), (start + timedelta(days=1)).timestamp()

    def counts(self, db, umo, persona, now, exclude=""):
        start, end = self.day_bounds(now)
        counts = {"ordinary": 0, "agreed": 0}
        for row in db.execute("SELECT kind,COUNT(*) AS n FROM contacts WHERE umo=? AND persona=? AND at>=? AND at<? AND status IN ('reserved','sending','sent','unknown') AND id<>? GROUP BY kind", (umo, persona, start, end, exclude)):
            counts[row["kind"]] = row["n"]
        for row in db.execute("SELECT * FROM tasks WHERE umo=? AND persona=? AND due>=? AND due<? AND status IN ('pending','preparing','ready','suspended','awaiting_delivery','sending','unknown') AND id<>?", (umo, persona, start, end, exclude)):
            task = self.ledger.task(row)
            if db.execute("SELECT 1 FROM contacts WHERE id=? AND status IN ('sent','unknown','sending')", (task["id"],)).fetchone():
                continue
            if task.get("notify", True):
                counts["agreed" if task["origin"] == "user" else "ordinary"] += 1
        return counts

    def gate(self, umo, persona, now, *, agreed=False, exclude="", frequency=True):
        with self.ledger.transaction() as db:
            return self._gate(db, umo, persona, now, agreed=agreed, exclude=exclude, frequency=frequency)

    def _gate(self, db, umo, persona, now, *, agreed=False, exclude="", frequency=True):
        # ``now`` may be a future reservation date. Never advance the profile's
        # unanswered observation clock while only checking tomorrow's quota.
        data = self.profile(db, umo, persona, self.runtime._now())
        if data["paused_until"] == -1 or data["paused_until"] > now.timestamp():
            return False, "用户已暂停主动联系"
        if frequency and not agreed and data.get("defer_until", 0) > now.timestamp():
            return False, "用户暂时忙碌或已结束本轮话题"
        counts = self.counts(db, umo, persona, now, exclude)
        hard = self.runtime.policy.settings_for(umo).get("daily_hard_max", 0)
        if agreed:
            hard = self.runtime.config["temporal_settings"]["scheduled_daily_limit"]
        if counts["agreed" if agreed else "ordinary"] >= hard:
            return False, "daily budget exhausted"
        if frequency and not agreed:
            adaptive = self.settings["enable"]
            interval = data["interval"] if adaptive else self.runtime.policy.settings_for(umo).get("cooldown_minutes", 0)
            anchor = data["last_sent"] or (data["created"] if adaptive else 0)
            if now.timestamp() < anchor + interval * 60:
                return False, "自适应主动间隔未到"
        return True, "allowed"

    def observe(self, event, persona, now, *, targeted, quality_pending=False):
        umo = str(event.unified_msg_origin)
        raw_id = str(getattr(getattr(event, "message_obj", None), "message_id", "") or "")
        # Adapter events without IDs use one event-local token, not content alone:
        # two identical genuine messages must remain separate observations.
        if not raw_id:
            raw_id = event.get_extra("decision_contact_signal", "") or self.ledger.new_id()
            event.set_extra("decision_contact_signal", raw_id)
        ident = hashlib.sha256((str(event.get_sender_id()) + ":" + raw_id).encode()).hexdigest()
        action = explicit_feedback(event.get_message_str()) if targeted else ""
        authority = group_authorized(event, self.settings)
        with self.ledger.transaction() as db:
            if not db.execute("INSERT OR IGNORE INTO signals VALUES (?,?,?,?)", (umo, persona, ident, now.timestamp())).rowcount:
                return ""
            data = self.profile(db, umo, persona, now)
            if targeted:
                data["last_incoming"] = now.timestamp()
                active_exchange = (data["awaiting"] and now.timestamp() - data["last_sent"] <= self.settings["response_window_minutes"] * 60) or (data.get("last_normal_reply", 0) and now.timestamp() - data["last_normal_reply"] <= self.settings["response_window_minutes"] * 60)
                if not quality_pending and active_exchange and now.timestamp() - data["last_positive"] >= self.settings["adjustment_cooldown_minutes"] * 60:
                    low, _ = self.bounds(umo)
                    # Explicit preference remains the baseline: an enthusiastic
                    # reply cannot immediately undo a request to be quieter.
                    floor = max(low, data["baseline"] * 0.8)
                    data["interval"] = max(floor, data["interval"] * self.settings["positive_factor"])
                    data["last_positive"] = now.timestamp()
                data["awaiting"] = False
                data["unanswered"] = 0
            if action and authority:
                self._feedback(data, umo, action, now)
                event.set_extra("decision_contact_feedback", action)
            self.ledger.save_profile(db, umo, persona, data)
        return action if authority else ""

    def apply_quality(self, umo, persona, source, signal, now):
        """Automatic evidence only; this never grants a force-adjust privilege."""
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            ident = "quality:" + source["id"]
            if not db.execute("INSERT OR IGNORE INTO signals VALUES (?,?,?,?)", (umo, persona, ident, now.timestamp())).rowcount:
                return
            kind = signal.get("engagement", "neutral")
            if kind not in {"engaged", "neutral", "closing", "busy"}:
                return
            data["engagement"] = {"kind": kind, "quote": signal["quote"], "at": now.timestamp(), "sender": source["sender"],
                                  "scope": "member" if ":GroupMessage:" in umo else "conversation"}
            if kind == "engaged" and self.settings["enable"] and now.timestamp() - data["last_positive"] >= self.settings["adjustment_cooldown_minutes"] * 60 and now.timestamp() - data.get("feedback_at", 0) >= self.settings["adjustment_cooldown_minutes"] * 60:
                low, _ = self.bounds(umo)
                data["interval"] = max(low, data["baseline"] * 0.8, data["interval"] * self.settings["positive_factor"])
                data["last_positive"] = now.timestamp()
            if kind in {"busy", "closing"}:
                minutes = signal.get("busy_minutes", 20 if kind == "closing" else 60)
                if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not math.isfinite(minutes):
                    minutes = 20
                minutes = max(1, min(self.runtime.config["continuity_settings"]["temporary_quiet_minutes"], minutes))
                until = now.timestamp() + minutes * 60
                if ":GroupMessage:" in umo:
                    members = data.setdefault("member_busy", {})
                    members[source["sender"]] = until
                    data["member_busy"] = dict(sorted(members.items(), key=lambda x: x[1], reverse=True)[:50])
                else:
                    data["defer_until"] = max(data.get("defer_until", 0), until)
            self.ledger.save_profile(db, umo, persona, data)

    def normal_reply(self, event, persona, now):
        if event.get_extra("decision_life_proactive", False) or event.get_extra("decision_contact_reply_recorded", False):
            return
        umo = str(event.unified_msg_origin)
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            data["last_normal_reply"] = now.timestamp()
            self.ledger.save_profile(db, umo, persona, data)
        event.set_extra("decision_contact_reply_recorded", True)

    def _feedback(self, data, umo, action, now, interval=0, pause_until=0):
        low, high = self.bounds(umo)
        if action == "pause":
            data["paused_until"] = pause_until or -1
        elif action == "resume":
            data["paused_until"] = 0
        else:
            value = interval if action == "set" else data["interval"] * (self.settings["feedback_factor"] if action == "less" else 1 / self.settings["feedback_factor"])
            data["baseline"] = data["interval"] = min(high, max(low, value))
        data["feedback"], data["feedback_at"] = action, now.timestamp()

    def preference(self, event, persona, action="status", interval_minutes=0, pause_until=0):
        umo, now = str(event.unified_msg_origin), self.runtime._now()
        if action not in {"status", "more", "less", "pause", "resume", "set"}:
            raise ValueError("未知联系偏好操作")
        if action != "status":
            if event.get_extra("decision_life_proactive", False) or not group_authorized(event, self.settings):
                raise ValueError("当前平台身份无权强制调整本群频率")
            if not self.settings["enable"] or not self.settings["preference_tool_enable"]:
                raise ValueError("联系偏好调整已关闭")
            if action == "set" and not self.bounds(umo)[0] <= interval_minutes <= self.bounds(umo)[1]:
                raise ValueError("间隔须在 WebUI 设置的上下限内")
            if pause_until and pause_until <= now.timestamp():
                raise ValueError("暂停结束时间须在未来")
            # The ordinary observation hook already commits simple feedback.
            if event.get_extra("decision_contact_feedback", "") != action:
                with self.ledger.transaction() as db:
                    data = self.profile(db, umo, persona, now)
                    self._feedback(data, umo, action, now, interval_minutes, pause_until)
                    self.ledger.save_profile(db, umo, persona, data)
                event.set_extra("decision_contact_feedback", action)
        return json.dumps(self.summary(umo, persona, now), ensure_ascii=False)

    def delivered(self, umo, persona, now, ident=None, *, kind="ordinary"):
        ident = ident or self.ledger.new_id()
        with self.ledger.transaction() as db:
            old = db.execute("SELECT status FROM contacts WHERE id=?", (ident,)).fetchone()
            if old and old[0] == "sent":
                return
            db.execute("INSERT OR REPLACE INTO contacts VALUES (?,?,?,?,?,?)", (ident, umo, persona, now.timestamp(), kind, "sent"))
            data = self.profile(db, umo, persona, now)
            data.update(last_sent=now.timestamp(), awaiting=True, negative_applied=False, unanswered=data["unanswered"] + 1)
            self.ledger.save_profile(db, umo, persona, data)

    def seed_legacy(self, umo, persona, state, now):
        """Migrate once, for the matching persona only; never erase old JSON."""
        if state is None or state.persona_id != persona:
            return
        with self.ledger.transaction() as db:
            if db.execute("SELECT 1 FROM profiles WHERE umo=? AND persona=?", (umo, persona)).fetchone():
                return
            data = self.profile(db, umo, persona, now)
            for field, old in (("last_sent", state.last_proactive_at), ("last_incoming", state.last_user_message_at)):
                if old:
                    try:
                        value = datetime.fromisoformat(old)
                        if value.tzinfo is None:
                            value = value.replace(tzinfo=now.tzinfo)
                        data[field] = value.timestamp()
                    except ValueError:
                        pass
            data["unanswered"] = state.unanswered_count
            data["awaiting"] = bool(state.unanswered_count)
            self.ledger.save_profile(db, umo, persona, data)
            if state.date == now.date().isoformat():
                prefix = hashlib.sha256((umo + persona + state.date).encode()).hexdigest()
                for index in range(min(100, state.sent_count)):
                    db.execute("INSERT OR IGNORE INTO contacts VALUES (?,?,?,?,?,?)", ("legacy:" + prefix + str(index), umo, persona, now.timestamp(), "ordinary", "sent"))

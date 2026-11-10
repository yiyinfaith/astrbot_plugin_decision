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
    if re.search(
        r"(?:太吵|太烦|少(?:一点|点)?(?:主动)?(?:找我|联系|发消息)|主动.{0,8}太频繁|别.{0,4}这么频繁)",
        text,
    ):
        return "less"
    if re.search(
        r"(?:多(?:一点|点)?(?:主动)?(?:找我|联系|发消息)|主动.{0,8}太少|怎么.{0,6}不找我)", text
    ):
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

    @property
    def humanization(self):
        return self.runtime.config.get("humanization_settings", {}) or {}

    @staticmethod
    def _relationship_stage(score):
        score = max(0.0, min(100.0, float(score)))
        if score < 25:
            return "初识"
        if score < 50:
            return "熟悉"
        if score < 75:
            return "稳定"
        return "亲近"

    def _decay_relationship(self, data, now):
        data.setdefault("relationship_score", 20.0)
        data.setdefault("relationship_updated_at", now.timestamp())
        if not self.humanization.get("enable", True) or not self.humanization.get(
            "relationship_memory_enable", True
        ):
            return
        try:
            previous = float(data.get("relationship_updated_at", now.timestamp()))
            elapsed = max(0.0, now.timestamp() - previous) / 86400
            days = max(1.0, float(self.humanization.get("relationship_decay_days", 30)))
            if elapsed > 0:
                # Decay toward neutral rather than erasing the relationship.
                data["relationship_score"] = 50 + (float(data["relationship_score"]) - 50) * (
                    0.5 ** (elapsed / days)
                )
                data["relationship_updated_at"] = now.timestamp()
        except (TypeError, ValueError, OverflowError):
            data["relationship_score"] = 20.0
            data["relationship_updated_at"] = now.timestamp()

    def _relationship_delta(self, data, delta, now):
        self._decay_relationship(data, now)
        if self.humanization.get("enable", True) and self.humanization.get(
            "relationship_memory_enable", True
        ):
            data["relationship_score"] = max(
                0.0, min(100.0, float(data.get("relationship_score", 20.0)) + delta)
            )
            data["relationship_updated_at"] = now.timestamp()

    @staticmethod
    def intent_category(intent, trigger=""):
        text = str(intent or "")
        if trigger == "sleep":
            return "sleep_wakeup"
        if re.search(r"跟进|话题|进展|结果|面试|尚未聊完", text):
            return "follow_up"
        if re.search(r"想法|分享|有趣|联想", text):
            return "share"
        if re.search(r"日程|活动|生活|状态|穿搭", text):
            return "life_update"
        return "check_in"

    def intent_allowed(self, umo, persona, intent, trigger, now):
        settings = self.humanization
        if (
            trigger == "manual"
            or not settings.get("enable", True)
            or not settings.get("content_diversity_enable", True)
            or self.intent_category(intent, trigger) in {"follow_up", "sleep_wakeup"}
        ):
            return True, "allowed"
        category = self.intent_category(intent, trigger)
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            for item in data.get("recent_intents", [])[-4:]:
                if item.get("category") != category:
                    continue
                try:
                    if (
                        now.timestamp() - float(item.get("at", 0))
                        < settings.get("intent_cooldown_minutes", 240) * 60
                    ):
                        return False, "同类主动内容冷却中"
                except (TypeError, ValueError):
                    continue
        return True, "allowed"

    def record_intent(self, umo, persona, intent, trigger, now):
        category = self.intent_category(intent, trigger)
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            history = [item for item in data.get("recent_intents", []) if isinstance(item, dict)]
            history.append(
                {"category": category, "at": now.timestamp(), "trigger": str(trigger)[:40]}
            )
            data["recent_intents"] = history[-8:]
            self.ledger.save_profile(db, umo, persona, data)

    def profile(self, db, umo, persona, now):
        low, high = self.bounds(umo)
        data = self.ledger.profile(db, umo, persona, now.timestamp(), low, high)
        self._decay_relationship(data, now)
        data.setdefault("recent_intents", [])
        if (
            data["awaiting"]
            and not data["negative_applied"]
            and now.timestamp() - data["last_sent"] >= self.settings["response_window_minutes"] * 60
        ):
            data["interval"] = min(high, data["interval"] * self.settings["negative_factor"])
            data["negative_applied"] = True
        self.ledger.save_profile(db, umo, persona, data)
        return data

    def summary(self, umo, persona, now):
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            relationship = {
                "score": round(float(data.get("relationship_score", 20.0)), 1),
                "stage": self._relationship_stage(data.get("relationship_score", 20.0)),
                "updated_at": data.get("relationship_updated_at", 0),
            }
            samples = int(data.get("response_samples", 0) or 0)
            latency = float(data.get("response_latency_seconds", 0) or 0)
            rhythm = "unknown"
            if samples:
                rhythm = (
                    "fast" if latency <= 15 * 60 else "normal" if latency <= 2 * 3600 else "slow"
                )
            public = {
                k: v
                for k, v in data.items()
                if k not in {"active_hours", "active_updated", "active_last", "pending_hook"}
            }
            return {
                **public,
                "active_hours": self.active_hours_summary(data),
                "relationship": relationship,
                "response_rhythm": {
                    "bucket": rhythm,
                    "average_seconds": round(latency, 1),
                    "samples": samples,
                },
                "umo": umo,
                "persona": persona,
                "today": self.counts(db, umo, persona, now),
                "today_interval": self._day_interval(umo, persona, data, now),
                "bounds_minutes": self.bounds(umo),
            }

    @staticmethod
    def brief(summary, now):
        """What a model needs from the ledger, in minutes rather than bookkeeping."""

        moment = now.timestamp()
        data = {
            key: summary[key]
            for key in ("interval", "unanswered", "feedback", "today", "bounds_minutes")
            if summary.get(key) not in (None, "")
        }
        if "today_interval" in summary:
            data["interval"] = round(summary["today_interval"])
        if summary.get("active_hours"):
            data["active_hours"] = summary["active_hours"]
        engagement = summary.get("engagement") or {}
        age = moment - float(engagement.get("at", 0) or 0)
        if engagement.get("kind") and age < 86400:
            data["engagement"] = {
                **{
                    k: engagement[k]
                    for k in ("kind", "quote", "scope", "sender")
                    if k in engagement
                },
                "minutes_ago": round(age / 60),
            }
        paused = float(summary.get("paused_until", 0) or 0)
        if paused == -1:
            data["paused"] = "until_resumed"
        elif paused > moment:
            data["paused_minutes"] = round((paused - moment) / 60)
        defer = float(summary.get("defer_until", 0) or 0)
        if defer > moment:
            data["busy_minutes"] = round((defer - moment) / 60)
        busy = {
            member: round((float(until) - moment) / 60)
            for member, until in (summary.get("member_busy") or {}).items()
            if float(until) > moment
        }
        if busy:
            data["member_busy_minutes"] = busy
        for key, field in (
            ("minutes_since_sent", "last_sent"),
            ("minutes_since_incoming", "last_incoming"),
        ):
            if summary.get(field):
                data[key] = max(0, round((moment - float(summary[field])) / 60))
        return data

    def social(self, summary):
        """Relationship and rhythm, only while their switches allow using them."""

        human = self.humanization
        if not human.get("enable", True):
            return {}
        data = {}
        if human.get("relationship_memory_enable", True):
            data["relationship"] = summary["relationship"]["stage"]
        if human.get("response_rhythm_enable", True):
            data["response_rhythm"] = summary["response_rhythm"]["bucket"]
        return data

    @staticmethod
    def day_bounds(now):
        start = datetime.combine(now.date(), time.min, now.tzinfo)
        return start.timestamp(), (start + timedelta(days=1)).timestamp()

    def counts(self, db, umo, persona, now, exclude=""):
        start, end = self.day_bounds(now)
        counts = {"ordinary": 0, "agreed": 0}
        for row in db.execute(
            "SELECT kind,COUNT(*) AS n FROM contacts WHERE umo=? AND persona=? AND at>=? AND at<? AND status IN ('reserved','sending','sent','unknown') AND id<>? GROUP BY kind",
            (umo, persona, start, end, exclude),
        ):
            counts[row["kind"]] = row["n"]
        for row in db.execute(
            "SELECT * FROM tasks WHERE umo=? AND persona=? AND due>=? AND due<? AND status IN ('pending','preparing','ready','suspended','awaiting_delivery','sending','unknown') AND id<>?",
            (umo, persona, start, end, exclude),
        ):
            task = self.ledger.task(row)
            if db.execute(
                "SELECT 1 FROM contacts WHERE id=? AND status IN ('sent','unknown','sending')",
                (task["id"],),
            ).fetchone():
                continue
            if task.get("notify", True):
                counts["agreed" if task["origin"] == "user" else "ordinary"] += 1
        return counts

    # ----- 对方常在线的时段 -------------------------------------------------

    def _decay_hours(self, data, now):
        hours = data.get("active_hours")
        if not isinstance(hours, list) or len(hours) != 24:
            hours = [0.0] * 24
        try:
            hours = [max(0.0, float(value)) for value in hours]
        except (TypeError, ValueError):
            hours = [0.0] * 24
        half_life = max(1, int(self.humanization.get("active_hours_half_life_days", 14)))
        updated = float(data.get("active_updated", 0) or 0)
        if updated and now.timestamp() > updated:
            factor = 0.5 ** ((now.timestamp() - updated) / 86400 / half_life)
            hours = [value * factor for value in hours]
        data["active_hours"] = hours
        data["active_updated"] = now.timestamp()
        return hours

    def record_activity(self, umo, persona, now):
        """Count one inbound message toward its local hour.

        A burst of messages within a few minutes is one sitting, so later
        messages of the same burst add only a small weight.
        """

        if not self.humanization.get("enable", True) or not self.humanization.get(
            "active_hours_enable", True
        ):
            return
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            hours = self._decay_hours(data, now)
            last = float(data.get("active_last", 0) or 0)
            weight = 0.2 if last and now.timestamp() - last < 300 else 1.0
            hours[now.hour] += weight
            data["active_samples"] = int(data.get("active_samples", 0) or 0) + 1
            data["active_last"] = now.timestamp()
            self.ledger.save_profile(db, umo, persona, data)

    def _hour_ratios(self, data):
        hours = data.get("active_hours")
        if not isinstance(hours, list) or len(hours) != 24:
            return None
        total = sum(hours)
        if total <= 0:
            return None
        mean = total / 24
        # Neighbouring hours smooth a sparse histogram: someone active at 21:00
        # is plausibly around at 20:30 as well.
        return [
            (hours[(h - 1) % 24] * 0.5 + hours[h] + hours[(h + 1) % 24] * 0.5) / 2 / mean
            for h in range(24)
        ]

    def activity(self, umo, persona, now):
        """``(ratio, samples)`` for the current hour; ratio is ``None`` if unknown."""

        settings = self.humanization
        if not settings.get("enable", True) or not settings.get("active_hours_enable", True):
            return None, 0
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
        samples = int(data.get("active_samples", 0) or 0)
        ratios = self._hour_ratios(data)
        if ratios is None or samples < int(settings.get("active_hours_min_samples", 30)):
            return None, samples
        return ratios[now.hour], samples

    def next_active_at(self, umo, persona, now, *, within_hours=8):
        """Start of the next hour the person is usually around, if any soon."""

        low = float(self.humanization.get("active_hours_low_ratio", 0.35))
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
        ratios = self._hour_ratios(data)
        if ratios is None:
            return None
        start = now.replace(minute=0, second=0, microsecond=0)
        for step in range(1, within_hours + 1):
            moment = start + timedelta(hours=step)
            if ratios[moment.hour] >= low:
                return moment
        return None

    def active_hours_summary(self, data):
        ratios = self._hour_ratios(data)
        if ratios is None:
            return None
        low = float(self.humanization.get("active_hours_low_ratio", 0.35))
        return {
            "samples": int(data.get("active_samples", 0) or 0),
            "usually_away_hours": [h for h, r in enumerate(ratios) if r < low],
            "busiest_hours": sorted(range(24), key=lambda h: ratios[h], reverse=True)[:3],
        }

    # ----- 两段式开场 -------------------------------------------------------

    def set_hook(self, umo, persona, now, hook):
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            data["pending_hook"] = {**hook, "at": now.timestamp()}
            self.ledger.save_profile(db, umo, persona, data)

    def take_hook(self, umo, persona, now):
        """Return and clear an unexpired opening the person has just answered."""

        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            hook = data.pop("pending_hook", None)
            if hook is None:
                return None
            self.ledger.save_profile(db, umo, persona, data)
        if not isinstance(hook, dict) or now.timestamp() > float(hook.get("expires", 0) or 0):
            return None
        return hook

    def gate(self, umo, persona, now, *, agreed=False, exclude="", frequency=True):
        with self.ledger.transaction() as db:
            return self._gate(
                db, umo, persona, now, agreed=agreed, exclude=exclude, frequency=frequency
            )

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
            interval = (
                self._day_interval(umo, persona, data, now)
                if adaptive
                else self.runtime.policy.settings_for(umo).get("cooldown_minutes", 0)
            )
            anchor = data["last_sent"] or (data["created"] if adaptive else 0)
            if now.timestamp() < anchor + interval * 60:
                return False, "自适应主动间隔未到"
        return True, "allowed"

    def _day_interval(self, umo, persona, data, now):
        """Today's target gap. A frequency the person asked for is left exact."""

        interval = data["interval"]
        if data.get("frequency_requested") or data.get("feedback") in {"more", "less", "set"}:
            return interval
        low, high = self.bounds(umo)
        floor = max(low, data["baseline"] * 0.8)
        return max(floor, min(high, interval * self._day_factor(persona, now)))

    def _day_factor(self, persona, now):
        """Some days the persona reaches out a little more, some days a little less."""

        if not self.humanization.get("enable", True):
            return 1.0
        digest = hashlib.sha256(f"day-rhythm:{now.date()}:{persona}".encode()).digest()
        return 0.75 + 0.6 * digest[0] / 255

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
            if not db.execute(
                "INSERT OR IGNORE INTO signals VALUES (?,?,?,?)",
                (umo, persona, ident, now.timestamp()),
            ).rowcount:
                return ""
            data = self.profile(db, umo, persona, now)
            if targeted:
                data["last_incoming"] = now.timestamp()
                if data.get("last_sent", 0):
                    latency = max(0.0, now.timestamp() - float(data["last_sent"]))
                    last_sample_at = float(data.get("response_last_at", 0) or 0)
                    history_days = max(1, int(self.humanization.get("rhythm_history_days", 30)))
                    samples = int(data.get("response_samples", 0) or 0)
                    if last_sample_at and now.timestamp() - last_sample_at > history_days * 86400:
                        samples = 0
                        data["response_latency_seconds"] = 0.0
                    data["response_latency_seconds"] = (
                        (float(data.get("response_latency_seconds", 0) or 0) * samples) + latency
                    ) / (samples + 1)
                    data["response_samples"] = min(100, samples + 1)
                    data["response_last_at"] = now.timestamp()
                self._relationship_delta(data, 1.0, now)
                active_exchange = (
                    data["awaiting"]
                    and now.timestamp() - data["last_sent"]
                    <= self.settings["response_window_minutes"] * 60
                ) or (
                    data.get("last_normal_reply", 0)
                    and now.timestamp() - data["last_normal_reply"]
                    <= self.settings["response_window_minutes"] * 60
                )
                if (
                    not quality_pending
                    and active_exchange
                    and now.timestamp() - data["last_positive"]
                    >= self.settings["adjustment_cooldown_minutes"] * 60
                ):
                    low, _ = self.bounds(umo)
                    # Explicit preference remains the baseline: an enthusiastic
                    # reply cannot immediately undo a request to be quieter.
                    floor = max(low, data["baseline"] * 0.8)
                    data["interval"] = max(
                        floor, data["interval"] * self.settings["positive_factor"]
                    )
                    data["last_positive"] = now.timestamp()
                data["awaiting"] = False
                data["unanswered"] = 0
            if action and authority:
                self._feedback(data, umo, action, now)
                self._relationship_delta(
                    data, {"more": 2, "less": -2, "pause": -4, "resume": 1}.get(action, 0), now
                )
                event.set_extra("decision_contact_feedback", action)
            self.ledger.save_profile(db, umo, persona, data)
        return action if authority else ""

    def apply_quality(self, umo, persona, source, signal, now):
        """Automatic evidence only; this never grants a force-adjust privilege."""
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            ident = "quality:" + source["id"]
            if not db.execute(
                "INSERT OR IGNORE INTO signals VALUES (?,?,?,?)",
                (umo, persona, ident, now.timestamp()),
            ).rowcount:
                return
            kind = signal.get("engagement", "neutral")
            if not isinstance(kind, str) or kind not in {"engaged", "neutral", "closing", "busy"}:
                return
            data["engagement"] = {
                "kind": kind,
                "quote": signal["quote"],
                "at": now.timestamp(),
                "sender": source["sender"],
                "scope": "member" if ":GroupMessage:" in umo else "conversation",
            }
            if (
                kind == "engaged"
                and self.settings["enable"]
                and now.timestamp() - data["last_positive"]
                >= self.settings["adjustment_cooldown_minutes"] * 60
                and now.timestamp() - data.get("feedback_at", 0)
                >= self.settings["adjustment_cooldown_minutes"] * 60
            ):
                low, _ = self.bounds(umo)
                data["interval"] = max(
                    low, data["baseline"] * 0.8, data["interval"] * self.settings["positive_factor"]
                )
                data["last_positive"] = now.timestamp()
                self._relationship_delta(data, 5.0, now)
            elif kind == "neutral":
                self._relationship_delta(data, 0.5, now)
            elif kind == "closing":
                self._relationship_delta(data, -1.0, now)
            if kind in {"busy", "closing"}:
                minutes = signal.get("busy_minutes", 20 if kind == "closing" else 60)
                if (
                    isinstance(minutes, bool)
                    or not isinstance(minutes, (int, float))
                    or not math.isfinite(minutes)
                ):
                    minutes = 20
                minutes = max(
                    1,
                    min(
                        self.runtime.config["continuity_settings"]["temporary_quiet_minutes"],
                        minutes,
                    ),
                )
                until = now.timestamp() + minutes * 60
                if ":GroupMessage:" in umo:
                    members = data.setdefault("member_busy", {})
                    members[source["sender"]] = until
                    data["member_busy"] = dict(
                        sorted(members.items(), key=lambda x: x[1], reverse=True)[:50]
                    )
                else:
                    data["defer_until"] = max(data.get("defer_until", 0), until)
            self.ledger.save_profile(db, umo, persona, data)

    def normal_reply(self, event, persona, now):
        if event.get_extra("decision_life_proactive", False) or event.get_extra(
            "decision_contact_reply_recorded", False
        ):
            return
        umo = str(event.unified_msg_origin)
        with self.ledger.transaction() as db:
            data = self.profile(db, umo, persona, now)
            had_recent_inbound = data.get("last_incoming", 0) > data.get("last_normal_reply", 0)
            data["last_normal_reply"] = now.timestamp()
            if had_recent_inbound:
                self._relationship_delta(data, 1.0, now)
            self.ledger.save_profile(db, umo, persona, data)
        event.set_extra("decision_contact_reply_recorded", True)

    def _feedback(self, data, umo, action, now, interval=0, pause_until=0):
        low, high = self.bounds(umo)
        if action == "pause":
            data["paused_until"] = pause_until or -1
        elif action == "resume":
            data["paused_until"] = 0
        else:
            value = (
                interval
                if action == "set"
                else data["interval"]
                * (
                    self.settings["feedback_factor"]
                    if action == "less"
                    else 1 / self.settings["feedback_factor"]
                )
            )
            data["baseline"] = data["interval"] = min(high, max(low, value))
            data["frequency_requested"] = True
        data["feedback"], data["feedback_at"] = action, now.timestamp()

    def preference(self, event, persona, action="status", interval_minutes=0, pause_until=0):
        umo, now = str(event.unified_msg_origin), self.runtime._now()
        if action not in {"status", "more", "less", "pause", "resume", "set"}:
            raise ValueError("未知联系偏好操作")
        if action != "status":
            if event.get_extra("decision_life_proactive", False) or not group_authorized(
                event, self.settings
            ):
                raise ValueError("当前平台身份无权强制调整本群频率")
            if not self.settings["enable"] or not self.settings["preference_tool_enable"]:
                raise ValueError("联系偏好调整已关闭")
            if (
                action == "set"
                and not self.bounds(umo)[0] <= interval_minutes <= self.bounds(umo)[1]
            ):
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
            db.execute(
                "INSERT OR REPLACE INTO contacts VALUES (?,?,?,?,?,?)",
                (ident, umo, persona, now.timestamp(), kind, "sent"),
            )
            data = self.profile(db, umo, persona, now)
            data.update(
                last_sent=now.timestamp(),
                awaiting=True,
                negative_applied=False,
                unanswered=data["unanswered"] + 1,
            )
            self.ledger.save_profile(db, umo, persona, data)

    def seed_legacy(self, umo, persona, state, now):
        """Migrate once, for the matching persona only; never erase old JSON."""
        if state is None or state.persona_id != persona:
            return
        with self.ledger.transaction() as db:
            if db.execute(
                "SELECT 1 FROM profiles WHERE umo=? AND persona=?", (umo, persona)
            ).fetchone():
                return
            data = self.profile(db, umo, persona, now)
            for field, old in (
                ("last_sent", state.last_proactive_at),
                ("last_incoming", state.last_user_message_at),
            ):
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
                    db.execute(
                        "INSERT OR IGNORE INTO contacts VALUES (?,?,?,?,?,?)",
                        (
                            "legacy:" + prefix + str(index),
                            umo,
                            persona,
                            now.timestamp(),
                            "ordinary",
                            "sent",
                        ),
                    )

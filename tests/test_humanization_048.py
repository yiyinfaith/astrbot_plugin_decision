"""0.4.8: thresholds, reply-delay wording and sleep, active hours, weather, usage, audit."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from decision import usage_meter
from decision.life_config import life_defaults, normalize_life
from decision.life_contacts import ContactControl
from decision.life_ledger import LifeLedger
from decision.life_weather import WEATHER_WORDS, WeatherService, random_day
from decision.virtual_life.core.models import DailyPlan, TimelineItem
from decision.virtual_life.core.reply_delay import (
    DelayBatch,
    ReplyDelayPolicy,
    format_delay_context,
)

UMO = "qq:FriendMessage:42"


# ----- configuration -------------------------------------------------------


def test_old_single_threshold_keeps_relevance_and_loosens_timing():
    raw = life_defaults()
    raw["integration_settings"] = {"jev_threshold": 0.82}
    out = normalize_life(raw)
    assert out["integration_settings"]["jev_threshold"] == 0.82
    assert out["integration_settings"]["jev_relevant_threshold"] == 0.82
    assert out["integration_settings"]["jev_timing_threshold"] == 0.5


def test_explicit_thresholds_are_kept_and_validated():
    raw = life_defaults()
    raw["integration_settings"].update(jev_relevant_threshold=0.3, jev_timing_threshold=0.9)
    out = normalize_life(raw)
    assert out["integration_settings"]["jev_relevant_threshold"] == 0.3
    assert out["integration_settings"]["jev_timing_threshold"] == 0.9
    raw["integration_settings"]["jev_timing_threshold"] = 1.2
    with pytest.raises(ValueError, match="阈值"):
        normalize_life(raw)


@pytest.mark.parametrize(
    ("group", "key", "value"),
    [
        ("humanization_settings", "short_opening_probability", 0.8),
        ("humanization_settings", "two_stage_window_minutes", 0),
        ("humanization_settings", "active_hours_low_ratio", 0),
        ("adaptive_settings", "floor_threshold", 1.5),
        ("weather_settings", "mode", "sunny"),
        ("weather_settings", "cache_minutes", 0),
        ("reply_delay_settings", "max_sleep_delay_seconds", 90000),
        ("reply_delay_settings", "notification_style", "robot"),
    ],
)
def test_new_settings_reject_bad_values(group, key, value):
    raw = life_defaults()
    raw[group][key] = value
    if key == "short_opening_probability":
        raw[group]["two_stage_probability"] = 0.5
    with pytest.raises(ValueError):
        normalize_life(raw)


def test_new_defaults_keep_ordinary_replies_immediate():
    defaults = life_defaults()
    assert defaults["reply_delay_settings"]["enable"] is False
    assert defaults["weather_settings"]["mode"] == "off"
    assert defaults["adaptive_settings"]["floor_threshold"] == 0.4


# ----- reply delay -----------------------------------------------------------


def _item(state="busy", availability="low", start="09:00", end="10:00"):
    return TimelineItem(
        id=f"x{start}",
        start=start,
        end=end,
        activity="开会",
        state=state,
        availability=availability,
    )


def test_sleep_waits_until_wake_and_never_announces_it():
    settings = life_defaults()["reply_delay_settings"]
    policy = ReplyDelayPolicy(settings)
    now = datetime(2026, 11, 3, 2, 0)
    decision = policy.decide(_item("sleep", "blocked", "00:00", "07:00"), now, 5)
    assert decision.sleeping
    assert decision.delay_seconds == 5 * 3600
    assert not policy.should_notify(decision)
    # The cap bounds a very long sleep, and continuing sleep counts.
    settings["max_sleep_delay_seconds"] = 3600
    decision = ReplyDelayPolicy(settings).decide(
        _item("sleep", "blocked", "23:00", "24:00"),
        datetime(2026, 11, 3, 23, 30),
        5,
        sleep_remaining=8 * 3600,
    )
    assert decision.delay_seconds == 3600


def test_busy_notice_needs_a_long_enough_wait():
    settings = life_defaults()["reply_delay_settings"]
    settings["delay_formulas"]["low"] = "60"
    policy = ReplyDelayPolicy(settings)
    decision = policy.decide(_item(), datetime(2026, 11, 3, 9, 0), 5)
    assert decision.delay_seconds == 60 and not policy.should_notify(decision)
    settings["delay_formulas"]["low"] = "600"
    decision = ReplyDelayPolicy(settings).decide(_item(), datetime(2026, 11, 3, 9, 0), 5)
    assert ReplyDelayPolicy(settings).should_notify(decision)
    assert ReplyDelayPolicy(settings).notification(decision) == "在忙，晚点回你"
    settings["notification_template"] = "{public_reason}，{delay_minutes} 分钟后回你"
    assert ReplyDelayPolicy(settings).notification(decision) == "在忙，10 分钟后回你"


def test_sleeping_batch_tells_the_agent_it_just_woke_up():
    policy = ReplyDelayPolicy(life_defaults()["reply_delay_settings"])
    now = datetime(2026, 11, 3, 2, 0)
    decision = policy.decide(_item("sleep", "blocked", "00:00", "07:00"), now, 5)
    batch = DelayBatch(umo=UMO, created_at=now, deadline=now, decision=decision)
    assert "刚醒来" in format_delay_context(batch, now + timedelta(hours=5), None)


def test_sleep_span_follows_midnight_into_tomorrows_plan():
    from decision.virtual_life.engine import VirtualLifeBase

    today = DailyPlan(
        date="2026-11-03",
        persona_id="p",
        theme="",
        mood="",
        outfit=None,
        timeline=(
            _item("available", "high", "00:00", "23:00"),
            _item("sleep", "blocked", "23:00", "24:00"),
        ),
    )
    tomorrow = DailyPlan(
        date="2026-11-04",
        persona_id="p",
        theme="",
        mood="",
        outfit=None,
        timeline=(
            _item("sleep", "blocked", "00:00", "07:30"),
            _item("available", "high", "07:30", "24:00"),
        ),
    )
    plans = {"2026-11-04": tomorrow}
    fake = SimpleNamespace(storage=SimpleNamespace(get_plan=lambda day, persona: plans.get(day)))
    seconds = VirtualLifeBase._sleep_remaining_seconds(
        fake, today, today.timeline[1], datetime(2026, 11, 3, 23, 30)
    )
    assert seconds == 8 * 3600
    plans.clear()
    assert (
        VirtualLifeBase._sleep_remaining_seconds(
            fake, today, today.timeline[1], datetime(2026, 11, 3, 23, 30)
        )
        == 1800
    )


# ----- active hours and opening hooks -----------------------------------------


def _contacts(tmp_path, **human):
    config = life_defaults()
    config["humanization_settings"].update(human)
    runtime = SimpleNamespace(
        config=config,
        _now=lambda: datetime(2026, 11, 3, 12, 0).astimezone(),
        policy=SimpleNamespace(settings_for=lambda umo: config["friend_settings"]),
    )
    return ContactControl(runtime, LifeLedger(tmp_path / "ledger.sqlite3"))


def test_active_hours_need_enough_samples_and_find_the_next_usual_hour(tmp_path):
    contacts = _contacts(tmp_path, active_hours_min_samples=10)
    base = datetime(2026, 11, 1, 20, 0).astimezone()
    for day in range(3):
        for minute in range(0, 60, 10):
            contacts.record_activity(UMO, "p", base + timedelta(days=day, minutes=minute))
    now = datetime(2026, 11, 4, 3, 0).astimezone()
    ratio, samples = contacts.activity(UMO, "p", now)
    assert samples == 18 and ratio is not None and ratio < 0.35
    assert contacts.activity(UMO, "p", now.replace(hour=20))[0] > 1
    assert contacts.next_active_at(UMO, "p", now, within_hours=24).hour == 19
    summary = contacts.summary(UMO, "p", now)
    assert "active_hours" in summary and 3 in summary["active_hours"]["usually_away_hours"]
    assert "pending_hook" not in summary


def test_burst_messages_count_as_one_sitting(tmp_path):
    contacts = _contacts(tmp_path)
    now = datetime(2026, 11, 3, 9, 0).astimezone()
    for second in range(0, 50, 10):
        contacts.record_activity(UMO, "p", now + timedelta(seconds=second))
    with contacts.ledger.transaction() as db:
        data = contacts.profile(db, UMO, "p", now)
    assert data["active_hours"][9] == pytest.approx(1.8, abs=0.01)


def test_active_hours_respect_the_switch(tmp_path):
    contacts = _contacts(tmp_path, active_hours_enable=False, active_hours_min_samples=1)
    now = datetime(2026, 11, 3, 9, 0).astimezone()
    contacts.record_activity(UMO, "p", now)
    assert contacts.activity(UMO, "p", now) == (None, 0)


def test_opening_hook_expires(tmp_path):
    contacts = _contacts(tmp_path)
    now = datetime(2026, 11, 3, 9, 0).astimezone()
    contacts.set_hook(UMO, "p", now, {"intent": "分享", "expires": now.timestamp() + 60})
    assert contacts.take_hook(UMO, "p", now + timedelta(minutes=5)) is None
    contacts.set_hook(UMO, "p", now, {"intent": "分享", "expires": now.timestamp() + 600})
    assert contacts.take_hook(UMO, "p", now + timedelta(minutes=5))["intent"] == "分享"


# ----- weather ---------------------------------------------------------------


def test_random_weather_is_stable_per_day_and_seasonal():
    winter = random_day(date(2026, 1, 15), "temperate")
    assert winter == random_day(date(2026, 1, 15), "temperate")
    summer = random_day(date(2026, 7, 15), "temperate")
    assert summer["high"] > winter["high"] + 15
    assert random_day(date(2026, 1, 15), "tropical")["low"] > 10


def test_weather_words_ignore_unrelated_chat():
    assert WEATHER_WORDS.search("明天天气怎么样")
    assert WEATHER_WORDS.search("今天好冷啊")
    assert not WEATHER_WORDS.search("他很热情")
    assert not WEATHER_WORDS.search("冷静一下")


class _WeatherPlugin:
    api_key = "k"
    default_city = "上海"

    def __init__(self):
        self.calls = []

    async def get_current_weather_by_city(self, city):
        self.calls.append(("now", city))
        return {
            "city": city,
            "desc": "小雨",
            "temp": "12",
            "feels_like": "10",
            "humidity": "90",
            "wind_speed": "8",
        }

    async def get_forecast_weather_by_city(self, city):
        self.calls.append(("forecast", city))
        return [
            {
                "date": "2026-11-03",
                "text_day": "小雨",
                "text_night": "阴",
                "high": "14",
                "low": "9",
                "humidity": "90",
                "wind_speed": "8",
            },
            {
                "date": "2026-11-04",
                "text_day": "晴",
                "text_night": "晴",
                "high": "18",
                "low": "8",
                "humidity": "50",
                "wind_speed": "5",
            },
        ]


def _weather(meta, **settings):
    config = life_defaults()
    config["weather_settings"].update(settings)
    stars = {"astrbot_plugin_weather": meta} if meta else {}
    runtime = SimpleNamespace(
        config=config, context=SimpleNamespace(get_registered_star=lambda name: stars.get(name))
    )
    return WeatherService(runtime)


def test_city_weather_uses_the_weather_plugin_and_caches():
    plugin = _WeatherPlugin()
    service = _weather(SimpleNamespace(star_cls=plugin, activated=True), mode="city")
    text = asyncio.run(service.now_text(datetime(2026, 11, 3, 15, 0)))
    assert text.startswith("上海当前小雨，12℃（体感 10℃），湿度 90%，风速 8 km/h")
    assert "9～14℃" in text
    day = asyncio.run(service.for_day(date(2026, 11, 4)))
    assert day.startswith("上海：晴，8～18℃")
    asyncio.run(service.now_text(datetime(2026, 11, 3, 15, 5)))
    assert plugin.calls.count(("now", "上海")) == 1
    assert plugin.calls.count(("forecast", "上海")) == 1
    assert service.status()["plugin"] == "可用"


@pytest.mark.parametrize(
    ("meta", "message"),
    [
        (None, "未安装"),
        (SimpleNamespace(star_cls=None, activated=False), "未启用"),
        (
            SimpleNamespace(star_cls=SimpleNamespace(api_key="", default_city="x"), activated=True),
            "API Key",
        ),
    ],
)
def test_city_weather_explains_a_missing_link(meta, message):
    service = _weather(meta, mode="city", city="北京")
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(service.now_text(datetime(2026, 11, 3, 15, 0)))
    assert message in service.status()["plugin"]


def test_off_mode_never_touches_the_plugin():
    service = _weather(None)
    assert asyncio.run(service.now_text(datetime(2026, 11, 3, 15, 0))) == ""
    assert asyncio.run(service.for_day(date(2026, 11, 3))) == ""


# ----- usage meter -----------------------------------------------------------


def test_usage_meter_counts_estimates_real_usage_and_persists(tmp_path):
    path = tmp_path / "usage.json"
    meter = usage_meter.UsageMeter(path)
    meter.record("Jev·主动发起", prompt="你好" * 10, output="ok")
    meter.record("Jev·主动发起", prompt="x", output="", ok=False)
    response = SimpleNamespace(usage=SimpleNamespace(input=120, output=30), completion_text="hi")
    meter.record("日程与穿搭", response=response, prompt="ignored")
    meter.record("多模态理解", prompt="看图", output="一只猫", extra_input=260)
    meter.save()
    today = usage_meter.UsageMeter(path).summary(7)
    rows = today["days"][today["today"]]
    assert rows["Jev·主动发起"]["calls"] == 2 and rows["Jev·主动发起"]["failures"] == 1
    assert rows["日程与穿搭"]["input_tokens"] == 120 and rows["日程与穿搭"]["estimated"] == 0
    assert rows["多模态理解"]["input_tokens"] >= 260
    assert len(today["days"]) == 7


def test_usage_record_never_raises():
    class Broken:
        def record(self, *a, **k):
            raise RuntimeError("disk full")

    original = usage_meter.METER
    usage_meter.install(Broken())
    try:
        usage_meter.record("x", prompt="y")
    finally:
        usage_meter.install(original)


# ----- persistent decision records ---------------------------------------------


def test_audit_records_survive_and_are_pruned(tmp_path):
    ledger = LifeLedger(tmp_path / "ledger.sqlite3")
    now = datetime(2026, 11, 3, 12, 0).timestamp()
    ledger.add_audit(UMO, now - 20 * 86400, False, {"reason": "old"})
    for index in range(3):
        ledger.add_audit(UMO, now + index, index == 2, {"reason": f"r{index}", "umo": UMO})
    ledger.add_audit("other", now, False, {"reason": "other"})
    reopened = LifeLedger(tmp_path / "ledger.sqlite3")
    assert [row["reason"] for row in reopened.audits(UMO)] == ["old", "r0", "r1", "r2"]
    reopened.prune(now)
    assert [row["reason"] for row in reopened.audits(UMO)] == ["r0", "r1", "r2"]
    assert [row["reason"] for row in reopened.audits(UMO, limit=1)] == ["r2"]


# ----- dashboard -------------------------------------------------------------


def test_dashboard_exposes_presets_usage_and_new_fields():
    root = Path(__file__).resolve().parents[1] / "views" / "dashboard"
    app = (root / "app.js").read_text(encoding="utf-8")
    html = (root / "index.html").read_text(encoding="utf-8")
    workspace = (root / "life-workspace.js").read_text(encoding="utf-8")
    for preset in ("restrained", "normal", "clingy"):
        assert preset in app
    assert "applyLifePreset(id)" in html and "usage-table" in html and "lifeAuditRows" in html
    for key in (
        "jev_timing_threshold",
        "floor_threshold",
        "two_stage_probability",
        "active_hours_enable",
        "notification_style",
        "sleep_until_wake",
        "weather_settings",
    ):
        assert key in workspace
    for css in ("style.css", "liquid.css"):
        assert ".usage-table" in (root / css).read_text(encoding="utf-8")


def test_normal_preset_matches_the_plugin_defaults():
    import re

    app = (Path(__file__).resolve().parents[1] / "views" / "dashboard" / "app.js").read_text(
        encoding="utf-8"
    )
    block = re.search(r"normal: \{label:'正常'.*?values:\{(.*?)\}\}\},", app, re.S).group(1)
    defaults = life_defaults()
    for group, body in re.findall(r"(\w+_settings):\{(.*?)\}(?:,|$)", block):
        for key, value in re.findall(r"(\w+):([\d.]+)", body):
            if key in defaults[group]:
                assert float(defaults[group][key]) == float(value), (group, key)
    assert defaults["delivery_settings"]["availability_probabilities"] == {
        "blocked": 0.0,
        "low": 0.25,
        "normal": 0.7,
        "high": 1.0,
    }


def test_missing_free_tier_fields_are_left_out():
    class FreeTier(_WeatherPlugin):
        async def get_current_weather_by_city(self, city):
            return {
                "city": city,
                "desc": "阴",
                "temp": "19",
                "feels_like": "19",
                "humidity": "0",
                "wind_speed": "0",
            }

    service = _weather(SimpleNamespace(star_cls=FreeTier(), activated=True), mode="city")
    text = asyncio.run(service.now_text(datetime(2026, 11, 3, 15, 0)))
    assert text.startswith("上海当前阴，19℃；今天")
    assert "湿度 0" not in text and "风速 0" not in text and "体感" not in text

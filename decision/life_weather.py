"""Weather for the virtual life: a seasonal random weather or a real city.

The city mode is a plugin link: it calls the public coroutine methods of the
installed ``astrbot_plugin_omni`` (综合插件) instance, so the API key stays in
that plugin's own configuration and is never copied here.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from datetime import date, datetime
from typing import Any

WEATHER_PLUGIN = "astrbot_plugin_omni"
WEATHER_WORDS = re.compile(
    r"天气|下雨|下雪|雨伞|带伞|晴天|阴天|多云|气温|温度|几度|降温|升温|好冷|好热|太冷|太热"
    r"|冷不冷|热不热|穿什么|穿多少|雾霾|台风|刮风|风好大|闷热"
)

# Monthly (low, high) temperature in ℃ and the chance of rain or snow, by
# climate. Values are coarse on purpose: they only need to read plausibly.
_CLIMATES: dict[str, list[tuple[int, int, float]]] = {
    "temperate": [
        (-8, 2, 0.08),
        (-5, 6, 0.1),
        (1, 13, 0.15),
        (8, 21, 0.2),
        (14, 27, 0.25),
        (19, 31, 0.35),
        (22, 31, 0.5),
        (21, 30, 0.45),
        (15, 26, 0.3),
        (8, 19, 0.2),
        (0, 10, 0.12),
        (-6, 3, 0.08),
    ],
    "subtropical": [
        (2, 9, 0.3),
        (4, 11, 0.35),
        (8, 15, 0.4),
        (13, 21, 0.4),
        (18, 26, 0.45),
        (22, 29, 0.6),
        (26, 34, 0.45),
        (26, 33, 0.45),
        (22, 29, 0.35),
        (16, 24, 0.25),
        (10, 18, 0.25),
        (4, 12, 0.25),
    ],
    "tropical": [
        (17, 24, 0.15),
        (18, 25, 0.2),
        (21, 28, 0.25),
        (24, 31, 0.35),
        (26, 33, 0.5),
        (27, 33, 0.6),
        (27, 34, 0.55),
        (27, 33, 0.6),
        (26, 33, 0.55),
        (24, 31, 0.35),
        (21, 28, 0.2),
        (18, 25, 0.15),
    ],
    "cold": [
        (-24, -12, 0.15),
        (-20, -7, 0.12),
        (-10, 2, 0.12),
        (0, 12, 0.15),
        (8, 20, 0.2),
        (14, 25, 0.3),
        (17, 28, 0.4),
        (15, 26, 0.35),
        (7, 19, 0.25),
        (-1, 10, 0.15),
        (-12, -2, 0.15),
        (-21, -10, 0.15),
    ],
}
_DRY = ("晴", "晴", "多云", "多云", "阴")
_WET_WARM = ("小雨", "小雨", "中雨", "阵雨", "雷阵雨")
_WET_COLD = ("小雪", "小雪", "中雪", "雨夹雪")


def random_day(target: date, climate: str) -> dict[str, Any]:
    """Seasonal weather for one day, identical every time it is asked for."""

    table = _CLIMATES.get(climate) or _CLIMATES["temperate"]
    low, high, wet = table[target.month - 1]
    seed = int(hashlib.sha256(f"{target.isoformat()}:{climate}".encode()).hexdigest()[:12], 16)
    roll = random.Random(seed)
    shift = roll.randint(-3, 3)
    low, high = low + shift, high + shift + roll.randint(-1, 1)
    cold = high <= 2

    def condition() -> str:
        if roll.random() < wet:
            return roll.choice(_WET_COLD if cold else _WET_WARM)
        return roll.choice(_DRY)

    day = condition()
    night = day if roll.random() < 0.6 else condition()
    if day.endswith(("雨", "雪")):
        high -= roll.randint(1, 3)
    return {
        "date": target.isoformat(),
        "text_day": day,
        "text_night": night,
        "high": high,
        "low": min(low, high - 2),
        "humidity": roll.randint(65, 95) if day.endswith(("雨", "雪")) else roll.randint(30, 70),
        "wind_speed": roll.choice((5, 8, 10, 12, 15, 20)),
    }


def _known(value: Any) -> bool:
    """The free Seniverse tier leaves some fields out; the weather plugin fills ``"0"``."""

    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _extras(data: dict[str, Any]) -> str:
    parts = []
    if _known(data.get("humidity")):
        parts.append(f"湿度 {data['humidity']}%")
    if _known(data.get("wind_speed")):
        parts.append(f"风速 {data['wind_speed']} km/h")
    return "".join("，" + part for part in parts)


def _day_text(day: dict[str, Any], place: str = "") -> str:
    text_day, text_night = day.get("text_day", "未知"), day.get("text_night", "未知")
    sky = text_day if text_day == text_night else f"白天{text_day}，夜间{text_night}"
    head = f"{place}：" if place else ""
    humidity = f"，湿度 {day['humidity']}%" if _known(day.get("humidity")) else ""
    return f"{head}{sky}，{day.get('low')}～{day.get('high')}℃{humidity}"


def _random_now(day: dict[str, Any], moment: datetime) -> str:
    low, high = float(day["low"]), float(day["high"])
    hour = moment.hour + moment.minute / 60
    # Coldest around 05:00, warmest around 14:00.
    if 5 <= hour <= 14:
        temp = low + (high - low) * (hour - 5) / 9
    else:
        since = (hour - 14) % 24
        temp = high - (high - low) * since / 15
    sky = day["text_day"] if 6 <= moment.hour < 18 else day["text_night"]
    return f"当前{sky}，约 {round(temp)}℃"


class WeatherService:
    def __init__(self, runtime):
        self.runtime = runtime
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self.last_error = ""
        self.last_ok = 0.0

    @property
    def settings(self) -> dict[str, Any]:
        return self.runtime.config["weather_settings"]

    # ----- city mode --------------------------------------------------------

    def _plugin(self):
        context = self.runtime.context
        getter = getattr(context, "get_registered_star", None)
        meta = getter(WEATHER_PLUGIN) if callable(getter) else None
        if meta is None:
            raise RuntimeError("未安装综合插件 astrbot_plugin_omni")
        instance = getattr(meta, "star_cls", None)
        if not getattr(meta, "activated", True) or instance is None:
            raise RuntimeError("综合插件 astrbot_plugin_omni 未启用")
        if not getattr(instance, "api_key", ""):
            raise RuntimeError("综合插件 astrbot_plugin_omni 的天气功能未开启或未配置 API Key")
        return instance

    def city(self) -> str:
        configured = str(self.settings.get("city") or "").strip()
        if configured:
            return configured
        try:
            return str(getattr(self._plugin(), "default_city", "") or "").strip()
        except RuntimeError:
            return ""

    async def _cached(self, key: tuple[str, str], ttl: float, loader):
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
        try:
            value = await loader()
        except Exception as exc:
            self.last_error = str(exc)[:200]
            raise
        if value is None:
            self.last_error = f"综合插件没有返回「{key[1]}」的天气"
            raise RuntimeError(self.last_error)
        self._cache[key] = (time.monotonic(), value)
        if len(self._cache) > 64:
            oldest = sorted(self._cache.items(), key=lambda item: item[1][0])[:16]
            for old, _ in oldest:
                self._cache.pop(old, None)
        self.last_error, self.last_ok = "", time.time()
        return value

    async def _forecast(self, city: str):
        plugin = self._plugin()
        return await self._cached(
            ("forecast", city), 6 * 3600, lambda: plugin.get_forecast_weather_by_city(city)
        )

    async def _current(self, city: str):
        plugin = self._plugin()
        ttl = max(1, int(self.settings.get("cache_minutes", 60))) * 60
        return await self._cached(
            ("now", city), ttl, lambda: plugin.get_current_weather_by_city(city)
        )

    # ----- public -----------------------------------------------------------

    async def for_day(self, target: date) -> str:
        mode = self.settings.get("mode", "off")
        if mode == "random":
            return _day_text(random_day(target, self.settings.get("random_climate", "temperate")))
        if mode != "city":
            return ""
        city = self.city()
        if not city:
            raise RuntimeError("没有可用的城市：请填写城市，或在综合插件的天气设置里填写默认城市")
        days = await self._forecast(city)
        if not isinstance(days, list) or not days:
            return ""
        day = next((d for d in days if d.get("date") == target.isoformat()), days[0])
        return _day_text(day, city)

    async def now_text(self, moment: datetime) -> str:
        mode = self.settings.get("mode", "off")
        if mode == "random":
            day = random_day(moment.date(), self.settings.get("random_climate", "temperate"))
            return f"今天{_day_text(day)}；{_random_now(day, moment)}"
        if mode != "city":
            return ""
        city = self.city()
        if not city:
            raise RuntimeError("没有可用的城市：请填写城市，或在综合插件的天气设置里填写默认城市")
        now = await self._current(city)
        feels = now.get("feels_like")
        text = f"{city}当前{now.get('desc', '未知')}，{now.get('temp')}℃"
        if feels not in (None, "", now.get("temp")):
            text += f"（体感 {feels}℃）"
        text += _extras(now)
        try:
            days = await self._forecast(city)
        except Exception:
            days = None
        if isinstance(days, list) and days:
            day = next((d for d in days if d.get("date") == moment.date().isoformat()), None)
            if day:
                text += "；今天" + _day_text(day)
        return text

    def status(self) -> dict[str, Any]:
        mode = self.settings.get("mode", "off")
        result: dict[str, Any] = {"mode": mode}
        if mode == "city":
            try:
                self._plugin()
                result["plugin"] = "可用"
            except RuntimeError as exc:
                result["plugin"] = str(exc)
            result["city"] = self.city()
        if self.last_error:
            result["last_error"] = self.last_error
        if self.last_ok:
            result["last_ok"] = datetime.fromtimestamp(self.last_ok).isoformat(timespec="seconds")
        return result

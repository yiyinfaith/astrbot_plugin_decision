"""0.4.9: city weather links to the omni plugin (astrbot_plugin_omni) only."""

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from decision.life_config import life_defaults
from decision.life_weather import WEATHER_PLUGIN, WeatherService


class _Plugin:
    def __init__(self, label, api_key="k", default_city="上海"):
        self.label = label
        self.api_key = api_key
        self.default_city = default_city

    async def get_current_weather_by_city(self, city):
        return {"city": city, "desc": self.label, "temp": "20", "feels_like": "20", "humidity": "50", "wind_speed": "3"}

    async def get_forecast_weather_by_city(self, city):
        return [{"date": "2026-11-04", "text_day": self.label, "text_night": "阴", "high": "22", "low": "15",
                 "humidity": "60", "wind_speed": "5"}]


def _service(stars, **settings):
    config = life_defaults()
    config["weather_settings"].update({"mode": "city", **settings})
    runtime = SimpleNamespace(config=config, context=SimpleNamespace(get_registered_star=lambda name: stars.get(name)))
    return WeatherService(runtime)


def _meta(plugin, activated=True):
    return SimpleNamespace(star_cls=plugin, activated=activated)


def test_links_to_omni_and_uses_its_default_city():
    assert WEATHER_PLUGIN == "astrbot_plugin_omni"
    service = _service({"astrbot_plugin_omni": _meta(_Plugin("晴", default_city="杭州"))})
    assert service.city() == "杭州"
    assert asyncio.run(service.now_text(datetime(2026, 11, 4, 15, 0))).startswith("杭州当前晴")
    assert service.status()["plugin"] == "可用"


def test_old_weather_plugin_is_ignored():
    service = _service({"astrbot_plugin_weather": _meta(_Plugin("旧"))}, city="北京")
    with pytest.raises(RuntimeError, match="未安装综合插件"):
        asyncio.run(service.now_text(datetime(2026, 11, 4, 15, 0)))


@pytest.mark.parametrize(
    ("stars", "message"),
    [
        ({}, "未安装综合插件 astrbot_plugin_omni"),
        ({"astrbot_plugin_omni": _meta(None, activated=False)}, "astrbot_plugin_omni 未启用"),
        ({"astrbot_plugin_omni": _meta(_Plugin("x", api_key=""))}, "天气功能未开启或未配置 API Key"),
    ],
)
def test_omni_link_errors(stars, message):
    service = _service(stars, city="北京")
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(service.now_text(datetime(2026, 11, 4, 15, 0)))
    assert message in service.status()["plugin"]

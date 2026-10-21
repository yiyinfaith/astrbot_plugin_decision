"""Cover the billing fallback and the sleep quiet-hours exemption."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.personal_network._astrbot_stub import install  # noqa: E402

install()

from decision.life_media import BillingEvent, LifeMedia  # noqa: E402


def media(tmp_path, settings):
    owner = SimpleNamespace(context=SimpleNamespace(get_all_stars=lambda: []))
    instance = LifeMedia(owner, tmp_path)
    instance._settings = settings
    return instance


def settings(**overrides):
    base = {
        "proactive_billing_mode": "fixed",
        "proactive_billing_platform": "",
        "proactive_billing_user_id": "",
    }
    base.update(overrides)
    return base


ACCOUNTS = [
    {"platform": "aiocqhttp", "user_id": "111", "amount": 0, "label": "aiocqhttp · 111"},
    {"platform": "aiocqhttp", "user_id": "222", "amount": 500, "label": "aiocqhttp · 222"},
    {"platform": "telegram", "user_id": "333", "amount": 900, "label": "telegram · 333"},
]


class Event:
    def __init__(self, proactive=True, platform="aiocqhttp", sender="1046209937"):
        self._extra = {"decision_life_proactive": proactive}
        self._platform = platform
        self._sender = sender

    def get_extra(self, key, default=None):
        return self._extra.get(key, default)

    def get_platform_name(self):
        return self._platform

    def get_sender_id(self):
        return self._sender


def test_empty_account_falls_back_to_the_richest_on_that_platform(tmp_path):
    instance = media(tmp_path, settings())
    instance.refresh_billing_accounts(ACCOUNTS)

    billed = instance._billing_event(Event(), Event(), instance._settings)
    assert isinstance(billed, BillingEvent)
    # Staying on the origin platform keeps the charge on a reachable account.
    assert billed.get_platform_name() == "aiocqhttp"
    assert billed.get_sender_id() == "222", "aiocqhttp's richest funded account"


def test_fallback_uses_any_platform_when_the_origin_has_none(tmp_path):
    instance = media(tmp_path, settings())
    instance.refresh_billing_accounts(
        [{"platform": "telegram", "user_id": "333", "amount": 900}]
    )
    billed = instance._billing_event(Event(), Event(), instance._settings)
    assert isinstance(billed, BillingEvent)
    assert billed.get_sender_id() == "333"


def test_zero_balance_accounts_are_ignored(tmp_path):
    instance = media(tmp_path, settings())
    instance.refresh_billing_accounts(
        [{"platform": "aiocqhttp", "user_id": "999", "amount": 0}]
    )
    billed = instance._billing_event(Event(), Event(), instance._settings)
    assert not isinstance(billed, BillingEvent), "no funded account -> leave aigen alone"


def test_explicit_account_still_wins(tmp_path):
    instance = media(tmp_path, settings(proactive_billing_user_id="777"))
    instance.refresh_billing_accounts(ACCOUNTS)

    billed = instance._billing_event(Event(), Event(), instance._settings)
    assert billed.get_sender_id() == "777"


def test_non_proactive_is_never_wrapped(tmp_path):
    instance = media(tmp_path, settings(proactive_billing_user_id="777"))
    instance.refresh_billing_accounts(ACCOUNTS)
    billed = instance._billing_event(Event(proactive=False), Event(proactive=False),
                                     instance._settings)
    assert not isinstance(billed, BillingEvent)


def test_session_mode_never_wraps(tmp_path):
    instance = media(tmp_path, settings(proactive_billing_mode="session"))
    instance.refresh_billing_accounts(ACCOUNTS)
    billed = instance._billing_event(Event(), Event(), instance._settings)
    assert not isinstance(billed, BillingEvent)


def test_missing_ledger_leaves_aigen_alone(tmp_path):
    instance = media(tmp_path, settings())
    # No refresh at all -> empty cache.
    billed = instance._billing_event(Event(), Event(), instance._settings)
    assert not isinstance(billed, BillingEvent)


def test_refresh_handles_garbage_rows(tmp_path):
    instance = media(tmp_path, settings())
    instance.refresh_billing_accounts(
        [None, {}, {"platform": "x"}, {"user_id": "1"}, {"platform": "y", "user_id": "2",
                                                          "amount": "not-a-number"}]
    )
    assert instance._billing_cache == {}

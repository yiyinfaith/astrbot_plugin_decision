"""Cover the billing fallback and the sleep quiet-hours exemption."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.personal_network._astrbot_stub import install  # noqa: E402

install()

from decision.life_media import BillingConfigurationError, BillingEvent, LifeMedia  # noqa: E402


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


def test_empty_account_refuses_automatic_selection(tmp_path):
    instance = media(tmp_path, settings())
    instance.refresh_billing_accounts(ACCOUNTS)

    try:
        instance._billing_event(Event(), Event(), instance._settings)
    except BillingConfigurationError as exc:
        assert "明确填写" in str(exc)
    else:
        raise AssertionError("an unspecified paid account must be refused")


def test_accounts_on_another_platform_are_never_selected(tmp_path):
    instance = media(tmp_path, settings())
    instance.refresh_billing_accounts(
        [{"platform": "telegram", "user_id": "333", "amount": 900}]
    )
    try:
        instance._billing_event(Event(), Event(), instance._settings)
    except BillingConfigurationError:
        pass
    else:
        raise AssertionError("an account must never be selected from the ledger")


def test_zero_balance_account_is_not_relevant_to_selection(tmp_path):
    instance = media(tmp_path, settings())
    instance.refresh_billing_accounts(
        [{"platform": "aiocqhttp", "user_id": "999", "amount": 0}]
    )
    try:
        instance._billing_event(Event(), Event(), instance._settings)
    except BillingConfigurationError:
        pass
    else:
        raise AssertionError("the ledger must not supply a billing identity")


def test_explicit_account_still_wins(tmp_path):
    instance = media(
        tmp_path,
        settings(proactive_billing_platform="aiocqhttp", proactive_billing_user_id="777"),
    )
    instance.refresh_billing_accounts(ACCOUNTS)

    billed = instance._billing_event(Event(), Event(), instance._settings)
    assert billed.get_sender_id() == "777"


def test_non_proactive_is_never_wrapped(tmp_path):
    instance = media(tmp_path, settings(proactive_billing_user_id="777"))
    instance.refresh_billing_accounts(ACCOUNTS)
    billed = instance._billing_event(Event(proactive=False), Event(proactive=False),
                                     instance._settings)
    assert not isinstance(billed, BillingEvent)


def test_session_mode_rejects_active_paid_media(tmp_path):
    instance = media(tmp_path, settings(proactive_billing_mode="session"))
    instance.refresh_billing_accounts(ACCOUNTS)
    try:
        instance._billing_event(Event(), Event(), instance._settings)
    except BillingConfigurationError:
        pass
    else:
        raise AssertionError("active media must not bill the session identity")


def test_missing_fixed_identity_is_rejected_without_ledger_lookup(tmp_path):
    instance = media(tmp_path, settings())
    # No refresh at all -> empty cache.
    try:
        instance._billing_event(Event(), Event(), instance._settings)
    except BillingConfigurationError:
        pass
    else:
        raise AssertionError("missing explicit identity must be rejected")


def test_refresh_handles_garbage_rows(tmp_path):
    instance = media(tmp_path, settings())
    instance.refresh_billing_accounts(
        [None, {}, {"platform": "x"}, {"user_id": "1"}, {"platform": "y", "user_id": "2",
                                                          "amount": "not-a-number"}]
    )
    assert not hasattr(instance, "_billing_cache")

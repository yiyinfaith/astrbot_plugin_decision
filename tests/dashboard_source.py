"""Text of the dashboard view, for static assertions on markup and behaviour."""

from __future__ import annotations

from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "views" / "dashboard"


def dashboard_markup() -> str:
    return (DASHBOARD / "index.html").read_text(encoding="utf-8")


def dashboard_source() -> str:
    """Template plus the Vue application script it is driven by."""

    return dashboard_markup() + "\n" + (DASHBOARD / "app.js").read_text(encoding="utf-8")

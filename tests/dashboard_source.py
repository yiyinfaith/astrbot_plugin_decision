"""Text of the dashboard view, for static assertions on markup and behaviour."""

from __future__ import annotations

from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "views" / "dashboard"


def dashboard_markup() -> str:
    return (DASHBOARD / "index.html").read_text(encoding="utf-8")


def dashboard_source() -> str:
    """Template, the Vue application script and both themes (default and liquid glass)."""

    parts = [dashboard_markup()]
    parts += [
        (DASHBOARD / name).read_text(encoding="utf-8")
        for name in ("app.js", "style.css", "liquid.css")
    ]
    return "\n".join(parts)

"""The network manager must live inside the dashboard page, scoped and lazy-loaded."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "pages" / "dashboard"


@pytest.fixture(scope="module")
def page():
    return (DASH / "index.html").read_text(encoding="utf-8")


def test_network_view_is_embedded_in_the_card(page):
    assert 'id="network-detail"' in page
    assert 'id="network-shell"' in page
    # The card must not link out to a standalone page any more.
    assert "pages/network" not in page
    assert "network/embed" not in page


def test_all_network_dialogs_are_present(page):
    for dialog in (
        "character-dialog",
        "relationship-dialog",
        "life-event-dialog",
        "merge-dialog",
        "generation-dialog",
        "confirm-dialog",
    ):
        assert f'id="{dialog}"' in page, dialog


def test_network_view_loads_lazily(page):
    assert "import('./network-app.js')" in page
    assert "cytoscape.min.js" in page
    # The panel starts hidden and is revealed only once the view is loaded.
    assert 'v-show="networkDetailLoaded"' in page
    assert "networkDetailLoaded: false" in page


def test_ported_assets_exist():
    assert (DASH / "network-app.js").is_file()
    assert (DASH / "network-view.css").is_file()
    assert (DASH / "vendor" / "cytoscape.min.js").is_file()


def test_stylesheet_is_fully_scoped():
    css = (DASH / "network-view.css").read_text(encoding="utf-8")
    # Strip comments, at-rules and the variable block, then require every
    # selector list to start with the scope.
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    for block in re.finditer(r"([^{}]+)\{", body):
        selector = block.group(1).strip()
        if not selector or selector.startswith("#network-shell"):
            continue
        if selector.startswith("@"):
            continue
        if selector in {"}", ""} or selector.endswith(")"):
            continue
        raise AssertionError(f"unscoped selector in network-view.css: {selector[:60]}")


def test_standalone_page_is_gone():
    assert not (ROOT / "pages" / "network").exists()


def test_server_has_no_dead_embed_route():
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "network_embed_url" not in main
    assert '"network/embed"' not in main

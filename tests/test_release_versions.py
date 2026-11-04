"""Every version string a release touches must match metadata.yaml."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _metadata_version() -> str:
    text = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*v?([\w.\-]+)\s*$", text, re.MULTILINE)
    assert match, "metadata.yaml has no version"
    return match.group(1)


def test_view_asset_versions_match_the_release():
    version = _metadata_version()
    page = (ROOT / "views" / "dashboard" / "index.html").read_text(encoding="utf-8")
    tags = re.findall(r"\?v=([\w.\-]+)", page)
    assert tags, "index.html should version its assets"
    assert set(tags) == {version}
    script = (ROOT / "views" / "dashboard" / "app.js").read_text(encoding="utf-8")
    assert re.search(r"const ASSET_VERSION = '([\w.\-]+)';", script).group(1) == version


def test_changelog_starts_with_the_release():
    version = _metadata_version()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    first = re.search(r"^## v?([\w.\-]+)", changelog, re.MULTILINE)
    assert first and first.group(1) == version

"""The plugin's Python entry source: main.py plus the DecisionPlugin mixins."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ROOT / "main.py",
    ROOT / "decision" / "plugin_shared.py",
    ROOT / "decision" / "plugin_routing.py",
    ROOT / "decision" / "plugin_life.py",
    ROOT / "decision" / "plugin_webui.py",
)


def plugin_source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in FILES)

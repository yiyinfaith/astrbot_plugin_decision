"""Bundled defaults for the OutputPro-compatible response pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_path = Path(__file__).with_name("outputpro_defaults.json")
DEFAULT_OUTPUT_CONFIG: dict[str, Any] = json.loads(_path.read_text(encoding="utf-8"))


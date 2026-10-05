from __future__ import annotations

import re
from collections.abc import Iterable

# AstrBot's built-in SkillManager renders one top-level ``## Skills`` block.
# Keep this parser intentionally format-tolerant: descriptions and paths are
# untrusted text, so only the generated inventory bullet is inspected.
_SKILLS_BLOCK_RE = re.compile(r"(?ms)^## Skills\s*\n.*?(?=^## (?!#)|\Z)")
_SKILL_ITEM_RE = re.compile(
    r"(?ms)^- \*\*(?P<name>[^*\n]+)\*\*:.*?\n  File: `[^\n]*`\s*"
)


def filter_skills_prompt(system_prompt: str, allowed_names: Iterable[str]) -> str:
    """Remove unselected skills from AstrBot's injected system prompt.

    The plugin runs after AstrBot has built the request, so rebuilding the
    whole prompt would risk dropping persona or another plugin's additions.
    This function edits only the generated ``## Skills`` section and leaves
    every other system-prompt section byte-for-byte intact.
    """

    text = str(system_prompt or "")
    allowed = {str(name).strip() for name in allowed_names if str(name).strip()}
    match = _SKILLS_BLOCK_RE.search(text)
    if not match:
        return text
    block = match.group(0)
    if not allowed:
        return text[: match.start()] + text[match.end() :]

    def keep_item(item: re.Match[str]) -> str:
        return item.group(0) if item.group("name").strip() in allowed else ""

    filtered = _SKILL_ITEM_RE.sub(keep_item, block)
    return text[: match.start()] + filtered + text[match.end() :]


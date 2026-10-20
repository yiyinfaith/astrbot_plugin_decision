"""Persona-scoped relationship networks merged into ``astrbot_plugin_decision``.

The subpackage was ported from the standalone ``astrbot_plugin_personal_network``
plugin. :class:`PersonalNetworkService` is a plain object (not an AstrBot ``Star``);
the main plugin owns one instance and wires the AstrBot hooks, LLM tools, chat
command and WebAPI routes to it.
"""

from __future__ import annotations

from .service import PersonalNetworkService
from .storage import VALID_RELATIONSHIP_STATUSES, NetworkStorage

__all__ = [
    "VALID_RELATIONSHIP_STATUSES",
    "NetworkStorage",
    "PersonalNetworkService",
]

"""Persona-scoped relationship networks merged into ``astrbot_plugin_decision``.

The subpackage was ported from the standalone ``astrbot_plugin_personal_network``
plugin. :class:`PersonalNetworkService` is a plain object (not an AstrBot ``Star``);
the main plugin owns one instance and wires the AstrBot hooks, LLM tools, chat
command and WebAPI routes to it.
"""

from __future__ import annotations

from .storage import VALID_RELATIONSHIP_STATUSES, NetworkStorage

__all__ = [
    "VALID_RELATIONSHIP_STATUSES",
    "NetworkStorage",
    "PersonalNetworkService",
]


def __getattr__(name):
    # Keep storage/generation imports usable with older AstrBot installations
    # that do not expose ``astrbot.api.web`` at import time.  The service is
    # loaded lazily only when the owning plugin actually needs its WebUI.
    if name == "PersonalNetworkService":
        from .service import PersonalNetworkService

        return PersonalNetworkService
    raise AttributeError(name)

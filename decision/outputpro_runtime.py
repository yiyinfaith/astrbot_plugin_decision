"""Optional OutputPro-compatible response pipeline.

The decision plugin keeps the output pipeline behind a lazy boundary.  This is
intentional: AstrBot can load the plugin in installations that do not have
OutputPro's optional media packages installed, and the rest of the decision
features must still work in that case.  The actual step implementations are
vendored under :mod:`decision.outputpro` and are only imported when the user
enables output enhancement in the WebUI.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:  # Keep utility imports usable by offline documentation/test tooling.
    from astrbot import logger
except ImportError:  # pragma: no cover - AstrBot supplies this in production
    import logging

    logger = logging.getLogger("astrbot_plugin_decision.output")

from .outputpro_defaults import DEFAULT_OUTPUT_CONFIG


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def output_config_defaults() -> dict[str, Any]:
    """Return a fresh copy so Vue edits never mutate module defaults."""

    return copy.deepcopy(DEFAULT_OUTPUT_CONFIG)


class OutputPipelineRuntime:
    """Lifecycle and event adapter for the vendored OutputPro pipeline."""

    def __init__(self, context: Any, data_dir: Path) -> None:
        self.context = context
        self.data_dir = data_dir
        self._pipeline: Any | None = None
        self._plugin_config: Any | None = None
        self.error: str = ""

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    async def configure(self, config: dict[str, Any]) -> bool:
        """Load or rebuild the pipeline using only local plugin data."""

        await self.close()
        try:
            from .outputpro.core.config import PluginConfig
            from .outputpro.core.pipeline import Pipeline

            payload = _deep_merge(output_config_defaults(), config)
            # Output steps always follow the bundled registry order.  The
            # WebUI only controls which steps are enabled; an old settings
            # file may still contain ``lock_order: false``, so normalize it
            # after merging user data as well as at the defaults layer.
            pipeline = payload.get("pipeline")
            if not isinstance(pipeline, dict):
                pipeline = {}
                payload["pipeline"] = pipeline
            pipeline["lock_order"] = True
            # Keep the bundled sample quote file in this plugin's data
            # directory, while preserving user-supplied quote files.  The
            # two known sample paths from OutputPro are aliases for the local
            # copy; arbitrary paths entered in the WebUI remain usable.
            summary = payload.get("summary")
            if not isinstance(summary, dict):
                summary = {}
                payload["summary"] = summary
            local_quotes = self.data_dir / "default_quotes.json"
            configured_quotes = summary.get("quotes_files")
            quote_files: list[str] = []
            if isinstance(configured_quotes, list):
                for item in configured_quotes:
                    value = str(item or "").strip()
                    if not value:
                        continue
                    normalized = value.replace("\\", "/")
                    if normalized in {
                        "data/plugins/astrbot_plugin_outputpro/default_quotes.json",
                        "data/plugins/astrbot_plugin_decision/outputboost/default_quotes.json",
                    }:
                        value = str(local_quotes)
                    quote_files.append(value)
            summary["quotes_files"] = quote_files or [str(local_quotes)]
            t2i = payload.get("t2i")
            if isinstance(t2i, dict):
                configured_style = str(t2i.get("pillowmd_style_dir", "") or "").strip()
                style_path = Path(configured_style).expanduser() if configured_style else None
                if style_path is not None and not style_path.is_absolute():
                    style_path = Path.cwd() / style_path
                # A server that previously used OutputPro may already have
                # its PillowMD style assets.  Reuse them transparently when
                # the migrated plugin has no bundled style directory yet.
                legacy_style = Path.cwd() / "data/plugins/astrbot_plugin_outputpro/t2i_style"
                if style_path is None or not style_path.exists():
                    if legacy_style.exists():
                        t2i["pillowmd_style_dir"] = str(legacy_style)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            quotes_path = self.data_dir / "default_quotes.json"
            if not quotes_path.exists():
                quotes_path.write_text(
                    json.dumps(payload.get("summary", {}).get("quotes", []), ensure_ascii=False),
                    encoding="utf-8",
                )
            plugin_config = PluginConfig(payload, self.context)
            plugin_config.data_dir = self.data_dir
            self._plugin_config = plugin_config
            self._pipeline = Pipeline(plugin_config)
            await self._pipeline.initialize()
            self.error = ""
            return True
        except Exception as exc:  # optional dependencies are deployment-specific
            self._pipeline = None
            self._plugin_config = None
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Decision Engine output enhancement is unavailable: %s", self.error)
            return False

    async def run(self, event: Any) -> None:
        if self._pipeline is None:
            return
        try:
            result = event.get_result()
            if not result or not result.chain:
                return
            # Steps mutate the live result chain in place.  Keep a deep copy so
            # a later optional step failure cannot leak a partially formatted
            # response to AstrBot.  Side effects that were already sent to a
            # platform cannot be undone, but the unsent final chain remains
            # fail-open.
            original_chain = copy.deepcopy(result.chain)
            from .outputpro.core.model import OutContext, StateManager

            gid = str(event.get_group_id() or event.get_sender_id() or "")
            message_obj = getattr(event, "message_obj", None)
            raw_timestamp = getattr(message_obj, "timestamp", 0) or 0
            try:
                timestamp = int(raw_timestamp)
            except (TypeError, ValueError, OverflowError):
                timestamp = 0
            context = OutContext(
                event=event,
                chain=result.chain,
                is_llm=result.is_llm_result(),
                plain=result.get_plain_text(),
                gid=gid,
                uid=str(event.get_sender_id() or ""),
                bid=str(event.get_self_id() or ""),
                group=StateManager.get_group(gid),
                timestamp=timestamp,
            )
            await self._pipeline.run(context)
        except Exception as exc:
            # Output enhancement is fail-open.  A formatting step must never
            # prevent AstrBot from sending the original response.
            try:
                result.chain[:] = original_chain
            except (NameError, AttributeError, TypeError, ValueError):
                pass
            logger.warning("Decision Engine output enhancement failed: %s", exc)

    async def prepare_message(self, event: Any) -> None:
        """Update OutputPro's lightweight group state before a new message.

        Smart Reply and fake-@ parsing need this state even when the outgoing
        response is produced by another plugin.  The operation is in-memory
        only and is intentionally a no-op when the relevant step is disabled.
        """

        if self._plugin_config is None:
            return
        try:
            from .outputpro.core.model import StateManager, StepName

            # Private chats have no group ID.  Fall back to the sender so
            # OutputPro's per-conversation state cannot leak between users.
            gid = str(event.get_group_id() or event.get_sender_id() or "")
            group = StateManager.get_group(gid)
            sender_id = str(event.get_sender_id() or "")
            self_id = str(event.get_self_id() or "")
            if self._plugin_config.reply.threshold > 0 and sender_id != self_id:
                message_obj = getattr(event, "message_obj", None)
                message_id = getattr(message_obj, "message_id", None)
                if message_id is not None:
                    group.msg_queue.append(message_id)
            if self._plugin_config.pipeline.is_enabled_step(StepName.AT):
                at_config = self._plugin_config.at
                if not at_config.at_str:
                    name = str(event.get_sender_name() or "").strip()
                    if name:
                        group.name_to_qq[name] = sender_id
                        while len(group.name_to_qq) > 100:
                            group.name_to_qq.popitem(last=False)
        except Exception as exc:
            logger.debug("Decision Engine output input state update failed: %s", exc)

    async def close(self) -> None:
        if self._pipeline is not None:
            try:
                await self._pipeline.terminate()
            except Exception as exc:
                logger.debug("Decision Engine output pipeline shutdown failed: %s", exc)
        self._pipeline = None
        self._plugin_config = None


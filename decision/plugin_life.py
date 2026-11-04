"""Life, multi-modal and Personal Network helpers for DecisionPlugin."""

# ruff: noqa: F403, F405
from __future__ import annotations

from .plugin_shared import *
from .plugin_shared import _ON_DECORATING_RESULT, _safe_error  # noqa: F401


class LifeMixin:
    """Runtime lifecycles and the LLM tool handlers of life, media and network."""

    def _multimodal_settings(self) -> dict[str, Any]:
        from .multimodal import normalize_settings

        return normalize_settings(dict(self._setting("multimodal", {}) or {}))

    def _get_multimodal(self):
        """Create the multi-modal runtime on first use."""

        if self._multimodal is not None:
            return self._multimodal
        from .multimodal.runtime import MultimodalRuntime

        base = (
            Path(self._settings_path).parent
            if getattr(self, "_settings_path", None)
            else Path.cwd()
        )
        runtime = MultimodalRuntime(self, data_dir=base, settings=self._multimodal_settings())
        self._multimodal = runtime
        return runtime

    async def _ensure_multimodal(self):
        runtime = self._get_multimodal()
        await runtime.initialize()
        self._sync_multimodal_tools(runtime)
        await self._apply_multimodal_tool_state(self._multimodal_settings())
        return runtime

    def _sync_multimodal_tools(self, runtime) -> None:
        """(Re)register the understanding tools according to the current switches."""

        from .multimodal.tools import build_tools, tool_names

        wanted = set(runtime.settings.tool_kinds)
        existing = {tool.name: tool for tool in self._multimodal_tools}
        for name, tool in list(existing.items()):
            if name not in tool_names(wanted):
                self._life_tools.remove(tool) if tool in self._life_tools else None
                self._multimodal_tools.remove(tool)
        for tool in build_tools(runtime, tuple(wanted)):
            if any(item.name == tool.name for item in self._multimodal_tools):
                continue
            self._multimodal_tools.append(tool)
            self._life_tools.append(tool)
            self.context.add_llm_tools(tool)

    @staticmethod
    def _media_kind_summary(refs: Any) -> str:
        labels = {"image": "图片", "video": "视频", "audio": "音频", "file": "文件"}
        counts: dict[str, int] = {}
        for ref in refs:
            counts[ref.kind] = counts.get(ref.kind, 0) + 1
        return "、".join(f"{labels.get(kind, kind)}×{count}" for kind, count in counts.items())

    async def _media_placeholder(self, refs: Any, umo: str, event=None) -> str:
        """Give Jev attachment semantics when this turn's auto task is ready."""

        summary = self._media_kind_summary(refs)
        cached: list[str] = []
        runtime = self._multimodal
        if runtime is not None:
            if event is not None:
                items = await runtime.collect_for_event(
                    event,
                    list(refs[:4]),
                    wait=float(runtime.settings.int("mm_auto_wait_seconds")),
                )
                cached.extend(item.summary for item in items if getattr(item, "ok", False))
            else:
                for ref in refs[:4]:
                    item = await runtime.cache.get_for_media(umo, ref.content_key)
                    if item is not None and getattr(item, "ok", False):
                        cached.append(item.summary)
            if not cached:
                for item in await runtime.cache.recent_understandings(umo, 2):
                    cached.append(item.summary)
        detail = ""
        if cached:
            detail = "\n已有理解结果：" + "\n".join(cached)[:800]
        return f"[附件资料] 对方发送了{summary}。{detail}"

    async def _media_refs_for(self, event: AstrMessageEvent) -> list[Any]:
        """Media on this message, independent of the understanding engine.

        The 非文字消息 switch must work even when multi-modal understanding is
        disabled, so detection never depends on decision_mm_refs being produced.
        """

        refs = event.get_extra("decision_mm_refs")
        if refs:
            return list(refs)
        from .multimodal.media import extract_media

        try:
            return await extract_media(event)
        except Exception as exc:
            logger.debug("Decision Engine 媒体探测失败：%s", _safe_error(exc))
            return []

    async def _get_life_runtime(self):
        async with self._life_lock:
            if self._life_runtime is None:
                await self._start_life_runtime_locked()
            return self._life_runtime

    async def _start_life_runtime_locked(self) -> None:
        """Build the life runtime from the saved settings; caller holds _life_lock."""

        from .life_runtime import LifeRuntime

        settings = normalize_life(dict(self._setting("life", {}) or {}))
        runtime = LifeRuntime(self, settings)
        try:
            await runtime.initialize()
        except BaseException:
            # Clean up, but never let a close error replace the real failure
            # (or a CancelledError).
            try:
                await runtime.terminate()
            except Exception as exc:
                logger.warning("Decision Engine 主动发起后台清理失败：%s", _safe_error(exc))
            raise
        self._life_runtime = runtime

    async def _stop_life_runtime_locked(self) -> None:
        """Tear down the current life runtime; caller holds _life_lock."""

        # Forget it first, so a failing terminate() still lets a new one start.
        runtime, self._life_runtime = self._life_runtime, None
        if runtime is not None:
            try:
                await runtime.terminate()
            except Exception as exc:
                # terminate() always stops the scheduler and tasks first, so
                # starting a new runtime afterwards is still safe.
                logger.warning("Decision Engine 主动发起后台关闭时出错：%s", _safe_error(exc))

    async def _configure_life_runtime(self):
        # One critical section, so a runtime still being built from older
        # settings is replaced instead of being returned as the new one.
        # 主动发起 is enabled by its own per-kind switches (私聊主动 / 群聊主动),
        # so there is no separate master switch to consult here.
        async with self._life_lock:
            await self._stop_life_runtime_locked()
            await self._start_life_runtime_locked()

    def _get_life_media(self):
        if self._life_media is None:
            from .life_media import LifeMedia

            self._life_media = LifeMedia(
                self,
                self._settings_path.parent.parent / "plugin_data" / PLUGIN_NAME / "virtual_life",
            )
        return self._life_media

    def _get_personal_network(self):
        """Return the merged Personal Network service (created lazily)."""

        if self._personal_network is None:
            from .personal_network import PersonalNetworkService

            self._personal_network = PersonalNetworkService(
                self,
                self._settings_path.parent.parent
                / "plugin_data"
                / PLUGIN_NAME
                / "personal_network",
            )
        return self._personal_network

    async def _take_photo(
        self, event: AstrMessageEvent, scene: str = "", subject: str = "self", count: int = 1
    ) -> str:
        return await self._get_life_media().capture(
            event, kind="photo", scene=scene, subject=subject, count=count
        )

    async def _take_video(
        self, event: AstrMessageEvent, scene: str = "", subject: str = "self", count: int = 1
    ) -> str:
        return await self._get_life_media().capture(
            event, kind="video", scene=scene, subject=subject, count=count
        )

    async def _send_voice(
        self, event: AstrMessageEvent, text: str, style: str = "", count: int = 1
    ) -> str:
        return await self._get_life_media().capture(
            event, kind="audio", scene=style, audio_text=text, count=count
        )

    async def _get_virtual_daily_schedule(self, event: AstrMessageEvent) -> str:
        runtime = await self._get_life_runtime()
        return await runtime.get_virtual_daily_schedule(event)

    async def _get_long_term_timeline(self, event: AstrMessageEvent) -> str:
        runtime = await self._get_life_runtime()
        return await runtime.get_long_term_timeline(event)

    async def _add_schedule_requirement(
        self, event: AstrMessageEvent, date_range: str, requirement: str
    ) -> str:
        if getattr(event, "role", "member") != "admin":
            return "只有管理员可以添加日程要求。"
        runtime = await self._get_life_runtime()
        return await runtime._add_schedule_requirement(event, date_range, requirement)

    async def _adjust_contact_frequency(
        self, event, action="status", interval_minutes=0, pause_until=""
    ):
        from .life_temporal import parse_time

        runtime = await self._get_life_runtime()
        persona = await runtime.personas.resolve(str(event.unified_msg_origin))
        try:
            until = parse_time(pause_until, runtime.timezone).timestamp() if pause_until else 0
            return runtime.contacts.preference(event, persona.id, action, interval_minutes, until)
        except (ValueError, TypeError) as exc:
            return "未修改联系偏好：" + str(exc)

    async def _future_contact(
        self,
        event,
        action="list",
        task_id="",
        run_at="",
        delay_seconds=0,
        note="",
        origin="user",
        kind="reminder",
        notify=True,
    ):
        runtime = await self._get_life_runtime()
        try:
            return await runtime.temporal.manage(
                event,
                action=action,
                task_id=task_id,
                run_at=run_at,
                delay_seconds=delay_seconds,
                note=note,
                origin=origin,
                kind=kind,
                notify=notify,
            )
        except (ValueError, TypeError) as exc:
            return "未来任务未修改：" + str(exc)

    async def _save_life_thought(self, event, text):
        from .virtual_life.core.utils import timeline_item_at

        runtime = await self._get_life_runtime()
        persona = await runtime.personas.resolve(str(event.unified_msg_origin))
        plan = runtime.visible_plan(
            runtime.storage.get_plan(runtime._now().date().isoformat(), persona.id)
        )
        item = timeline_item_at(plan, runtime._now()) if plan else None
        if not item:
            return "没有可验证的当前日程活动，未保存想法。"
        try:
            ident = await runtime.ideas.save_thought(
                persona.id,
                text,
                {
                    "activity": item.activity,
                    "location": item.location,
                    "at": runtime._now().isoformat(),
                },
                scope=str(event.unified_msg_origin),
            )
            return "想法已暂存，是否分享由限频与偏好决定。记录 " + ident
        except ValueError as exc:
            return str(exc)

    @staticmethod
    def _mm_allowed_tool_names(settings: dict[str, Any]) -> set[str]:
        from .multimodal.config import MEDIA_KINDS
        from .multimodal.tools import tool_names

        if not settings.get("mm_enabled") or not settings.get("mm_tools_enabled"):
            return set()
        return tool_names([kind for kind in MEDIA_KINDS if settings.get(f"mm_tool_{kind}")])

    async def _apply_multimodal_tool_state(self, settings: dict[str, Any]) -> None:
        """Activate or deactivate the understanding tools inside AstrBot itself.

        ``Context.activate_llm_tool`` / ``deactivate_llm_tool`` flip the global
        active flag used by JEV routing, the tool list and the actual call path, so
        a disabled tool is really unavailable instead of just hidden.
        """

        from .multimodal.tools import all_tool_names

        allowed = self._mm_allowed_tool_names(settings)
        for name in sorted(all_tool_names()):
            try:
                verb = "activate" if name in allowed else "deactivate"
                toggle = getattr(self.context, f"{verb}_llm_tool_async", None)
                if callable(toggle):
                    await toggle(name)
                else:
                    # Older hosts only offer the synchronous API.
                    toggle = getattr(self.context, f"{verb}_llm_tool", None)
                    if callable(toggle):
                        toggle(name)
            except Exception as exc:  # pragma: no cover - host API differences
                logger.debug("Decision Engine 切换多模态工具 %s 失败：%s", name, _safe_error(exc))

    def _mm_scope_allows(self, event: AstrMessageEvent, settings: dict[str, Any]) -> bool:
        """Apply the multi-modal auto-understanding black/white list."""

        configured = settings.get("mm_auto_scope") or []
        values = {str(item).strip() for item in configured if str(item).strip()}
        matched = bool(self._event_scope_candidates(event) & values)
        if settings.get("mm_auto_scope_blacklist"):
            return not matched
        return matched

    def _network_enabled(self) -> bool:
        settings = (self._setting("life", {}) or {}).get("personal_network_integration", {})
        return bool(isinstance(settings, Mapping) and settings.get("enable", False))

    def _network_settings(self) -> Mapping:
        settings = (self._setting("life", {}) or {}).get("personal_network_integration", {})
        return settings if isinstance(settings, Mapping) else {}

    @staticmethod
    def _hide_network_tools(req: ProviderRequest, *, update: bool, query: bool) -> None:
        tool_set = getattr(req, "func_tool", None)
        if tool_set is None or not callable(getattr(tool_set, "remove_tool", None)):
            return
        if update:
            tool_set.remove_tool("update_personal_network")
        if query:
            tool_set.remove_tool("query_personal_network")

    async def network_update_tool(
        self,
        event: AstrMessageEvent,
        characters: list[dict[str, Any]] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        interactions: list[dict[str, Any]] | None = None,
    ) -> str:
        return await self._network().update_tool(
            event, characters=characters, relationships=relationships, interactions=interactions
        )

    async def network_query_tool(self, event: AstrMessageEvent, query: str = "") -> str:
        return await self._network().query_tool(event, query)

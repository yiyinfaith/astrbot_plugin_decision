"""WebUI API handlers of DecisionPlugin (registered with register_web_api)."""

# ruff: noqa: F403, F405
from __future__ import annotations

from .plugin_shared import *
from .plugin_shared import _ON_DECORATING_RESULT, _safe_error  # noqa: F401


class WebUIMixin:
    """Page settings, life workspace and Personal Network endpoints."""

    def _output_page_settings(self) -> dict[str, Any]:
        """Return a complete, editable OutputPro-compatible configuration."""

        from .outputpro_runtime import _deep_merge, output_config_defaults

        configured = self._setting("output_pipeline", {})
        merged = _deep_merge(
            output_config_defaults(),
            configured if isinstance(configured, Mapping) else {},
        )
        pipeline = merged.get("pipeline")
        if not isinstance(pipeline, dict):
            pipeline = {}
            merged["pipeline"] = pipeline
        pipeline["lock_order"] = True
        return {
            "conversation_flow_analysis_enabled": self._bool(
                "conversation_flow_analysis_enabled", True
            ),
            "conversation_flow_window": self._int("conversation_flow_window", 8),
            "conversation_flow_scope": [
                str(item)
                for item in (self._setting("conversation_flow_scope", []) or [])
                if str(item).strip()
            ],
            "conversation_flow_scope_blacklist": self._bool(
                "conversation_flow_scope_blacklist", True
            ),
            "output_enhancement_enabled": self._bool("output_enhancement_enabled", False),
            "output_scope": [
                str(item) for item in (self._setting("output_scope", []) or []) if str(item).strip()
            ],
            "output_scope_blacklist": self._bool("output_scope_blacklist", True),
            "output_pipeline": merged,
            "optimize_tool_history_context": self._bool(
                "optimize_tool_history_context", False
            ),
            "disable_group_at_bot_wake": self._bool(
                "disable_group_at_bot_wake", False
            ),
            "disable_group_at_bot_wake_all_groups": self._bool(
                "disable_group_at_bot_wake_all_groups", False
            ),
            "disable_group_at_bot_wake_group_ids": [
                str(item)
                for item in (
                    self._setting("disable_group_at_bot_wake_group_ids", []) or []
                )
                if str(item).strip()
            ],
            "disable_group_reply_to_bot_wake": self._bool(
                "disable_group_reply_to_bot_wake", False
            ),
            "disable_group_reply_to_bot_wake_all_groups": self._bool(
                "disable_group_reply_to_bot_wake_all_groups", False
            ),
            "disable_group_reply_to_bot_wake_group_ids": [
                str(item)
                for item in (
                    self._setting("disable_group_reply_to_bot_wake_group_ids", []) or []
                )
                if str(item).strip()
            ],
            "multimodal": self._multimodal_page_settings(),
        }

    def _multimodal_page_settings(self) -> dict[str, Any]:
        """Editable multi-modal section plus the provider dropdown."""

        from .multimodal import page_settings
        from .multimodal.adapters import provider_catalog
        from .multimodal.config import MultimodalSettings

        settings = MultimodalSettings(self._multimodal_settings())
        payload = page_settings(settings)
        payload["models"] = provider_catalog(self.context)
        return payload

    async def page_tools(self):
        from astrbot.api.web import json_response

        tools = []
        seen_names: set[str] = set()
        builtin_names = self._builtin_tool_names()
        from .multimodal.tools import all_tool_names

        allowed_mm = self._mm_allowed_tool_names(self._multimodal_settings())
        mm_tool_names = all_tool_names()
        for tool in [*self._manager_tool_objects(), *self._subagent_tool_objects()]:
            name = str(getattr(tool, "name", ""))
            if not name or name in seen_names:
                continue
            if name in mm_tool_names and name not in allowed_mm:
                continue
            seen_names.add(name)
            tools.append(
                {
                    "name": name,
                    "description": short_description(
                        getattr(tool, "description", ""),
                    ),
                    "handoff": is_handoff_tool(tool),
                    "mcp": is_mcp_tool(tool),
                    "skill": False,
                    "active": tool_is_active(tool),
                    "builtin": name in builtin_names,
                    "origin_display": self._tool_origin(tool, builtin=name in builtin_names),
                }
            )
        for tool in self._builtin_tool_objects():
            name = str(getattr(tool, "name", ""))
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            tools.append(
                {
                    "name": name,
                    "description": short_description(
                        getattr(tool, "description", "") or "AstrBot 内置工具",
                    ),
                    "handoff": False,
                    "mcp": False,
                    "skill": False,
                    "active": tool_is_active(tool),
                    "builtin": True,
                    "origin_display": "Astrbot内置工具",
                }
            )
        if DECISION_TOOL_NAME not in seen_names:
            tools.insert(
                0,
                {
                    "name": DECISION_TOOL_NAME,
                    "description": "调用 AstrBot 原生 JEV Decision Provider 对当前场景执行一次结构化判断；不会执行其他工具。",
                    "handoff": False,
                    "mcp": False,
                    "skill": False,
                    "active": True,
                    "builtin": False,
                    "origin_display": PLUGIN_DISPLAY_NAME,
                },
            )
        for skill in self._skill_objects():
            name = str(getattr(skill, "name", ""))
            if not name:
                continue
            tools.append(
                {
                    "name": name,
                    "description": short_description(getattr(skill, "description", "")),
                    "handoff": False,
                    "mcp": False,
                    "skill": True,
                    "active": tool_is_active(skill),
                    "builtin": False,
                    "origin_display": self._skill_origin(skill),
                }
            )
        tools.sort(
            key=lambda item: (
                bool(item.get("builtin")),
                str(item.get("origin_display", "")).casefold(),
                str(item.get("name", "")).casefold(),
            )
        )
        return json_response({"tools": tools})

    def _reserve_life_job(self, ident: str, action: str, umo: str) -> bool:
        """Register a new WebUI job, evicting finished ones first.

        Returns False when every retained job is still running, so the caller can
        answer 429 without leaving a half-registered entry behind.
        """

        while len(self._life_jobs) >= self._life_job_limit:
            removable = next(
                (key for key, value in self._life_jobs.items() if value["state"] != "running"),
                None,
            )
            if removable is None:
                return False
            self._life_jobs.pop(removable, None)
        self._life_jobs[ident] = {"state": "running", "action": action, "umo": umo}
        return True

    def _life_tasks_payload(self) -> dict[str, dict[str, Any]]:
        """Serialise recent life jobs without the embedded base64 images.

        ``/life`` is polled every two seconds while a job runs. Returning the raw
        results meant re-sending every past screenshot on each poll (tens of MB
        once a few render jobs had accumulated), so images are reported as a
        count and only text parts are kept.
        """

        payload: dict[str, dict[str, Any]] = {}
        for ident, job in self._life_jobs.items():
            entry = {key: value for key, value in job.items() if key != "result"}
            result = job.get("result")
            if isinstance(result, list):
                texts = [
                    part
                    for part in result
                    if isinstance(part, Mapping) and part.get("type") == "text"
                ]
                images = sum(
                    1
                    for part in result
                    if isinstance(part, Mapping) and part.get("type") == "image"
                )
                entry["result"] = texts
                if images:
                    entry["image_count"] = images
            elif result is not None:
                entry["result"] = result
            payload[ident] = entry
        return payload

    async def page_save_ui_settings(self):
        """Save only the display preference; every other stored value is kept as is."""

        from astrbot.api.web import error_response, json_response, request

        payload = await request.json(default={})
        value = payload.get("ui_liquid_glass") if isinstance(payload, Mapping) else None
        if not isinstance(value, bool):
            return error_response("ui_liquid_glass 必须是布尔值", status_code=400)
        try:
            self._save_detail_settings({**self._detail_settings, "ui_liquid_glass": value})
        except SETTINGS_WRITE_ERRORS as exc:
            logger.error("Decision Engine could not save the view preference: %s", _safe_error(exc))
            return error_response(f"保存失败：{_safe_error(exc)}", status_code=500)
        return json_response({"saved": True, "ui_liquid_glass": value})

    async def page_multimodal_models(self):
        """List the providers the multi-modal understanding can use."""

        from astrbot.api.web import json_response

        from .multimodal.adapters import provider_catalog

        return json_response({"models": provider_catalog(self.context)})

    async def page_life(self):
        from astrbot.api.web import json_response

        providers = []
        getter = getattr(self.context, "get_all_providers", None)
        if callable(getter):
            for provider in getter():
                try:
                    meta = provider.meta()
                    providers.append({"id": meta.id, "name": meta.id})
                except (AttributeError, RuntimeError, TypeError):
                    continue
        return json_response(
            {
                "providers": providers,
                "schema": life_schema(),
                "settings": normalize_life(dict(self._setting("life", {}) or {})),
                "status": self._life_runtime.status()
                if self._life_runtime
                else {"enabled": False, "jobs": [], "audit": [], "sessions": {}},
                "tasks": self._life_tasks_payload(),
                "aigen_available": self._get_life_media().aigen() is not None,
                "usage": usage_meter.METER.summary(7),
            }
        )

    async def page_life_personas(self):
        """List AstrBot personas for the appearance selector."""

        from astrbot.api.web import json_response

        personas = await self._persona_options()
        return json_response({"personas": personas})

    async def _persona_options(self) -> list[dict[str, str]]:
        getter = getattr(self.context.persona_manager, "get_all_personas", None)
        rows: list[dict[str, str]] = []
        if callable(getter):
            try:
                for persona in await getter():
                    persona_id = str(getattr(persona, "persona_id", "") or "").strip()
                    if persona_id:
                        rows.append({"id": persona_id, "name": persona_id})
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Decision Engine could not list personas: %s", _safe_error(exc))
        if not any(row["id"] == "default" for row in rows):
            rows.append({"id": "default", "name": "default"})
        rows.sort(key=lambda row: row["id"].casefold())
        return rows

    async def page_life_sessions(self):
        """List AstrBot conversations so the WebUI can offer a searchable picker."""

        from astrbot.api.web import json_response, request

        query = str((request.query or {}).get("search", "") or "").strip().casefold()
        manager = getattr(self.context, "conversation_manager", None)
        getter = getattr(manager, "get_conversations", None)
        conversations = []
        if callable(getter):
            try:
                conversations = await getter()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Decision Engine could not list conversations: %s", _safe_error(exc))
        persona_names = {row["id"]: row["name"] for row in await self._persona_options()}
        sessions: list[dict[str, str]] = []
        seen: set[str] = set()
        for conversation in conversations or []:
            umo = str(getattr(conversation, "user_id", "") or "").strip()
            if not umo or umo in seen:
                continue
            seen.add(umo)
            persona_id = str(getattr(conversation, "persona_id", "") or "").strip()
            title = str(getattr(conversation, "title", "") or "").strip()
            platform, _, remainder = umo.partition(":")
            message_type, _, _ = remainder.partition(":")
            row = {
                "umo": umo,
                "title": title,
                "platform": platform,
                "message_type": message_type,
                "persona_id": persona_id,
                "persona_name": persona_names.get(persona_id, persona_id),
            }
            if query and query not in " ".join(row.values()).casefold():
                continue
            sessions.append(row)
            if len(sessions) >= 2000:
                break
        # AstrBot's "自定义规则" can exist before a conversation row is
        # created.  Include concrete routing entries as selectable sessions so
        # the appearance page can resolve their persona immediately.
        try:
            acm = getattr(self.context, "astrbot_config_mgr", None)
            router = getattr(acm, "ucr", None)
            routes = getattr(router, "umop_to_conf_id", {}) or {}
            conf_meta = getattr(acm, "abconf_data", {}) or {}
            for routed_umo, conf_id in routes.items():
                routed_umo = str(routed_umo).strip()
                if not routed_umo or "*" in routed_umo or routed_umo in seen:
                    continue
                parts = routed_umo.split(":", 2)
                if len(parts) != 3 or parts[1] not in {"FriendMessage", "GroupMessage"}:
                    continue
                seen.add(routed_umo)
                meta = conf_meta.get(str(conf_id), {}) if isinstance(conf_meta, dict) else {}
                title = str(meta.get("name", "") or "自定义规则")
                row = {
                    "umo": routed_umo,
                    "title": title,
                    "platform": parts[0],
                    "message_type": parts[1],
                    "persona_id": "",
                    "persona_name": "",
                }
                if not query or query in " ".join(row.values()).casefold():
                    sessions.append(row)
        except (AttributeError, TypeError, ValueError):
            pass
        sessions.sort(key=lambda row: (row["message_type"], row["umo"]))
        return json_response({"sessions": sessions})

    def _network(self):
        return self._get_personal_network()

    async def network_personas(self):
        return await self._network().api_personas()

    async def network_network(self):
        return await self._network().api_network()

    async def network_set_enabled(self):
        return await self._network().api_set_enabled()

    async def network_save_character(self):
        return await self._network().api_save_character()

    async def network_delete_character(self):
        return await self._network().api_delete_character()

    async def network_merge_characters(self):
        return await self._network().api_merge_characters()

    async def network_save_relationship(self):
        return await self._network().api_save_relationship()

    async def network_delete_relationship(self):
        return await self._network().api_delete_relationship()

    async def network_save_life_event(self):
        return await self._network().api_save_life_event()

    async def network_delete_life_event(self):
        return await self._network().api_delete_life_event()

    async def network_generate(self):
        return await self._network().api_generate_network()

    async def network_generation_status(self):
        return await self._network().api_generation_status()

    async def network_generation_validate(self):
        return await self._network().api_validate_generation()

    async def network_generation_apply(self):
        return await self._network().api_apply_generation()

    async def network_avatar(self, character_id: str):
        return await self._network().api_avatar(character_id)

    async def network_export(self):
        return await self._network().api_export()

    async def network_import_preview(self, persona_key: str):
        return await self._network().api_import_preview(persona_key)

    async def network_import_apply(self):
        return await self._network().api_import_apply()

    async def page_life_billing(self):
        """List the platforms and aigen billing accounts for the pickers.

        Platforms come from AstrBot's own configuration; accounts come from
        aigen's billing ledger, which is the authoritative list of identities
        that can actually be charged. Both reads are best-effort so a missing
        aigen install never breaks the settings page.
        """

        from astrbot.api.web import json_response

        platforms: list[dict[str, str]] = []
        try:
            configured = self.context.get_config().get("platform", []) or []
        except (AttributeError, TypeError, ValueError):
            configured = []
        seen: set[str] = set()
        for row in configured:
            if not isinstance(row, Mapping):
                continue
            # aigen keys the account by the adapter *type*, not the instance id.
            name = str(row.get("type") or row.get("id") or "").strip()
            if name and name not in seen:
                seen.add(name)
                platforms.append({"id": name, "name": name})
        platforms.sort(key=lambda item: item["id"].casefold())

        accounts = await self._aigen_billing_accounts()
        # Keep the runtime fallback in sync with what the picker shows, so an
        # unset account can never disagree with the visible list.
        try:
            self._get_life_media().refresh_billing_accounts(accounts)
        except Exception as exc:
            logger.debug("Decision Engine 同步计费账户失败：%s", _safe_error(exc))
        return json_response({"platforms": platforms, "accounts": accounts})

    async def _aigen_billing_accounts(self) -> list[dict[str, Any]]:
        """Read aigen's balance rows without importing or mutating its store."""

        getter = getattr(self.context, "get_all_stars", None)
        if not callable(getter):
            return []
        try:
            stars = list(getter())
        except (AttributeError, TypeError, ValueError):
            return []
        directory = None
        for star in stars:
            if getattr(star, "name", "") != "astrbot_plugin_aigen":
                continue
            root = str(getattr(star, "root_dir_name", "") or "")
            module = getattr(star, "module_path", "") or ""
            if root or module:
                directory = root or Path(module).parent.name
            break
        if not directory:
            return []
        # <data>/plugin_data/<plugin>/billing.sqlite3
        db_path = (
            Path(get_astrbot_config_path()).parent / "plugin_data" / directory / "billing.sqlite3"
        )
        if not db_path.is_file():
            return []

        def read() -> list[dict[str, Any]]:
            import sqlite3

            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
            try:
                rows = connection.execute(
                    "SELECT owner, amount FROM balances ORDER BY amount DESC"
                ).fetchall()
            finally:
                connection.close()
            result = []
            for owner, amount in rows:
                try:
                    platform, user_id = json.loads(owner)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                platform, user_id = str(platform), str(user_id)
                if not platform or not user_id:
                    continue
                result.append(
                    {
                        "platform": platform,
                        "user_id": user_id,
                        "amount": int(amount or 0),
                        "label": f"{platform} · {user_id}",
                    }
                )
            return result

        try:
            return await asyncio.to_thread(read)
        except Exception as exc:
            logger.debug("Decision Engine 读取 aigen 记账账户失败：%s", _safe_error(exc))
            return []

    async def page_life_appearance(self):
        from astrbot.api.web import error_response, json_response, request

        payload = await request.json(default={})
        try:
            if not isinstance(payload, Mapping):
                raise ValueError("请求必须是对象")
            runtime = await self._get_life_runtime()
            persona_id = str(payload.get("persona_id", "") or "").strip()
            if not persona_id:
                # Legacy callers selected a conversation; resolve its persona.
                umo = validate_umo(str(payload.get("umo", "")))
                from astrbot.core.platform.message_session import MessageSession

                MessageSession.from_str(umo)
                persona_id = (await runtime.personas.resolve(umo)).id
            else:
                persona_id = (await runtime.personas.resolve_id(persona_id)).id
            store = self._get_life_media().appearance
            async with self._life_asset_lock:
                if payload.get("action") == "upload":
                    await store.upload(
                        persona_id, payload.get("name", "reference"), payload.get("data_url", "")
                    )
                elif payload.get("action") == "set_description":
                    await store.set_description(persona_id, str(payload.get("description", "")))
                elif payload.get("action") == "delete":
                    await store.remove(persona_id, str(payload.get("id", "")))
                elif payload.get("action", "list") != "list":
                    raise ValueError("未知外貌操作")
                return json_response(
                    {
                        "persona_id": persona_id,
                        "images": await store.list(persona_id, previews=True),
                        "description": await store.get_description(persona_id)
                        or str(
                            (
                                (self._setting("life", {}) or {}).get("appearance_settings", {})
                                or {}
                            ).get("description", "")
                        ),
                    }
                )
        except (ValueError, OSError, TypeError) as exc:
            return error_response(_safe_error(exc), status_code=400)

    async def page_life_control(self):
        from astrbot.api.web import error_response, json_response, request

        from .life_actions import perform_control_action

        payload = await request.json(default={})
        if not isinstance(payload, Mapping):
            return error_response("请求必须是对象", status_code=400)
        if not isinstance(payload.get("action"), str) or payload["action"] not in {
            "create_task",
            "edit_task",
            "cancel_task",
            "resolve_topic",
            "dismiss_idea",
        }:
            return error_response("未知会话控制操作", status_code=400)
        try:
            umo = validate_umo(str(payload.get("umo", "")))
            runtime = await self._get_life_runtime()
            return json_response(await perform_control_action(runtime, umo, payload))
        except (ValueError, TypeError) as exc:
            return error_response(_safe_error(exc), status_code=400)

    async def page_life_action(self):
        from astrbot.api.web import error_response, json_response, request

        payload = await request.json(default={})
        if not isinstance(payload, Mapping):
            return error_response("请求必须是对象", status_code=400)
        allowed = {
            "inspect",
            "generate",
            "rewrite_schedule",
            "rewrite_outfit",
            "add_requirement",
            "generate_timeline",
            "regenerate_timeline",
            "import_timeline",
            "approve_timeline",
            "reject_timeline",
            "settle_delay",
            "initiate",
            "render_schedule",
            "render_outfit",
            "render_timeline",
            "render_stage",
        }
        action, umo = str(payload.get("action", "")), str(payload.get("umo", ""))
        if action not in allowed:
            return error_response("未知生活操作", status_code=400)
        try:
            from astrbot.core.platform.message_session import MessageSession

            validate_umo(umo)
            MessageSession.from_str(umo)
            runtime = await self._get_life_runtime()
        except (ValueError, RuntimeError) as exc:
            return error_response(_safe_error(exc), status_code=400)
        import uuid

        ident = uuid.uuid4().hex
        if not self._reserve_life_job(ident, action, umo):
            return error_response("已有太多后台操作，请稍后再试", status_code=429)

        async def run():
            try:
                from .life_actions import perform_life_action

                # The inspect that follows a mutation must run under the same
                # lock, otherwise a concurrent action can write a plan between
                # the mutation and its read-back and the UI shows stale state.
                async with runtime.action_lock:
                    value = await perform_life_action(
                        runtime,
                        umo,
                        action,
                        str(payload.get("text", ""))[:50000],
                        str(payload.get("date_range", "")),
                    )
                    inspection = None
                    if action in INSPECT_AFTER_ACTIONS:
                        inspection = await perform_life_action(runtime, umo, "inspect")
                job = self._life_jobs.get(ident)
                if job is not None:
                    job.update(state="done", result=value)
                    if inspection is not None:
                        job["inspection"] = inspection
            except asyncio.CancelledError:
                job = self._life_jobs.get(ident)
                if job is not None:
                    job.update(state="cancelled")
                raise
            except Exception as exc:
                job = self._life_jobs.get(ident)
                if job is not None:
                    job.update(state="error", error=_safe_error(exc))

        task = runtime.spawn(run())
        if isinstance(task, asyncio.Task):
            self._life_tasks.add(task)
            task.add_done_callback(self._life_tasks.discard)
        return json_response({"task_id": ident})

    async def page_settings(self):
        from astrbot.api.web import json_response

        self._refresh_native_provider_schema()
        always_keep = [str(item) for item in (self._setting("always_keep_tools", []) or [])]
        if not self._bool("always_keep_tools_customized", False):
            always_keep.extend(sorted(self._builtin_tool_names()))
            always_keep.append(DECISION_TOOL_NAME)
        always_keep = list(dict.fromkeys(always_keep))
        always_keep_recommend = [
            str(item)
            for item in (self._setting("always_keep_recommend_tools", []) or [])
            if str(item) in always_keep
        ]
        always_keep_subagents = [
            str(item) for item in (self._setting("always_keep_subagents", []) or [])
        ]
        always_keep_recommend_subagents = [
            str(item)
            for item in (self._setting("always_keep_recommend_subagents", []) or [])
            if str(item) in always_keep_subagents
        ]
        always_keep_mcp = [str(item) for item in (self._setting("always_keep_mcp", []) or [])]
        always_keep_recommend_mcp = [
            str(item)
            for item in (self._setting("always_keep_recommend_mcp", []) or [])
            if str(item) in always_keep_mcp
        ]
        always_keep_skills = [str(item) for item in (self._setting("always_keep_skills", []) or [])]
        always_keep_recommend_skills = [
            str(item)
            for item in (self._setting("always_keep_recommend_skills", []) or [])
            if str(item) in always_keep_skills
        ]
        proactive = {}
        for key, default in PROACTIVE_PAGE_DEFAULTS.items():
            value = self._setting(key, default)
            if isinstance(default, list):
                value = [str(item) for item in value] if isinstance(value, (list, tuple)) else []
            elif isinstance(default, bool):
                value = self._bool(key, default)
            elif isinstance(default, int):
                value = self._int(key, default)
            elif isinstance(default, float):
                value = self._float(key, default)
            else:
                value = str(value if value is not None else default)
            proactive[key] = value
        return json_response(
            {
                "decision_provider_id": str(self._setting("decision_provider_id", "") or ""),
                "decision_providers": decision_provider_catalog(self.context),
                "always_keep_tools": always_keep,
                "message_interjection_enabled": self._bool("message_interjection_enabled", True),
                "life": normalize_life(dict(self._setting("life", {}) or {})),
                "always_keep_recommend_tools": always_keep_recommend,
                "always_keep_subagents": always_keep_subagents,
                "always_keep_recommend_subagents": always_keep_recommend_subagents,
                "always_keep_mcp": always_keep_mcp,
                "always_keep_recommend_mcp": always_keep_recommend_mcp,
                "always_keep_skills": always_keep_skills,
                "always_keep_recommend_skills": always_keep_recommend_skills,
                "jev_pre_prompt": self._policy(),
                "main_llm_post_prompt": self._main_llm_post_prompt(),
                "tool_filter_enabled": self._bool("tool_filter_enabled", True),
                "mcp_filter_enabled": self._bool("mcp_filter_enabled", True),
                "skill_filter_enabled": self._bool("skill_filter_enabled", True),
                "subagent_filter_enabled": self._bool("subagent_filter_enabled", False),
                "tool_decision_enabled": self._bool("tool_decision_enabled", True),
                "mcp_decision_enabled": self._bool("mcp_decision_enabled", True),
                "skill_decision_enabled": self._bool("skill_decision_enabled", True),
                "subagent_decision_enabled": self._bool("subagent_decision_enabled", True),
                "tools_scope": self._configured_scope_values("tools_scope"),
                "tools_scope_blacklist": self._bool("tools_scope_blacklist", True),
                "tool_noul_threshold": self._float(
                    "tool_noul_threshold", DEFAULT_TOOL_NOUL_THRESHOLD
                ),
                "tools_subagents_decision_enabled": self._bool(
                    "tools_subagents_decision_enabled", True
                ),
                "proactive_reply_enabled": self._bool("proactive_reply_enabled", False),
                "conversation_flow_enabled": self._bool("conversation_flow_enabled", True),
                "ui_liquid_glass": self._setting("ui_liquid_glass", False) is True,
                "proactive": proactive,
                "dialogue_enhancement": self._output_page_settings(),
            }
        )

    async def page_save_settings(self):
        from astrbot.api.web import error_response, json_response, request

        self._refresh_native_provider_schema()
        payload = await request.json(default={})
        if not isinstance(payload, Mapping):
            return error_response("request body must be an object", status_code=400)
        life_values = None
        if "life" in payload:
            try:
                life_values = normalize_life(payload["life"])
            except (ValueError, TypeError) as exc:
                return error_response(str(exc), status_code=400)
        interjection = payload.get("message_interjection_enabled")
        if interjection is not None and not isinstance(interjection, bool):
            return error_response("消息接话开关必须是布尔值", status_code=400)
        values = payload.get("always_keep_tools")
        recommendation_values = payload.get("always_keep_recommend_tools", [])
        mcp_values = payload.get("always_keep_mcp", [])
        mcp_recommendation_values = payload.get("always_keep_recommend_mcp", [])
        skill_values = payload.get("always_keep_skills", [])
        skill_recommendation_values = payload.get("always_keep_recommend_skills", [])
        subagent_values = payload.get("always_keep_subagents", [])
        subagent_recommendation_values = payload.get("always_keep_recommend_subagents", [])
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            return error_response("always_keep_tools must be a string list", status_code=400)
        if not isinstance(recommendation_values, list) or any(
            not isinstance(item, str) for item in recommendation_values
        ):
            return error_response(
                "always_keep_recommend_tools must be a string list", status_code=400
            )
        if not isinstance(mcp_values, list) or any(
            not isinstance(item, str) for item in mcp_values
        ):
            return error_response("always_keep_mcp must be a string list", status_code=400)
        if not isinstance(mcp_recommendation_values, list) or any(
            not isinstance(item, str) for item in mcp_recommendation_values
        ):
            return error_response(
                "always_keep_recommend_mcp must be a string list", status_code=400
            )
        if not isinstance(skill_values, list) or any(
            not isinstance(item, str) for item in skill_values
        ):
            return error_response("always_keep_skills must be a string list", status_code=400)
        if not isinstance(skill_recommendation_values, list) or any(
            not isinstance(item, str) for item in skill_recommendation_values
        ):
            return error_response(
                "always_keep_recommend_skills must be a string list", status_code=400
            )
        if not isinstance(subagent_values, list) or any(
            not isinstance(item, str) for item in subagent_values
        ):
            return error_response("always_keep_subagents must be a string list", status_code=400)
        if not isinstance(subagent_recommendation_values, list) or any(
            not isinstance(item, str) for item in subagent_recommendation_values
        ):
            return error_response(
                "always_keep_recommend_subagents must be a string list", status_code=400
            )
        jev_pre_prompt = payload.get("jev_pre_prompt")
        decision_provider_id = payload.get("decision_provider_id")
        main_llm_post_prompt = payload.get("main_llm_post_prompt")
        tool_filter_enabled = payload.get("tool_filter_enabled")
        mcp_filter_enabled = payload.get("mcp_filter_enabled")
        skill_filter_enabled = payload.get("skill_filter_enabled")
        subagent_filter_enabled = payload.get("subagent_filter_enabled")
        tool_decision_enabled = payload.get("tool_decision_enabled")
        mcp_decision_enabled = payload.get("mcp_decision_enabled")
        skill_decision_enabled = payload.get("skill_decision_enabled")
        subagent_decision_enabled = payload.get("subagent_decision_enabled")
        tools_scope = payload.get("tools_scope")
        tools_scope_blacklist = payload.get("tools_scope_blacklist")
        tool_noul_threshold = payload.get("tool_noul_threshold")
        if jev_pre_prompt is not None and not isinstance(jev_pre_prompt, str):
            return error_response("jev_pre_prompt must be a string", status_code=400)
        if decision_provider_id is not None and (
            not isinstance(decision_provider_id, str) or len(decision_provider_id) > 300
        ):
            return error_response("decision_provider_id must be a string", status_code=400)
        if main_llm_post_prompt is not None and not isinstance(main_llm_post_prompt, str):
            return error_response("main_llm_post_prompt must be a string", status_code=400)
        if tool_filter_enabled is not None and not isinstance(tool_filter_enabled, bool):
            return error_response("tool_filter_enabled must be a boolean", status_code=400)
        if mcp_filter_enabled is not None and not isinstance(mcp_filter_enabled, bool):
            return error_response("mcp_filter_enabled must be a boolean", status_code=400)
        if skill_filter_enabled is not None and not isinstance(skill_filter_enabled, bool):
            return error_response("skill_filter_enabled must be a boolean", status_code=400)
        if subagent_filter_enabled is not None and not isinstance(subagent_filter_enabled, bool):
            return error_response("subagent_filter_enabled must be a boolean", status_code=400)
        if tool_decision_enabled is not None and not isinstance(tool_decision_enabled, bool):
            return error_response("tool_decision_enabled must be a boolean", status_code=400)
        if mcp_decision_enabled is not None and not isinstance(mcp_decision_enabled, bool):
            return error_response("mcp_decision_enabled must be a boolean", status_code=400)
        if skill_decision_enabled is not None and not isinstance(skill_decision_enabled, bool):
            return error_response("skill_decision_enabled must be a boolean", status_code=400)
        if subagent_decision_enabled is not None and not isinstance(
            subagent_decision_enabled, bool
        ):
            return error_response("subagent_decision_enabled must be a boolean", status_code=400)
        if tools_scope is not None and (
            not isinstance(tools_scope, list)
            or any(not isinstance(item, str) for item in tools_scope)
        ):
            return error_response("tools_scope must be a string list", status_code=400)
        if tools_scope_blacklist is not None and not isinstance(tools_scope_blacklist, bool):
            return error_response("tools_scope_blacklist must be a boolean", status_code=400)
        if tool_noul_threshold is not None and (
            isinstance(tool_noul_threshold, bool)
            or not isinstance(tool_noul_threshold, (int, float))
            or not math.isfinite(float(tool_noul_threshold))
        ):
            return error_response("tool_noul_threshold must be a number", status_code=400)
        proactive_payload = payload.get("proactive", {})
        if not isinstance(proactive_payload, Mapping):
            return error_response("proactive must be an object", status_code=400)
        dialogue_payload = payload.get("dialogue_enhancement", {})
        if not isinstance(dialogue_payload, Mapping):
            return error_response("dialogue_enhancement must be an object", status_code=400)
        proactive_payload = dict(proactive_payload)
        for key in (
            "conversation_flow_analysis_enabled",
            "conversation_flow_window",
            "conversation_flow_scope",
            "conversation_flow_scope_blacklist",
        ):
            if key in dialogue_payload:
                proactive_payload[key] = dialogue_payload[key]
        input_bool_keys = (
            "optimize_tool_history_context",
            "disable_group_at_bot_wake",
            "disable_group_at_bot_wake_all_groups",
            "disable_group_reply_to_bot_wake",
            "disable_group_reply_to_bot_wake_all_groups",
        )
        for key in input_bool_keys:
            if key in dialogue_payload and not isinstance(dialogue_payload[key], bool):
                return error_response(f"{key} must be a boolean", status_code=400)
        input_group_keys = (
            "disable_group_at_bot_wake_group_ids",
            "disable_group_reply_to_bot_wake_group_ids",
        )
        cleaned_input_groups: dict[str, list[str]] = {}
        for key in input_group_keys:
            if key not in dialogue_payload:
                continue
            value = dialogue_payload[key]
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                return error_response(f"{key} must be a string list", status_code=400)
            cleaned_input_groups[key] = list(
                dict.fromkeys(item.strip() for item in value if item.strip())
            )[:500]
        pending_multimodal: dict[str, Any] | None = None
        multimodal_payload = dialogue_payload.get("multimodal")
        if multimodal_payload is not None:
            if not isinstance(multimodal_payload, Mapping):
                return error_response("multimodal must be an object", status_code=400)
            from .multimodal import validate_page_payload

            cleaned_mm, mm_error = validate_page_payload(multimodal_payload)
            if mm_error:
                return error_response(mm_error, status_code=400)
            if cleaned_mm:
                # Staged; the whole config is written once validation succeeds.
                pending_multimodal = {**self._multimodal_settings(), **cleaned_mm}
        output_enabled = dialogue_payload.get("output_enhancement_enabled")
        output_scope = dialogue_payload.get("output_scope")
        output_scope_blacklist = dialogue_payload.get("output_scope_blacklist")
        output_pipeline = dialogue_payload.get("output_pipeline")
        if output_enabled is not None and not isinstance(output_enabled, bool):
            return error_response("output_enhancement_enabled must be a boolean", status_code=400)
        if output_scope is not None and (
            not isinstance(output_scope, list)
            or any(not isinstance(item, str) for item in output_scope)
        ):
            return error_response("output_scope must be a string list", status_code=400)
        if output_scope_blacklist is not None and not isinstance(output_scope_blacklist, bool):
            return error_response("output_scope_blacklist must be a boolean", status_code=400)
        if output_pipeline is not None and not isinstance(output_pipeline, Mapping):
            return error_response("output_pipeline must be an object", status_code=400)
        if output_pipeline is not None:
            try:
                if len(json.dumps(output_pipeline, ensure_ascii=False)) > 300_000:
                    return error_response("output_pipeline is too large", status_code=400)
            except (TypeError, ValueError):
                return error_response("output_pipeline must be JSON serializable", status_code=400)
        cleaned_proactive: dict[str, Any] = {}
        for key, default in PROACTIVE_PAGE_DEFAULTS.items():
            if key not in proactive_payload:
                continue
            value = proactive_payload[key]
            if isinstance(default, list):
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    return error_response(f"{key} must be a string list", status_code=400)
                cleaned_proactive[key] = list(
                    dict.fromkeys(item.strip() for item in value if item.strip())
                )[:500]
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    return error_response(f"{key} must be a boolean", status_code=400)
                cleaned_proactive[key] = value
            elif isinstance(default, int):
                if isinstance(value, bool) or not isinstance(value, int):
                    return error_response(f"{key} must be an integer", status_code=400)
                if key == "conversation_flow_window":
                    value = min(30, max(2, value))
                cleaned_proactive[key] = value
            elif isinstance(default, float):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    return error_response(f"{key} must be a number", status_code=400)
                cleaned_proactive[key] = float(value)
            else:
                if not isinstance(value, str):
                    return error_response(f"{key} must be a string", status_code=400)
                cleaned_proactive[key] = value[:1000]
        available = {
            str(getattr(tool, "name", ""))
            for tool in self._manager_tool_objects()
            if getattr(tool, "name", None) and not is_handoff_tool(tool) and not is_mcp_tool(tool)
        }
        available.add(DECISION_TOOL_NAME)
        available.update(self._builtin_tool_names())
        cleaned = list(
            dict.fromkeys(
                item.strip() for item in values if item.strip() and item.strip() in available
            )
        )[:500]
        cleaned_recommend = list(
            dict.fromkeys(
                item.strip()
                for item in recommendation_values
                if item.strip() in cleaned and item.strip() in available
            )
        )[:500]
        available_mcp = {
            str(getattr(tool, "name", ""))
            for tool in self._manager_tool_objects()
            if getattr(tool, "name", None) and is_mcp_tool(tool)
        }
        cleaned_mcp = list(
            dict.fromkeys(
                item.strip()
                for item in mcp_values
                if item.strip() and item.strip() in available_mcp
            )
        )[:500]
        cleaned_recommend_mcp = list(
            dict.fromkeys(
                item.strip() for item in mcp_recommendation_values if item.strip() in cleaned_mcp
            )
        )[:500]
        available_skills = {
            str(getattr(skill, "name", ""))
            for skill in self._skill_objects()
            if getattr(skill, "name", None)
        }
        cleaned_skills = list(
            dict.fromkeys(
                item.strip()
                for item in skill_values
                if item.strip() and item.strip() in available_skills
            )
        )[:500]
        cleaned_recommend_skills = list(
            dict.fromkeys(
                item.strip()
                for item in skill_recommendation_values
                if item.strip() in cleaned_skills
            )
        )[:500]
        available_subagents = {
            str(getattr(tool, "name", "")) for tool in self._subagent_tool_objects()
        }
        cleaned_subagents = list(
            dict.fromkeys(
                item.strip()
                for item in subagent_values
                if item.strip() and item.strip() in available_subagents
            )
        )[:500]
        cleaned_recommend_subagents = list(
            dict.fromkeys(
                item.strip()
                for item in subagent_recommendation_values
                if item.strip() in cleaned_subagents
            )
        )[:500]
        detail_values = {
            key: self._setting(key, default) for key, default in DETAIL_DEFAULTS.items()
        }
        if life_values is not None:
            detail_values["life"] = life_values
        if interjection is not None:
            detail_values["message_interjection_enabled"] = interjection
        detail_values.update(
            always_keep_tools=cleaned,
            always_keep_recommend_tools=cleaned_recommend,
            always_keep_mcp=cleaned_mcp,
            always_keep_recommend_mcp=cleaned_recommend_mcp,
            always_keep_skills=cleaned_skills,
            always_keep_recommend_skills=cleaned_recommend_skills,
            always_keep_subagents=cleaned_subagents,
            always_keep_recommend_subagents=cleaned_recommend_subagents,
            always_keep_tools_customized=True,
        )
        if tool_filter_enabled is not None:
            detail_values["tool_filter_enabled"] = tool_filter_enabled
        if mcp_filter_enabled is not None:
            detail_values["mcp_filter_enabled"] = mcp_filter_enabled
        if skill_filter_enabled is not None:
            detail_values["skill_filter_enabled"] = skill_filter_enabled
        if subagent_filter_enabled is not None:
            detail_values["subagent_filter_enabled"] = subagent_filter_enabled
        if tool_decision_enabled is not None:
            detail_values["tool_decision_enabled"] = tool_decision_enabled
        if mcp_decision_enabled is not None:
            detail_values["mcp_decision_enabled"] = mcp_decision_enabled
        if skill_decision_enabled is not None:
            detail_values["skill_decision_enabled"] = skill_decision_enabled
        if subagent_decision_enabled is not None:
            detail_values["subagent_decision_enabled"] = subagent_decision_enabled
        if tools_scope is not None:
            detail_values["tools_scope"] = list(
                dict.fromkeys(item.strip() for item in tools_scope if item.strip())
            )[:500]
        if tools_scope_blacklist is not None:
            detail_values["tools_scope_blacklist"] = tools_scope_blacklist
        if jev_pre_prompt is not None:
            detail_values["jev_pre_prompt"] = jev_pre_prompt[:50000]
        if decision_provider_id is not None:
            selected_provider = decision_provider_id.strip()
            available_decision_providers = {
                row["id"] for row in decision_provider_catalog(self.context)
            }
            current_provider = str(self._setting("decision_provider_id", "") or "")
            if selected_provider and selected_provider not in available_decision_providers:
                # Keep a removed provider ID only when it is already saved;
                # this lets unrelated settings save safely while surfacing
                # the stale selection in the UI for explicit correction.
                if selected_provider != current_provider:
                    return error_response(
                        "所选模型提供商不是 AstrBot 当前可用的 JEV Provider",
                        status_code=400,
                    )
            provider_to_write = selected_provider
        if main_llm_post_prompt is not None:
            detail_values["main_llm_post_prompt"] = main_llm_post_prompt[:50000]
        if tool_noul_threshold is not None:
            detail_values["tool_noul_threshold"] = min(1.0, max(0.0, float(tool_noul_threshold)))
        detail_values.update(cleaned_proactive)
        for key in input_bool_keys:
            if key in dialogue_payload:
                detail_values[key] = dialogue_payload[key]
        detail_values.update(cleaned_input_groups)
        if output_enabled is not None:
            detail_values["output_enhancement_enabled"] = output_enabled
        if output_scope is not None:
            detail_values["output_scope"] = list(
                dict.fromkeys(item.strip() for item in output_scope if item.strip())
            )[:500]
        if output_scope_blacklist is not None:
            detail_values["output_scope_blacklist"] = output_scope_blacklist
        from .outputpro_runtime import _deep_merge, output_config_defaults

        if output_pipeline is not None:
            detail_values["output_pipeline"] = _deep_merge(
                output_config_defaults(), dict(output_pipeline)
            )
        # The output pipeline has one built-in execution order.  Keep the
        # legacy field for schema compatibility, but never persist a
        # user-editable false value, including when the current save payload
        # does not include output settings.
        configured_output = detail_values.get("output_pipeline")
        normalized_output = _deep_merge(
            output_config_defaults(),
            dict(configured_output) if isinstance(configured_output, Mapping) else {},
        )
        output_pipeline_node = normalized_output.get("pipeline")
        if not isinstance(output_pipeline_node, dict):
            output_pipeline_node = {}
            normalized_output["pipeline"] = output_pipeline_node
        output_pipeline_node["lock_order"] = True
        detail_values["output_pipeline"] = normalized_output
        if pending_multimodal is not None:
            detail_values["multimodal"] = pending_multimodal
        # Commit first: a failed disk write must not change the running policy
        # or prune proactive history while reporting a failed save to the UI.
        previous_detail_settings = copy.deepcopy(self._detail_settings)
        settings_file_existed = self._settings_path.exists()
        try:
            self._save_detail_settings(detail_values)
        except SETTINGS_WRITE_ERRORS as exc:
            logger.error("Decision Engine WebUI settings save failed: %s", _safe_error(exc))
            return error_response("配置写入失败，原设置保持不变", status_code=500)
        if decision_provider_id is not None:
            # Persist the selected Provider through AstrBot's own config
            # object when available. This keeps the native field and the
            # dashboard selection synchronized without touching credentials.
            previous_native_present = "decision_provider_id" in self.config
            previous_native_value = self.config.get("decision_provider_id")
            self.config["decision_provider_id"] = provider_to_write
            save_config = getattr(self.config, "save_config", None)
            if not callable(save_config):
                logger.warning(
                    "Decision Engine: this config cannot be saved, so the JEV Provider "
                    "selection only lasts until the plugin reloads."
                )
            else:
                try:
                    save_config()
                except Exception as exc:
                    # The detail file was committed first so a native config
                    # write failure must roll both stores back.  Otherwise a
                    # failed response would leave the dashboard and AstrBot
                    # disagreeing about the selected Provider.
                    if previous_native_present:
                        self.config["decision_provider_id"] = previous_native_value
                    else:
                        self.config.pop("decision_provider_id", None)
                    try:
                        save_config()
                    except Exception as rollback_exc:
                        logger.error(
                            "Decision Engine could not roll back AstrBot's JEV Provider config: %s",
                            _safe_error(rollback_exc),
                        )
                    try:
                        if settings_file_existed:
                            self._save_detail_settings(previous_detail_settings)
                        else:
                            self._detail_settings = previous_detail_settings
                            self._settings_path.unlink(missing_ok=True)
                    except SETTINGS_WRITE_ERRORS as rollback_exc:
                        logger.error(
                            "Decision Engine could not roll back JEV Provider settings: %s",
                            _safe_error(rollback_exc),
                        )
                    logger.error(
                        "Decision Engine JEV Provider selection could not be persisted: %s",
                        _safe_error(exc),
                    )
                    return error_response("Jev 模型提供商写入失败，原设置保持不变", status_code=500)
        self.proactive.reconfigure(
            self._int("proactive_history_max_messages", 12),
            self._int("proactive_max_sessions", 500),
        )
        if self.provider is not None and decision_provider_id is not None:
            self.provider.provider_id = str(self._setting("decision_provider_id", "") or "").strip()
        if pending_multimodal is not None and self._multimodal is not None:
            self._multimodal.refresh(self._multimodal_settings())
            self._sync_multimodal_tools(self._multimodal)
            try:
                await self._apply_multimodal_tool_state(self._multimodal_settings())
            except Exception as exc:  # pragma: no cover - host API differences
                logger.warning("Decision Engine 多模态工具状态刷新失败：%s", _safe_error(exc))
        await self._configure_output_runtime()
        self._configure_input_enhancements()
        # Restarting the life runtime cancels running WebUI jobs and drops
        # queued delayed replies, so only do it when its settings changed.
        if life_values is not None and (
            self._life_runtime is None
            or life_values != getattr(self._life_runtime, "config", None)
            # started while 主动发起 was off (or on): the scheduler only starts in
            # initialize(), so a switch change needs a fresh runtime
            or getattr(self._life_runtime, "started_enabled", None)
            != getattr(self._life_runtime, "enabled", None)
        ):
            try:
                await self._configure_life_runtime()
            except Exception as exc:
                logger.error("Decision Engine 生活设置已保存，后台启动失败：%s", _safe_error(exc))
                return json_response({"saved": True, "runtime_warning": _safe_error(exc)})
        return json_response(
            {
                "saved": True,
                "decision_provider_id": str(self._setting("decision_provider_id", "") or ""),
                "decision_providers": decision_provider_catalog(self.context),
                "always_keep_tools": cleaned,
                "always_keep_recommend_tools": cleaned_recommend,
                "always_keep_mcp": cleaned_mcp,
                "always_keep_recommend_mcp": cleaned_recommend_mcp,
                "always_keep_skills": cleaned_skills,
                "always_keep_recommend_skills": cleaned_recommend_skills,
                "always_keep_subagents": cleaned_subagents,
                "always_keep_recommend_subagents": cleaned_recommend_subagents,
                "tool_filter_enabled": self._bool("tool_filter_enabled", True),
                "mcp_filter_enabled": self._bool("mcp_filter_enabled", True),
                "skill_filter_enabled": self._bool("skill_filter_enabled", True),
                "subagent_filter_enabled": self._bool("subagent_filter_enabled", False),
                "tool_decision_enabled": self._bool("tool_decision_enabled", True),
                "mcp_decision_enabled": self._bool("mcp_decision_enabled", True),
                "skill_decision_enabled": self._bool("skill_decision_enabled", True),
                "subagent_decision_enabled": self._bool("subagent_decision_enabled", True),
                "tools_scope": self._configured_scope_values("tools_scope"),
                "tools_scope_blacklist": self._bool("tools_scope_blacklist", True),
                "tool_noul_threshold": self._float(
                    "tool_noul_threshold", DEFAULT_TOOL_NOUL_THRESHOLD
                ),
                "proactive": {
                    key: self._setting(key, default)
                    for key, default in PROACTIVE_PAGE_DEFAULTS.items()
                },
                "dialogue_enhancement": self._output_page_settings(),
            }
        )

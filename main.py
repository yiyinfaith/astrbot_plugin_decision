"""AstrBot entry point of the Decision Engine plugin.

Event handlers are bound by module, so every decorated handler stays here;
plain methods live in the decision.plugin_* mixins.
"""

# ruff: noqa: F401, F403, F405
from __future__ import annotations

from .decision.plugin_life import LifeMixin
from .decision.plugin_routing import RoutingMixin
from .decision.plugin_shared import *
from .decision.plugin_shared import (
    _ON_DECORATING_RESULT,
    DEFAULT_POLICY,
    DEFAULT_TOOL_NOUL_THRESHOLD,
    DETAIL_DEFAULTS,
    INSPECT_AFTER_ACTIONS,
    NATIVE_PROVIDER_KEY,
    OUTPUT_PAGE_DEFAULTS,
    PLUGIN_DISPLAY_NAME,
    PLUGIN_NAME,
    PROACTIVE_PAGE_DEFAULTS,
    PROACTIVE_REPLY_EXTRA,
    SETTINGS_WRITE_ERRORS,
    _safe_error,
)
from .decision.plugin_webui import WebUIMixin
from .decision.input_enhancement.group_wake_suppression import GroupWakeSuppressionModule
from .decision.input_enhancement.tool_history_context import ToolHistoryContextModule


class DecisionFunctionTool(FunctionTool):
    """Keep framework ownership, disable and reload tracking on this plugin."""


# Plugin metadata (name, author, version) comes from metadata.yaml; the
# register_star decorator is deprecated in AstrBot.
class DecisionPlugin(RoutingMixin, LifeMixin, WebUIMixin, Star):
    """AstrBot integration layer for a provider-neutral decision engine."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self._settings_path = self._get_settings_path(config)
        usage_meter.install(
            usage_meter.UsageMeter(
                self._settings_path.parent.parent / "plugin_data" / PLUGIN_NAME / "usage.json"
            )
        )
        # A Provider saved in the detail file by older dashboards; read only
        # by the one-time migration into the native config below.
        self._loaded_legacy_provider: str | None = None
        self._detail_settings = self._load_detail_settings()
        legacy_provider_id = self._loaded_legacy_provider
        # AstrBot also renders live config keys. Never merge WebUI settings
        # into this framework-owned object, even only in memory.
        legacy = {key: config[key] for key in DETAIL_DEFAULTS if key in config}
        if legacy:
            try:
                self._save_detail_settings({**legacy, **self._detail_settings})
            except SETTINGS_WRITE_ERRORS as exc:
                # Do not remove framework-owned values until the durable
                # migration succeeds.  Keeping them in memory preserves the
                # user's policy for this run and allows a later retry.
                self._detail_settings = {**legacy, **self._detail_settings}
                logger.error(
                    "Decision Engine could not migrate legacy settings; native values were kept: %s",
                    _safe_error(exc),
                )
            else:
                for key in legacy:
                    config.pop(key, None)
        if legacy_provider_id is not None:
            self._migrate_legacy_provider(config, legacy_provider_id)
        self._refresh_native_provider_schema()
        # The host owns the JEV transport, credentials and model.  This
        # adapter only translates the plugin's questions to AstrBot's native
        # DecisionService.
        self.provider: AstrBotDecisionProvider | None = None
        self._ready = False
        self._active_reply_warning_emitted = False
        self._output_runtime: Any | None = None
        self._life_runtime: Any | None = None
        self._multimodal: Any | None = None
        self._multimodal_tools: list[Any] = []
        self._tool_history_context = ToolHistoryContextModule(logger=logger)
        self._group_wake_suppression = GroupWakeSuppressionModule(
            logger=logger,
            disable_at_bot_wake=self._bool("disable_group_at_bot_wake", False),
            disable_at_bot_wake_scope_blacklist=self._bool(
                "disable_group_at_bot_wake_scope_blacklist", True
            ),
            disable_at_bot_wake_group_ids=self._setting(
                "disable_group_at_bot_wake_group_ids", []
            ),
            disable_reply_to_bot_wake=self._bool(
                "disable_group_reply_to_bot_wake", False
            ),
            disable_reply_to_bot_wake_scope_blacklist=self._bool(
                "disable_group_reply_to_bot_wake_scope_blacklist", True
            ),
            disable_reply_to_bot_wake_group_ids=self._setting(
                "disable_group_reply_to_bot_wake_group_ids", []
            ),
        )
        self._life_media: Any | None = None
        self._personal_network: Any | None = None
        self._life_lock = asyncio.Lock()
        self._life_asset_lock = asyncio.Lock()
        self._life_jobs: dict[str, dict[str, Any]] = {}
        # Track the background WebUI jobs so a hot reload can cancel them instead
        # of leaving orphaned tasks mutating a dead runtime.
        self._life_tasks: set[asyncio.Task] = set()
        self._life_job_limit = 20
        self.proactive = ProactiveState(
            self._int("proactive_history_max_messages", 12),
            self._int("proactive_max_sessions", 500),
        )
        self._last_call_latency_ms: float | None = None
        self._call_count = 0
        self._failure_count = 0

        self._decision_tool = DecisionFunctionTool(
            name=DECISION_TOOL_NAME,
            description=(
                "Use the configured Decision Engine for one fast structured noul, choice, "
                "or score judgment. It uses AstrBot native JEV and never executes a tool."
            ),
            parameters=decision_tool_parameters(),
            handler=self._run_decision_tool,
        )
        # Context.add_llm_tools is the current AstrBot API. This tool is also
        # explicitly re-added to a request after filtering so it stays available.
        self.context.add_llm_tools(self._decision_tool)
        self._life_tools = []
        for name, description, handler, properties, required in (
            (
                "take_photo",
                "生成并发送一张照片：可以拍自己、拍别人、拍周围的景物或物品，"
                "也可以生成手机截图、聊天截图等屏幕内容。通过 aigen 使用当前 AstrBot 人格的"
                "外貌参考图与今日穿搭；不是真实摄像头。",
                self._take_photo,
                {
                    "scene": {
                        "type": "string",
                        "description": "画面内容、动作、构图与氛围要求，可留空使用当前活动。",
                    },
                    "subject": {
                        "type": "string",
                        "description": (
                            "拍摄对象：self=自己，other=别人，scene=景色/环境，"
                            "object=物品，screenshot=手机截图或录屏画面。"
                        ),
                        "enum": ["self", "other", "scene", "object", "screenshot"],
                    },
                    "count": {
                        "type": "integer",
                        "description": "生成张数；必须在管理员配置的数量区间内，越界会被自动收敛。",
                    },
                },
                [],
            ),
            (
                "take_video",
                "生成并发送一段短视频：可以拍自己、拍别人、拍场景，也可以生成手机录屏。"
                "通过 aigen 保持外貌与今日穿搭；不访问真实摄像头。",
                self._take_video,
                {
                    "scene": {
                        "type": "string",
                        "description": "视频内容、动作和镜头要求，可留空。",
                    },
                    "subject": {
                        "type": "string",
                        "description": (
                            "拍摄对象：self=自己，other=别人，scene=景色/环境，"
                            "object=物品，screenshot=手机录屏画面。"
                        ),
                        "enum": ["self", "other", "scene", "object", "screenshot"],
                    },
                    "count": {
                        "type": "integer",
                        "description": "生成段数；必须在管理员配置的数量区间内，越界会被自动收敛。",
                    },
                },
                [],
            ),
            (
                "send_voice",
                "生成并发送一段音频，用于偶尔开口说话、朗读或唱歌；也可以生成环境音等描述性音频。"
                "通过 aigen 的音频接口实现，可指定音色。",
                self._send_voice,
                {
                    "text": {
                        "type": "string",
                        "description": "要说出、朗读或唱出的内容；提示词模式下则是音频的描述。",
                    },
                    "style": {
                        "type": "string",
                        "description": "语气、情绪与表达风格，例如温柔、兴奋、轻声哼唱，可留空。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "生成段数；必须在管理员配置的数量区间内，越界会被自动收敛。",
                    },
                },
                ["text"],
            ),
            (
                "get_virtual_daily_schedule",
                "查询当前 AstrBot 人格的今日日程、穿搭和当前活动。",
                self._get_virtual_daily_schedule,
                {},
                [],
            ),
            (
                "get_long_term_timeline",
                "查询当前 AstrBot 人格已批准的长期生活阶段。",
                self._get_long_term_timeline,
                {},
                [],
            ),
            (
                "add_schedule_requirement",
                "仅当当前管理员明确要求时，为未来日期添加一次性日程要求。",
                self._add_schedule_requirement,
                {
                    "date_range": {
                        "type": "string",
                        "description": "日期或区间，例如 2026-12-01..2026-12-03",
                    },
                    "requirement": {"type": "string", "description": "管理员要求，最多500字"},
                },
                ["date_range", "requirement"],
            ),
        ):
            parameters = {"type": "object", "properties": properties}
            if required:
                parameters["required"] = required
            tool = DecisionFunctionTool(
                name=name, description=description, parameters=parameters, handler=handler
            )
            self._life_tools.append(tool)
            self.context.add_llm_tools(tool)
        from .decision.life_control_tools import control_tools

        for name, description, handler, properties, required in control_tools(self):
            tool = DecisionFunctionTool(
                name=name,
                description=description,
                parameters={"type": "object", "properties": properties, "required": required},
                handler=handler,
            )
            self._life_tools.append(tool)
            self.context.add_llm_tools(tool)
        # The merged Personal Network tools keep their upstream names so stored
        # prompts and habits keep working. They are registered unconditionally
        # and removed per request when the feature or a sub-switch is off.
        for tool in (
            DecisionFunctionTool(
                name="update_personal_network",
                description=(
                    "更新当前 AstrBot 人格的人际网络：新增或修正人物、人物之间的关系，"
                    "以及已经发生的共同经历。只写入对话中明确、持久的人物与关系事实。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "characters": {
                            "type": "array",
                            "description": "要新增或更新的人物列表，最多 20 个。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ref": {
                                        "type": "string",
                                        "description": "本次调用内的临时引用名",
                                    },
                                    "id": {"type": "string", "description": "已存在人物的 UUID"},
                                    "name": {"type": "string", "description": "人物姓名"},
                                    "aliases": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "别名或昵称",
                                    },
                                    "bio": {"type": "string", "description": "人物简介"},
                                    "personality": {"type": "string", "description": "性格"},
                                    "preferences": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "偏好",
                                    },
                                    "facts": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "确定的事实",
                                    },
                                    "current_sender": {
                                        "type": "boolean",
                                        "description": "该人物是否为当前消息发送者",
                                    },
                                },
                                "required": ["name"],
                            },
                        },
                        "relationships": {
                            "type": "array",
                            "description": "要新增或更新的人物关系列表，最多 30 条。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {
                                        "type": "string",
                                        "description": "主体，可用 persona 表示当前人格",
                                    },
                                    "target": {"type": "string", "description": "关系目标"},
                                    "type": {"type": "string", "description": "关系类型"},
                                    "strength": {"type": "integer", "description": "亲密度 0-100"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["active", "ended", "uncertain"],
                                    },
                                    "description": {"type": "string", "description": "关系说明"},
                                },
                                "required": ["source", "target", "type"],
                            },
                        },
                        "interactions": {
                            "type": "array",
                            "description": "要记录的共同经历，最多 30 条。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "participants": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "参与者，至少两位",
                                    },
                                    "type": {"type": "string", "description": "经历类型"},
                                    "summary": {"type": "string", "description": "经历摘要"},
                                    "occurred_at": {
                                        "type": "string",
                                        "description": "ISO 8601 时间",
                                    },
                                    "importance": {
                                        "type": "integer",
                                        "description": "重要度 0-100",
                                    },
                                    "emotional_tone": {"type": "string", "description": "情绪基调"},
                                },
                                "required": ["participants", "type", "summary"],
                            },
                        },
                    },
                },
                handler=self.network_update_tool,
            ),
            DecisionFunctionTool(
                name="query_personal_network",
                description="查询当前 AstrBot 人格的人际网络：人物、关系与共同经历。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要查询的人物姓名、别名或关键词，留空返回总览。",
                        }
                    },
                },
                handler=self.network_query_tool,
            ),
        ):
            self._life_tools.append(tool)
            self.context.add_llm_tools(tool)
        for endpoint, handler, methods in (
            ("life", self.page_life, ["GET"]),
            ("life/action", self.page_life_action, ["POST"]),
            ("life/control", self.page_life_control, ["POST"]),
            ("life/appearance", self.page_life_appearance, ["POST"]),
            ("life/personas", self.page_life_personas, ["GET"]),
            ("life/sessions", self.page_life_sessions, ["GET"]),
            ("life/billing", self.page_life_billing, ["GET"]),
        ):
            self.context.register_web_api(
                f"/{PLUGIN_NAME}/{endpoint}", handler, methods, "Virtual Life WebUI"
            )

        self.context.register_web_api(
            f"/{PLUGIN_NAME}/ui-settings/save",
            self.page_save_ui_settings,
            ["POST"],
            "Save the plugin view display preference",
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/multimodal/models",
            self.page_multimodal_models,
            ["GET"],
            "Multi-modal model catalog",
        )

        self.context.register_web_api(
            f"/{PLUGIN_NAME}/tools",
            self.page_tools,
            ["GET"],
            "List tools available to the Decision Engine settings page",
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/settings",
            self.page_settings,
            ["GET"],
            "Read Decision Engine page settings",
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/settings/save",
            self.page_save_settings,
            ["POST"],
            "Save Decision Engine page settings",
        )
        for route, handler, methods in (
            ("network/personas", self.network_personas, ["GET"]),
            ("network/network", self.network_network, ["GET"]),
            ("network/network/enabled", self.network_set_enabled, ["POST"]),
            ("network/character/save", self.network_save_character, ["POST"]),
            ("network/character/delete", self.network_delete_character, ["POST"]),
            ("network/character/merge", self.network_merge_characters, ["POST"]),
            ("network/relationship/save", self.network_save_relationship, ["POST"]),
            ("network/relationship/delete", self.network_delete_relationship, ["POST"]),
            ("network/life-event/save", self.network_save_life_event, ["POST"]),
            ("network/life-event/delete", self.network_delete_life_event, ["POST"]),
            ("network/generation/generate", self.network_generate, ["POST"]),
            ("network/generation/status", self.network_generation_status, ["POST"]),
            ("network/generation/validate", self.network_generation_validate, ["POST"]),
            ("network/generation/apply", self.network_generation_apply, ["POST"]),
            ("network/avatar/<character_id>", self.network_avatar, ["GET", "POST"]),
            ("network/export", self.network_export, ["GET"]),
            ("network/import/<persona_key>/preview", self.network_import_preview, ["POST"]),
            ("network/import/apply", self.network_import_apply, ["POST"]),
        ):
            self.context.register_web_api(
                f"/{PLUGIN_NAME}/{route}", handler, methods, f"Personal network {route}"
            )

    @staticmethod
    def _get_settings_path(config: AstrBotConfig) -> Path:
        config_path = getattr(config, "config_path", "")
        if config_path:
            return Path(str(config_path)).resolve().with_name(f"{PLUGIN_NAME}_settings.json")
        return Path(get_astrbot_config_path()) / f"{PLUGIN_NAME}_settings.json"

    def _load_detail_settings(self) -> dict[str, Any]:
        try:
            if not self._settings_path.is_file():
                return {}
            with self._settings_path.open(encoding="utf-8-sig") as file:
                value = json.load(file)
            payload = (
                {key: value[key] for key in DETAIL_DEFAULTS if key in value}
                if isinstance(value, Mapping)
                else {}
            )
            if isinstance(value, Mapping) and NATIVE_PROVIDER_KEY in value:
                # Read only by the startup migration into the native config.
                self._loaded_legacy_provider = str(value.get(NATIVE_PROVIDER_KEY) or "").strip()
            if "multimodal" in payload:
                from .decision.multimodal import normalize_settings

                payload["multimodal"] = normalize_settings(payload["multimodal"])
            return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Decision Engine could not read WebUI settings; defaults will be used: %s",
                _safe_error(exc),
            )
            return {}

    def _save_detail_settings(self, values: Mapping[str, Any]) -> None:
        payload = {key: values[key] for key in DETAIL_DEFAULTS if key in values}
        if "multimodal" in payload:
            from .decision.multimodal import normalize_settings

            payload["multimodal"] = normalize_settings(payload["multimodal"])
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=str(self._settings_path.parent),
            prefix=f".{self._settings_path.name}.",
            suffix=".tmp",
        )
        committed = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._settings_path)
            committed = True
        finally:
            if not committed:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        self._detail_settings = payload

    def _migrate_legacy_provider(self, config: AstrBotConfig, legacy_provider_id: str) -> None:
        """Move an old dashboard Provider into AstrBot's native config, once.

        The copy is dropped even if the native write fails: the value stays in
        the in-memory config (saved with the next native config save), and a
        retry on a later start could override a clear made in the meantime.
        """

        native_provider_id = str(config.get(NATIVE_PROVIDER_KEY, "") or "").strip()
        if not native_provider_id and legacy_provider_id:
            config[NATIVE_PROVIDER_KEY] = legacy_provider_id
            save_config = getattr(config, "save_config", None)
            if callable(save_config):
                try:
                    save_config()
                except Exception as exc:
                    logger.warning(
                        "Decision Engine could not persist the migrated JEV Provider selection: %s",
                        _safe_error(exc),
                    )
        try:
            self._save_detail_settings(self._detail_settings)
        except SETTINGS_WRITE_ERRORS as exc:
            logger.warning(
                "Decision Engine could not remove the dashboard copy of the JEV Provider: %s",
                _safe_error(exc),
            )

    def _setting(self, key: str, default: Any = None) -> Any:
        if key in DETAIL_DEFAULTS:
            return self._detail_settings.get(key, default)
        return self.config.get(key, default)

    def _refresh_native_provider_schema(self) -> None:
        """Populate the host config dropdown from live Decision Providers.

        AstrBot's generic config renderer supports static ``options`` but does
        not yet have a Decision-specific Provider selector.  Keep the schema
        fallback limited to the host default, then replace it in memory with
        the actual provider IDs exposed by the host manager.  This never writes
        provider credentials or changes the plugin config file's shape.
        """

        schema = getattr(self.config, "schema", None)
        if not isinstance(schema, dict):
            return
        field = schema.get("decision_provider_id")
        if not isinstance(field, dict):
            return
        rows = decision_provider_catalog(self.context)
        options = [""]
        labels = ["使用 AstrBot 默认判断 Provider"]
        subtitles = ["由 AstrBot 当前默认 Decision Provider 处理"]
        for row in rows:
            provider_id = str(row.get("id", "") or "").strip()
            if not provider_id or provider_id in options:
                continue
            options.append(provider_id)
            labels.append(str(row.get("name") or provider_id))
            subtitles.append(
                " · ".join(
                    value
                    for value in (
                        str(row.get("model", "") or "").strip(),
                        str(row.get("type", "") or "").strip(),
                    )
                    if value
                )
            )
        field["options"] = options
        field["labels"] = labels
        field["subtitles"] = subtitles

    async def initialize(self) -> None:
        self.provider = AstrBotDecisionProvider(
            self.context,
            provider_id=str(self._setting("decision_provider_id", "") or ""),
            retries=self._int("retries", 1),
            retry_backoff_sec=self._float("retry_backoff_seconds", 0.0),
            retry_logger=self._log_provider_retry,
        )
        self._ready = self.provider.configured
        if not self._ready:
            logger.warning(
                "Decision Engine 未找到 AstrBot 原生 JEV DecisionService；插件保持 fail-open。"
            )
        elif not decision_provider_catalog(self.context):
            logger.warning(
                "Decision Engine 未找到 AstrBot 原生 JEV Provider；请在 AstrBot 模型提供方中配置并选择。"
            )
        await self._configure_output_runtime()
        # Multi-modal understanding is independent of 主动对话: its tools and
        # auto-understanding have their own switches.
        try:
            await self._ensure_multimodal()
        except Exception as exc:
            logger.warning("Decision Engine 多模态初始化失败：%s", _safe_error(exc))
        self._configure_input_enhancements()
        # 主动发起 is optional: a failing life store must not take down the
        # fail-open Tool/MCP/Skill routing.  A later settings save retries it.
        try:
            await self._configure_life_runtime()
        except Exception as exc:
            logger.error("Decision Engine 主动发起后台启动失败：%s", _safe_error(exc))
        # Provider credentials are intentionally never inspected here. They
        # belong exclusively to AstrBot's native provider manager.

    def _configure_input_enhancements(self) -> None:
        """Apply the three input-enhancement switches without restarting the host."""

        flow_enabled = self._bool("conversation_flow_enabled", True)
        if flow_enabled and self._bool("optimize_tool_history_context", False):
            self._tool_history_context.install()
        else:
            self._tool_history_context.terminate()

        self._group_wake_suppression.configure(
            disable_at_bot_wake=self._bool("disable_group_at_bot_wake", False),
            disable_at_bot_wake_scope_blacklist=self._bool(
                "disable_group_at_bot_wake_scope_blacklist", True
            ),
            disable_at_bot_wake_group_ids=self._setting(
                "disable_group_at_bot_wake_group_ids", []
            ),
            disable_reply_to_bot_wake=self._bool(
                "disable_group_reply_to_bot_wake", False
            ),
            disable_reply_to_bot_wake_scope_blacklist=self._bool(
                "disable_group_reply_to_bot_wake_scope_blacklist", True
            ),
            disable_reply_to_bot_wake_group_ids=self._setting(
                "disable_group_reply_to_bot_wake_group_ids", []
            ),
        )
        if flow_enabled and (
            self._bool("disable_group_at_bot_wake", False)
            or self._bool("disable_group_reply_to_bot_wake", False)
        ):
            self._group_wake_suppression.install()
        else:
            self._group_wake_suppression.terminate()

    async def _configure_output_runtime(self) -> None:
        """Enable the vendored output pipeline only when explicitly selected."""

        enabled = self._bool("conversation_flow_enabled", True) and self._bool(
            "output_enhancement_enabled", False
        )
        if not enabled:
            if self._output_runtime is not None:
                await self._output_runtime.close()
                self._output_runtime = None
            return
        if self._output_runtime is None:
            from .decision.outputpro_runtime import OutputPipelineRuntime

            self._output_runtime = OutputPipelineRuntime(
                self.context,
                self._settings_path.parent / PLUGIN_NAME,
            )
        config = self._setting("output_pipeline", {})
        if not isinstance(config, Mapping):
            config = {}
        await self._output_runtime.configure(dict(config))

    def _log_provider_retry(
        self, retry_number: int, max_retries: int, reason: str, backoff_seconds: float
    ) -> None:
        logger.warning(
            "Decision Engine Jev 请求失败，将进行第 %d/%d 次重试；间隔 %.2f 秒：%s",
            retry_number,
            max_retries,
            backoff_seconds,
            _safe_error(RuntimeError(reason)),
        )

    async def terminate(self) -> None:
        usage_meter.METER.save()
        self._tool_history_context.terminate()
        self._group_wake_suppression.terminate()
        # Cancel WebUI jobs first: they hold the runtime and would otherwise keep
        # mutating a runtime that is about to be torn down.
        pending = [task for task in self._life_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._life_tasks.clear()
        self._life_jobs.clear()
        self._life_media = None
        if self._multimodal is not None:
            try:
                await self._multimodal.terminate()
            except Exception as exc:
                logger.warning("Decision Engine 多模态关闭失败：%s", _safe_error(exc))
            self._multimodal = None
        # Under the lock: a runtime still being built is torn down once ready
        # instead of being assigned after shutdown.
        async with self._life_lock:
            await self._stop_life_runtime_locked()
        if self._personal_network is not None:
            try:
                await self._personal_network.terminate()
            except Exception as exc:
                logger.warning("Decision Engine 人际网络关闭失败：%s", _safe_error(exc))
            self._personal_network = None
        if self.provider is not None:
            try:
                await self.provider.close()
            except Exception as exc:
                logger.warning("Decision Engine Provider 关闭失败：%s", _safe_error(exc))
        self.provider = None
        self._ready = False
        if self._output_runtime is not None:
            try:
                await self._output_runtime.close()
            except Exception as exc:
                logger.warning("Decision Engine 输出增强关闭失败：%s", _safe_error(exc))
            self._output_runtime = None

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self._setting(key, default))
        except (TypeError, ValueError):
            return default

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self._setting(key, default))
        except (TypeError, ValueError):
            return default

    def _bool(self, key: str, default: bool = False) -> bool:
        value = self._setting(key, default)
        return value if isinstance(value, bool) else bool(value)

    def _policy(self) -> str:
        return str(self._setting("jev_pre_prompt", "")).strip() or DEFAULT_POLICY

    def _main_llm_post_prompt(self) -> str:
        configured = self._setting("main_llm_post_prompt")
        if configured is None:
            return DEFAULT_MAIN_LLM_POST_PROMPT
        # An explicitly empty value is a supported way to disable this
        # optional advisory section without disabling Tool filtering.
        return str(configured)

    @filter.on_llm_request(priority=1000)
    async def filter_tools_before_llm(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Run one fan-out decision request before AstrBot's main LLM call."""
        return await self._filter_tools_before_llm_impl(event, req)

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE,
        priority=1000,
    )
    async def proactive_reply(self, event: AstrMessageEvent):
        """Use Jev to decide whether AstrBot's normal Agent should interject."""
        async for item in self._proactive_reply_impl(event):
            yield item

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE,
        priority=1000,
    )
    async def output_enhancement_message(self, event: AstrMessageEvent) -> None:
        """Collect the small amount of input state required by OutputPro."""

        runtime = self._output_runtime
        if (
            runtime is None
            or not self._bool("conversation_flow_enabled", True)
            or not self._scope_allows(
                event,
                "output_scope",
                "output_scope_blacklist",
                default_blacklist=True,
            )
        ):
            return
        await runtime.prepare_message(event)

    @(
        _ON_DECORATING_RESULT(priority=15)
        if callable(_ON_DECORATING_RESULT)
        else (lambda target: target)
    )
    async def output_enhancement(self, event: AstrMessageEvent) -> None:
        """Run the optional OutputPro-compatible last-mile pipeline.

        The scope is intentionally independent from active conversation.  A
        user can therefore enable output formatting for a set of groups while
        keeping Jev proactive decisions disabled there.
        """

        runtime = self._output_runtime
        if (
            runtime is None
            or not self._bool("conversation_flow_enabled", True)
            or not self._scope_allows(
                event,
                "output_scope",
                "output_scope_blacklist",
                default_blacklist=True,
            )
        ):
            return
        await runtime.run(event)

    @filter.on_llm_response()
    async def remember_bot_response(self, event: AstrMessageEvent, response: Any) -> None:
        if event.get_extra("decision_life_proactive", False):
            return
        session = str(event.unified_msg_origin)
        # Settle our own proactive reservation first, independent of the
        # switches below: only the reply to that request may do it, never an
        # ordinary user-addressed reply in the same session.
        reply_token = event.get_extra(PROACTIVE_REPLY_EXTRA, None)
        if reply_token is not None:
            self.proactive.mark_reply_success(session, reply_token)
        if not self._bool("proactive_reply_enabled", False):
            return
        if event.get_message_type() not in {
            MessageType.GROUP_MESSAGE,
            MessageType.FRIEND_MESSAGE,
        }:
            return
        if not self._proactive_whitelist_allows(event):
            return
        text = str(getattr(response, "completion_text", "") or "").strip()
        if text:
            replied_to_id = str(event.get_sender_id() or "")
            replied_to_name = str(event.get_sender_name() or replied_to_id or "用户")
            self.proactive.add(
                session,
                ProactiveRecord(
                    "bot",
                    str(event.get_self_id() or "bot"),
                    text,
                    True,
                    talking_to=replied_to_id or GROUP_TARGET_ID,
                    talking_to_name=replied_to_name or GROUP_TARGET_NAME,
                ),
            )

    @filter.command("decision")
    async def decision_command(self, event: AstrMessageEvent):
        """管理员诊断：/decision status、/decision test、/decision tools。"""

        if not self._is_admin(event):
            event.stop_event()
            return
        text = str(event.get_message_str() or "").strip().lstrip("/")
        parts = text.split(maxsplit=1)
        action = parts[1].strip().lower() if len(parts) > 1 else "status"
        if action == "status":
            tools = self._manager_tool_objects()
            handoffs = sum(isinstance(item, HandoffTool) for item in tools)
            output = (
                f"Tool/MCP/Skill/SubAgent decision: {'enabled' if self._bool('tools_subagents_decision_enabled', True) else 'disabled'}\n"
                f"proactive_reply={self._bool('proactive_reply_enabled', False)}\n"
                f"jev_provider_id={self._setting('decision_provider_id', '') or 'AstrBot default'}\n"
                f"tool_filter={self._bool('tool_filter_enabled', True)} mcp_filter={self._bool('mcp_filter_enabled', True)} skill_filter={self._bool('skill_filter_enabled', True)} subagent_filter={self._bool('subagent_filter_enabled', False)} "
                f"tool_decision={self._bool('tool_decision_enabled', True)} mcp_decision={self._bool('mcp_decision_enabled', True)} skill_decision={self._bool('skill_decision_enabled', True)} subagent_decision={self._bool('subagent_decision_enabled', True)} "
                f"threshold={self._float('tool_noul_threshold', DEFAULT_TOOL_NOUL_THRESHOLD):.3f}\n"
                f"registered_tools={len(tools)} handoffs={handoffs} always_keep={len(self._setting('always_keep_tools', []) or [])}\n"
                f"calls={self._call_count} failures={self._failure_count} latency_ms={self._last_call_latency_ms or 0:.1f}"
            )
            yield event.plain_result(output)
        elif action == "test":
            try:
                result = await self._evaluate(
                    state="The service is available.",
                    questions={
                        "service": {
                            "type": "noul",
                            "instructions": "Is the service available?",
                        }
                    },
                )
                yield event.plain_result(
                    f"AstrBot JEV OK: {compact_answer(result.answers['service'])}"
                )
            except Exception as exc:
                yield event.plain_result(f"AstrBot JEV failed: {_safe_error(exc)}")
        elif action == "tools":
            tools = self._manager_tool_objects()
            names = [str(getattr(item, "name", "")) for item in tools]
            shown = names[:80]
            suffix = f"\n... 共 {len(names)} 个" if len(names) > len(shown) else ""
            yield event.plain_result("\n".join(shown) + suffix)
        else:
            yield event.plain_result("用法：/decision status | /decision test | /decision tools")
        event.stop_event()

    @filter.on_llm_request(priority=1300)
    async def life_control_tool_state(self, event, req):
        runtime = self._life_runtime
        tools = getattr(req, "func_tool", None)
        if runtime is None or tools is None:
            return
        settings = runtime.config
        hide = set()
        for name, group, flag in (
            ("adjust_contact_frequency", "adaptive_settings", "preference_tool_enable"),
            ("future_contact", "temporal_settings", "task_tool_enable"),
            ("save_life_thought", "thought_settings", "share_tool_enable"),
        ):
            if not settings[group]["enable"] or not settings[group][flag]:
                hide.add(name)
        if settings["temporal_settings"]["enable"]:
            hide.add("future_task")
        if event.get_extra("decision_life_proactive", False):
            # Only Decision wrappers can enforce the one-kind/count budget.
            hide.update({"generate_image", "generate_video", "generate_audio"})
            kind = event.get_extra("decision_life_media_kind", "")
            for name, media_kind in (
                ("take_photo", "photo"),
                ("take_video", "video"),
                ("send_voice", "audio"),
            ):
                if kind != media_kind:
                    hide.add(name)
        if event.get_extra("decision_temporal_guard", False):
            hide.add("send_message_to_user")
        tools.tools = [tool for tool in tools.tools if getattr(tool, "name", "") not in hide]

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE, priority=1100
    )
    async def life_observe_message(self, event: AstrMessageEvent):
        # One reference per call: a settings save may swap the runtime meanwhile.
        runtime = self._life_runtime
        if event.get_extra("decision_life_proactive", False):
            return
        if runtime:
            if self._command_handler_matched(event):
                runtime._mark_conversation_activity(str(event.unified_msg_origin))
                return
            await runtime.temporal.prepare_event(event)
            if (
                runtime.config["continuity_settings"]["enable"]
                or runtime.config["temporal_settings"]["enable"]
            ):
                persona = await runtime.personas.resolve(str(event.unified_msg_origin))
                runtime.delivery.wrap_event(event, persona.id)
            await runtime.record_incoming(event)

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE, priority=1001
    )
    async def multimodal_observe_message(self, event: AstrMessageEvent):
        """Index media and start background understanding for any message.

        This deliberately runs for messages that are not addressed to the bot: the
        listener only marks ``is_wake`` for plugin handlers and never sets
        ``is_at_or_wake_command``, so no reply is triggered here.
        """

        if event.get_extra("decision_life_proactive", False):
            return
        settings = self._multimodal_settings()
        if not settings.get("mm_enabled"):
            return
        try:
            runtime = self._get_multimodal()
            allowed = self._mm_scope_allows(event, settings)
            refs = await runtime.observe(event, auto_allowed=allowed)
            if refs:
                event.set_extra("decision_mm_refs", refs)
        except Exception as exc:
            logger.debug("Decision Engine 多模态观察跳过：%s", _safe_error(exc))

    @filter.on_llm_request(priority=900)
    async def multimodal_prepare_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """Make GIFs provider-safe and inject the media digest."""

        if self._bool("conversation_flow_enabled", True) and self._bool(
            "optimize_tool_history_context", False
        ):
            self._tool_history_context.sanitize_request(req)

        settings = self._multimodal_settings()
        if not settings.get("mm_enabled"):
            return
        runtime = self._get_multimodal()
        try:
            await runtime.sanitize_request_images(req)
        except Exception as exc:
            logger.warning("Decision Engine 动图兼容处理失败：%s", _safe_error(exc))
        refs = event.get_extra("decision_mm_refs") or []
        try:
            # Always call in: a later text-only question must still recall the
            # media understood earlier in this same conversation.
            await runtime.inject(event, req, refs)
        except Exception as exc:
            logger.warning("Decision Engine 注入多模态结果失败：%s", _safe_error(exc))

    @filter.on_llm_request(priority=1100)
    async def life_inject_state(self, event: AstrMessageEvent, req: ProviderRequest):
        # One reference per call: a settings save may swap the runtime meanwhile.
        runtime = self._life_runtime
        if runtime and not self._command_handler_matched(event):
            if (
                runtime.enabled
                and not event.get_extra("decision_life_proactive", False)
                and not event.get_extra("decision_life_counted", False)
            ):
                await runtime.mark_agent_active(event, None)
                event.set_extra("decision_life_counted", True)
            await runtime.inject_cached_state(event, req)

    @getattr(filter, "on_waiting_llm_request", lambda **kw: lambda fn: fn)(priority=1100)
    async def life_waiting_request(self, event: AstrMessageEvent):
        # One reference per call: a settings save may swap the runtime meanwhile.
        runtime = self._life_runtime
        if not runtime or not runtime.enabled or self._command_handler_matched(event):
            return
        if not runtime.config["schedule_settings"]["daily_schedule_generation_enable"]:
            return
        # Explicit summons keep the immediate native reply contract.
        if event.get_extra("decision_life_proactive", False) or self._direct_reply_requested(event):
            return
        at_bot, reply_bot, _, _ = self._message_address_flags(event)
        if at_bot or reply_bot:
            return
        await runtime.queue_reply_delay(event)

    @getattr(filter, "on_agent_begin", lambda **kw: lambda fn: fn)()
    async def life_agent_begin(self, event, run_context):
        # One reference per call: a settings save may swap the runtime meanwhile.
        runtime = self._life_runtime
        if (
            runtime
            and not event.get_extra("decision_life_proactive", False)
            and not event.get_extra("decision_life_counted", False)
        ):
            await runtime.mark_agent_active(event, run_context)
            event.set_extra("decision_life_counted", True)

    @getattr(filter, "on_agent_done", lambda **kw: lambda fn: fn)()
    async def life_agent_done(self, event, run_context, response):
        # One reference per call: a settings save may swap the runtime meanwhile.
        runtime = self._life_runtime
        if runtime and event.get_extra("decision_life_counted", False):
            event.set_extra("decision_life_counted", False)
            await runtime.defer_agent_release(event, run_context, response)

    @getattr(filter, "after_message_sent", lambda **kw: lambda fn: fn)()
    async def life_after_send(self, event):
        # One reference per call: a settings save may swap the runtime meanwhile.
        runtime = self._life_runtime
        if runtime:
            if runtime.config["adaptive_settings"]["enable"] and getattr(
                event, "__is_llm_reply", False
            ):
                persona = await runtime.personas.resolve(str(event.unified_msg_origin))
                runtime.contacts.normal_reply(event, persona.id, runtime._now())
        if runtime and not event.get_extra("decision_life_proactive", False):
            await runtime.mark_active_conversation(event)

    @filter.on_llm_response()
    async def life_track_response(self, event, response):
        # One reference per call: a settings save may swap the runtime meanwhile.
        runtime = self._life_runtime
        if runtime:
            await runtime.temporal.guard_response(event, response)
            await runtime.track_delayed_llm_response(event, response)
            if not hasattr(filter, "on_agent_done"):
                await self.life_agent_done(event, None, response)

    @getattr(filter, "on_astrbot_loaded", lambda **kw: lambda fn: fn)()
    async def network_loaded(self) -> None:
        if self._network_enabled():
            await self._network().on_astrbot_loaded()

    @filter.on_llm_request(priority=1200)
    async def network_inject_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """Gate the network tools and inject relationship context.

        This runs *before* the Tool decision hook (priority 1000) so a disabled
        network tool is removed from the candidate set instead of being judged
        and possibly re-added. When enabled, the tools stay in the request and
        are judged like any other plugin tool.
        """

        settings = self._network_settings()
        enabled = bool(settings.get("enable", False)) and not self._command_handler_matched(event)
        self._hide_network_tools(
            req,
            update=not (enabled and settings.get("enable_llm_update_tool", True)),
            query=not (enabled and settings.get("enable_llm_query_tool", True)),
        )
        if enabled:
            await self._network().inject_context(event, req)

    @filter.command("关系查询", alias={"查询关系", "人际关系"})
    async def network_query_command(self, event: AstrMessageEvent, query: str = ""):
        if not self._network_enabled():
            yield event.plain_result("人际网络已关闭，请在插件 WebUI 的主动发起页面开启。")
        else:
            yield event.plain_result(await self._network().query_command_text(event, query))
        event.stop_event()

    @filter.command("生活")
    async def life_command(self, event: AstrMessageEvent, action: str = "状态"):
        if action not in {"状态", "日程", "穿搭", "立即"}:
            yield event.plain_result(
                "用法：/生活 状态｜日程｜穿搭｜立即（管理员）；其他设置请打开插件 WebUI。"
            )
            return
        if action == "立即" and getattr(event, "role", "member") != "admin":
            yield event.plain_result("只有管理员可以请求立即发起。")
            return
        runtime = await self._get_life_runtime()
        if action == "状态":
            yield event.plain_result(
                json.dumps(runtime.status(str(event.unified_msg_origin)), ensure_ascii=False)
            )
        elif action == "日程":
            async for result in runtime.show_schedule(event):
                yield result
        elif action == "穿搭":
            async for result in runtime.show_outfit(event):
                yield result
        else:
            persona, plan = await runtime._ensure_plan_for_umo(event.unified_msg_origin)
            sent, reason = await runtime._attempt_unsolicited(
                event.unified_msg_origin,
                persona,
                plan,
                "管理员请求此刻自然地发起一次聊天",
                "manual",
            )
            yield event.plain_result("已发起。" if sent else f"本次未发起：{reason}")
        event.stop_event()

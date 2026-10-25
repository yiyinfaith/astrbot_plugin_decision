"""Configuration for conversation/persona contact control and temporal facts."""


def _field(kind, description, default, hint="", **extra):
    return {"type": kind, "description": description, "default": default, "hint": hint, **extra}


def control_schema():
    return {
        "continuity_settings": {
            "type": "object", "description": "对话连续性与生活状态",
            "items": {
                "enable": _field("bool", "启用话题跟进与连续状态", True, "按会话＋人格记录真实话题；生活状态按人格共享。"),
                "quality_enabled": _field("bool", "根据互动质量调节联系", True, "结合真实内容判断热情、忙碌、收尾；无关群聊不参与，不推测已读状态。"),
                "topics_enabled": _field("bool", "自然跟进未结束的话题", True, "话题到期只是普通主动候选，仍遵守限频、名单和免打扰，不替代明确提醒。"),
                "state_enabled": _field("bool", "保持生活状态前后连贯", True),
                "max_semantic_calls_day": _field("int", "每会话人格每日语义分析上限", 12, "包括互动分析与发送后分享确认；失败也占预算，不自动重复提交。"),
                "analysis_interval_seconds": _field("int", "互动分析最小间隔（秒）", 120, "期间的新消息合并分析，不延迟正常聊天回复。"),
                "topic_retention_days": _field("int", "未结束话题保留天数", 14),
                "max_open_topics": _field("int", "每会话人格未结束话题上限", 20),
                "temporary_quiet_minutes": _field("int", "临时忙碌最长避让（分）", 240, "普通群成员的忙碌只影响其本人话题，不修改群频率。"),
            },
        },
        "adaptive_settings": {
            "type": "object", "description": "📈 自动联系频率与权限",
            "items": {
                "enable": _field("bool", "启用自动频率调节", False, "按会话＋人格隔离；开启后以目标间隔及每日硬上限取代随机基础预算。"),
                "friend_min_minutes": _field("int", "私聊最短主动间隔（分）", 120),
                "group_min_minutes": _field("int", "群聊最短主动间隔（分）", 60),
                "max_interval_minutes": _field("int", "最长主动间隔／低频保底（分）", 2880, "未回应只逐步降频；在允许联系的时段保留低频联系。明确暂停、禁用和名单限制优先。"),
                "response_window_minutes": _field("int", "主动消息回应观察期（分）", 120),
                "adjustment_cooldown_minutes": _field("int", "积极互动自动调整间隔（分）", 360),
                "positive_factor": _field("float", "积极互动间隔倍率", 0.85),
                "negative_factor": _field("float", "未回应间隔倍率", 1.6),
                "feedback_factor": _field("float", "明确增减频率倍率", 1.5),
                "group_adjust_blacklist": _field("bool", "频率强制调整使用黑名单", False, "关闭＝白名单：Bot 管理员、群主、群管理员及名单 QQ 可调整；开启＝黑名单：名单 QQ 禁止，其余成员可调整。身份仅从平台消息读取。"),
                "group_adjust_users": _field("list", "可强制调整频率的 QQ 名单", [], "每行一个具体 QQ 号；默认白名单为空，仅内置管理角色有权调整。"),
                "preference_tool_enable": _field("bool", "启用联系偏好工具", True),
            },
        },
        "temporal_settings": {
            "type": "object", "description": "🕰️ 时间承诺与未来联系",
            "items": {
                "enable": _field("bool", "启用时间承诺与未来任务", False, "时间事实按人格／会话保存；关闭主动发送后仍为正常回复注入准确进度。"),
                "task_tool_enable": _field("bool", "启用未来任务工具", True),
                "scheduled_daily_limit": _field("int", "每日明确约定提醒上限", 3, "与普通主动额度分开并提前预留；Bot 自发任务仍占普通主动上限。"),
                "max_pending_tasks": _field("int", "每会话人格待办上限", 50),
                "max_future_days": _field("int", "任务最远天数", 366),
                "prepare_seconds": _field("int", "提前准备消息（秒）", 120),
                "max_lateness_minutes": _field("int", "过期任务允许补发（分）", 30),
                "claim_timeout_seconds": _field("int", "未发送任务处理租约（秒）", 600),
                "request_timeout_seconds": _field("int", "任务／一致性模型超时（秒）", 90),
                "guard_responses": _field("bool", "回复时间一致性校验", True),
                "use_native_scheduler": _field("bool", "复用 AstrBot 未来计划唤醒", True, "原生调度负责到点唤醒，Decision 负责状态和发送规则；原生接口不可用时由插件自己的持久化任务恢复。"),
            },
        },
        "thought_settings": {
            "type": "object", "description": "💡 日程思考与择机分享",
            "items": {
                "enable": _field("bool", "启用活动与思考节点", False),
                "max_generations_per_persona_day": _field("int", "每人格每日想法生成上限", 3),
                "min_generation_minutes": _field("int", "想法生成最小间隔（分）", 120),
                "ttl_minutes": _field("int", "想法有效期（分）", 360),
                "max_pending_per_persona": _field("int", "每人格待分享想法上限", 10),
                "share_tool_enable": _field("bool", "启用想法暂存工具", True),
            },
        },
    }


def validate_control(settings):
    continuity = settings["continuity_settings"]
    for key, low, high in (("max_semantic_calls_day", 0, 200), ("analysis_interval_seconds", 0, 3600),
                           ("topic_retention_days", 1, 90), ("max_open_topics", 1, 100),
                           ("temporary_quiet_minutes", 1, 1440)):
        if not low <= continuity[key] <= high:
            raise ValueError(f"{key} 须在 {low} 到 {high} 之间")
    adaptive = settings["adaptive_settings"]
    if not 1 <= adaptive["friend_min_minutes"] <= adaptive["max_interval_minutes"] <= 43200:
        raise ValueError("私聊最短间隔须为 1 分钟以上且不大于最长间隔；最长不超过 30 天")
    if not 1 <= adaptive["group_min_minutes"] <= adaptive["max_interval_minutes"]:
        raise ValueError("群聊最短间隔须为 1 分钟以上且不大于最长间隔")
    for key in ("response_window_minutes", "adjustment_cooldown_minutes"):
        if not 1 <= adaptive[key] <= 43200:
            raise ValueError("频率观察／调整间隔须在 1 到 43200 分钟之间")
    if not 0.1 <= adaptive["positive_factor"] < 1:
        raise ValueError("积极互动倍率须在 0.1 到 1 之间（不含 1）")
    if not 1 < adaptive["negative_factor"] <= 5 or not 1 < adaptive["feedback_factor"] <= 5:
        raise ValueError("降频／明确反馈倍率须大于 1 且不超过 5")
    if any(not ident.isascii() or not ident.isdecimal() or len(ident) > 20 for ident in adaptive["group_adjust_users"]):
        raise ValueError("频率权限名单须填写具体 QQ 号，每行一个，最多 20 位纯数字")
    temporal = settings["temporal_settings"]
    for key, low, high in (
        ("scheduled_daily_limit", 1, 100), ("max_pending_tasks", 1, 500),
        ("max_future_days", 1, 3660), ("prepare_seconds", 0, 3600),
        ("max_lateness_minutes", 0, 1440), ("claim_timeout_seconds", 60, 3600),
        ("request_timeout_seconds", 10, 600),
    ):
        if not low <= temporal[key] <= high:
            raise ValueError(f"{key} 须在 {low} 到 {high} 之间")
    if temporal["claim_timeout_seconds"] < temporal["request_timeout_seconds"] + 30:
        raise ValueError("任务租约须比模型超时至少长 30 秒")
    thoughts = settings["thought_settings"]
    for key, low, high in (("max_generations_per_persona_day", 0, 50), ("min_generation_minutes", 1, 1440), ("ttl_minutes", 1, 10080), ("max_pending_per_persona", 1, 100)):
        if not low <= thoughts[key] <= high:
            raise ValueError(f"{key} 须在 {low} 到 {high} 之间")

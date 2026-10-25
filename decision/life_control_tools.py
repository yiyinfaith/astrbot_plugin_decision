"""Native tool schemas for contact controls (registered under Decision's owner)."""


def control_tools(owner):
    return [
        ("adjust_contact_frequency", "查询或持久化当前会话＋人格的主动联系偏好。用户说太吵/太少时调用；群聊强制调整须有平台管理角色或 QQ 名单权限。不得把普通聊天回复当作主动联系。", owner._adjust_contact_frequency,
         {"action": {"type": "string", "enum": ["status", "more", "less", "set", "pause", "resume"]},
          "interval_minutes": {"type": "number", "description": "set 时的目标间隔（分），须在 WebUI 上下限内"},
          "pause_until": {"type": "string", "description": "可选，暂停至带时区的 ISO 时间；省略表示直到用户恢复"}}, ["action"]),
        ("future_contact", "创建/编辑/取消/查询当前会话人格的未来联系或时间承诺。用户约定 origin=user，Bot 自发计划 origin=bot。说几小时后开始/完成前必须先保存，成功送出承诺才启动 Bot 相对计时；普通 native future_task 不可绕过本工具。", owner._future_contact,
         {"action": {"type": "string", "enum": ["create", "edit", "cancel", "list", "complete"]},
          "task_id": {"type": "string"}, "run_at": {"type": "string", "description": "带时区 ISO 时间，与 delay_seconds 二选一"},
          "delay_seconds": {"type": "number", "description": "从本条真实来信起算的延迟；Bot 相对承诺从实际发出承诺起算"},
          "note": {"type": "string", "description": "准确的活动/提醒内容，不超过2000字"},
          "origin": {"type": "string", "enum": ["user", "bot"]},
          "kind": {"type": "string", "enum": ["reminder", "start", "virtual_finish", "external_finish"], "description": "start 只代表到时开始；external_finish 未有真实确认不能宣称已完成"},
          "notify": {"type": "boolean", "description": "到点是否主动发送；仅追踪进度可设 false"}}, ["action"]),
        ("save_life_thought", "暂存当前真实日程活动产生的可分享想法。只保存简短内容及活动来源，不保存内部推理；分享仍受用户偏好、每日额度和免打扰约束。", owner._save_life_thought,
         {"text": {"type": "string", "description": "可分享的小想法，不超过1500字"}}, ["text"]),
    ]

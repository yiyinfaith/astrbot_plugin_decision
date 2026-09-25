# astrbot_plugin_decision

AstrBot 智能决策引擎。插件在 AstrBot 主 LLM 请求前调用一个 Decision Model，先做低延迟、结构化的判断，再把结果用于普通 Tool 预筛选、SubAgent 推荐和可选的群聊主动回复。它不生成最终回答，也不直接执行 Tool 或 handoff；主 LLM 仍然负责参数、调用顺序、是否委派和最终回复。

当前默认 Provider 是 TypeSafe 兼容的 SystemOne API（模型 `jev-latest`），业务层只依赖 `DecisionProvider` 接口，后续可以替换 Provider。配置页不会预填第三方 Base URL；请由部署者按自己的服务填写。

## 架构

```text
用户消息
  └─ AstrBot ProviderRequest
       └─ DecisionPlugin.on_llm_request
            └─ DecisionProvider.evaluate
                 └─ SystemOneProvider (aiohttp)
            ├─ 多个 Noul：普通 Tool 是否保留
            ├─ 一个 Choice：优先考虑哪个 SubAgent
            └─ 修改本轮 ToolSet + 临时 routing hint
       └─ AstrBot 主 Agent / Tool Loop / SubAgent / Provider
```

Jev 的 Decision Policy 是插件配置中的独立文本，不会把 AstrBot 主 LLM 的完整 `system_prompt` 发送给 SystemOne。状态只包含当前请求、截断后的最近上下文、Tool 名称和简短描述、SubAgent 概览；不会发送 Tool 参数 JSON Schema。

## Tool Filter

普通 Tool 一工具一个 Noul，尽量放在一个 SystemOne 请求中。Tool 只发送 `name + description`，不使用 Choice 选择多个 Tool。达到 `tool_noul_threshold` 的 Tool 才保留，主模型不会看到内部概率。若服务端返回当前已知的“混合题型 usage”400，Provider 会按 Noul、Choice、Score 题型拆成最多三路；同一题型（包括 202 个 Noul）仍保持单请求。

Always Keep 页面位于插件详情页的 `settings` Plugin Page。页面通过 AstrBot 官方 `window.AstrBotPluginPage` bridge 调用插件 Web API，自动读取已注册 Tool、搜索、勾选和保存。Always Keep 只在本轮原本允许的 `req.func_tool` 中生效，因此不会突破 Persona 的 Tool 权限。SubAgent `HandoffTool` 按类型识别并始终保留，`decision_evaluate` 也始终保留。

## SubAgent 推荐

当前请求的 handoff tools 使用一个 Choice question，候选项来自本轮已有的 handoff tools，并额外提供 `none`。推荐结果只通过 `TextPart(...).mark_as_temp()` 写入 `req.extra_user_content_parts`，内容是 advisory hint，不追加到 `req.system_prompt`，也不会执行 handoff。

## `decision_evaluate`

插件使用 AstrBot 4.28.1 的 `Context.add_llm_tools()` 注册一个主 LLM 可主动调用的工具。它接受：

- `decision_type`: `noul`、`choice` 或 `score`
- `state`、`instructions`
- `choice_options`: `{key, description}` 数组
- `score_levels`: 有序字符串数组

它一次只评估一个问题，并返回简洁的结构化 JSON。它不会执行外部 Tool。

## 主动回复

主动回复默认关闭，只处理 AstrBot 实际收到的群聊 ambient message。插件维护每个会话的少量消息队列，用一次 SystemOne 请求同时询问 `is_addressing_bot` 和 `should_interject`，通过阈值、冷却和窗口计数后，使用官方 `event.request_llm()` 进入 AstrBot 原生 Agent 流程。

会跳过 @/wake、指令、回复 Bot 和已经准备正常响应的消息。若 AstrBot 内置 `active_reply` 已开启，插件会停用自己的主动回复并记录 warning；不会修改全局配置。某些平台只有 @Bot 消息才会推送给 Bot，插件无法判断未收到的群聊消息。

主动回复在 SystemOne 失败时保持 fail-closed；普通 Tool Filter 在失败时保留原始 Tool，保证主聊天能力不被决策服务故障拖垮。

## 安装和配置

AstrBot 要求 `>=4.28.1,<5`。将运行时文件放入 `data/plugins/astrbot_plugin_decision/`，在 WebUI 配置：

1. `base_url`: 必填，配置页默认留空，不预置第三方地址
2. `systemone_path`: 默认 `/v1/systemone`
3. `api_key`: SystemOne Bearer Key，配置 schema 使用 `secret: true`（仅遮罩，不是加密）
4. `model`: 默认 `jev-latest`
5. 开启 `tool_filter_enabled`，按需要调整 Noul 阈值
6. 如需主动回复，先关闭 AstrBot 自带 `provider_ltm_settings.active_reply`，再开启插件 `proactive_reply_enabled`

其余配置包括超时、重试、Decision Policy、Tool 描述截断、历史消息限制、Always Keep、SubAgent 推荐、主动回复冷却/窗口和调试日志，均定义在 `_conf_schema.json`。

管理员诊断命令：

```text
/decision status
/decision test
/decision tools
```

命令不会显示 API Key 或 Authorization。

## 安全和故障处理

API Key 不写入日志、README、测试或 Git。错误信息会去掉请求头并截断。SystemOne Provider 使用可复用的异步 `aiohttp.ClientSession`，处理 timeout、DNS/连接错误、401/403、429、5xx、非 JSON 和协议字段错误；大问题数请求只有在服务端明确拒绝 payload 时才回退为小块并发请求。

本插件不依赖 TypeSafe Skill，不创建 `SKILL.md` 或 `skills/`，也不修改 AstrBot Core。

## 开发检查

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

`tests/`、Git 元数据、CI 文件和开发脚本只留在源码仓库，不应复制到服务器插件运行目录。

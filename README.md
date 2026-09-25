# astrbot_plugin_decision

AstrBot 智能决策引擎。插件在 AstrBot 主 LLM 请求前调用一个 Decision Model，先做低延迟、结构化的判断，再把结果用于普通 Tool 的逐个过滤与推荐、SubAgent 的逐个推荐和可选的群聊主动回复。它不生成最终回答，也不直接执行 Tool 或 handoff；主 LLM 仍然负责参数、调用顺序、是否委派和最终回复。

当前默认 Provider 是 TypeSafe 兼容的 SystemOne API（模型 `jev-latest`），业务层只依赖 `DecisionProvider` 接口，后续可以替换 Provider。配置页不会预填第三方 Base URL；请由部署者按自己的服务填写。

## 架构

```text
用户消息
  └─ AstrBot ProviderRequest
       └─ DecisionPlugin.on_llm_request
            └─ DecisionProvider.evaluate
                 └─ SystemOneProvider (aiohttp)
            ├─ 多个 Noul：逐个判断普通 Tool 是否保留并推荐
            ├─ 多个 Noul：逐个判断 SubAgent 是否推荐（不从 ToolSet 过滤）
            └─ 修改本轮 ToolSet + 追加主 LLM system prompt routing hint
       └─ AstrBot 主 Agent / Tool Loop / SubAgent / Provider
```

插件配置中有两套可编辑提示词：`jev_pre_prompt` 发送给 Jev/Decision Model，`main_llm_post_prompt` 在 Jev 判断完成后追加到主 LLM 的 system prompt。后置提示词支持 `{tools}` 和 `{subagents}` 占位符；渲染只替换这两个占位符，不会破坏用户提示词中的其它大括号。Jev 不会收到 AstrBot 主 LLM 的完整 `system_prompt`。状态只包含当前请求、截断后的最近上下文、Tool 名称和简短描述、SubAgent 概览；不会发送 Tool 参数 JSON Schema。

## Tool Filter

普通 Tool 一工具一个 Noul，尽量放在一个 SystemOne 请求中。Tool 只发送 `name + description`，不使用 Choice 选择多个 Tool。达到 `tool_noul_threshold` 的 Tool 会保留，并进入主 LLM system prompt 中的推荐提示；因此模型既能看到工具，也会知道本轮哪些工具值得优先考虑，结果可以是 0 个、1 个或多个。若服务端返回当前已知的“混合题型 usage”400，Provider 会按 Noul、Choice、Score 题型拆成最多三路；同一题型（包括 202 个 Noul）仍保持单请求。

Always Keep 页面位于插件详情页的 `settings` Plugin Page。页面通过 AstrBot 官方 `window.AstrBotPluginPage` bridge 调用插件 Web API，自动读取已注册 Tool、搜索、勾选和保存。每个普通 Tool 有两个独立选项：`始终保留` 和 `推荐主 LLM`。第二项只有在第一项勾选后才可用；因此可以让一个 Tool 始终留在本轮 ToolSet 中，但不主动推荐它。Always Keep 只在本轮原本允许的 `req.func_tool` 中生效，因此不会突破 Persona 的 Tool 权限。SubAgent `HandoffTool` 按类型识别并始终保留，`decision_evaluate` 也始终保留。

## SubAgent 推荐

当前请求的每个 handoff tool 都使用独立的 Noul question。SubAgent 永远不会因为 Noul 结果从本轮 ToolSet 中被过滤；只有通过阈值的候选会进入推荐列表，所以可以推荐 0 个、1 个或多个 SubAgent。推荐不会执行 handoff，而是和普通 Tool 推荐一起渲染到后置提示词中。

普通 Tool 和 SubAgent 的推荐会在同一个可配置提示中分开列出。插件把这段提示追加到本次请求已有的 `req.system_prompt` 末尾，因此 AstrBot 原人格和其它插件已经追加的系统提示词都会保留；插件不会覆盖、清空或重建其它内容。提示明确允许主 LLM 选择零个、一个或多个候选；主 LLM 仍然可以忽略推荐并自行决定调用方式。普通 Tool 的逐个过滤始终启用，不提供关闭开关。

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

AstrBot 要求 `>=4.28.1,<5`。将运行时文件放入 `data/plugins/astrbot_plugin_decision/`，在 WebUI 配置。运行时配置由 AstrBot 保存到 `data/config/astrbot_plugin_decision_config.json`，插件目录只包含代码、Schema、页面和静态资源，更新插件不会覆盖配置：

1. `base_url`: 必填，配置页默认留空，不预置第三方地址
2. `systemone_path`: 默认 `/v1/systemone`
3. `api_key`: SystemOne Bearer Key，配置 schema 使用 `secret: true`（仅遮罩，不是加密）
4. `model`: 默认 `jev-latest`
5. 按需要调整 Noul 阈值；普通 Tool 逐个过滤始终启用
6. 在插件详情页 `settings` 中分别勾选普通 Tool 的 `始终保留` 和 `推荐主 LLM`
7. 如需主动回复，先关闭 AstrBot 自带 `provider_ltm_settings.active_reply`，再开启插件 `proactive_reply_enabled`

其余配置包括超时、重试、两套可编辑提示词、Tool 描述截断、历史消息限制、Always Keep、SubAgent 推荐、主动回复冷却/窗口和调试日志，均定义在 `_conf_schema.json`。插件详情页的 `settings` 页面也可直接编辑这两套提示词。

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

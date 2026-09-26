# astrbot_plugin_decision

AstrBot 智能决策引擎。插件在 AstrBot 主 LLM 请求前调用一个 Decision Model，先做低延迟、结构化的判断，再把结果用于普通 Tool 和 SubAgent 的逐个过滤与推荐，以及可选的群聊主动回复。它不生成最终回答，也不直接执行 Tool 或 handoff；主 LLM 仍然负责参数、调用顺序、是否委派和最终回复。

当前默认 Provider 是 TypeSafe 兼容的 SystemOne API（模型 `jev-latest`），业务层只依赖 `DecisionProvider` 接口，后续可以替换 Provider。配置页不会预填第三方 Base URL；请由部署者按自己的服务填写。

## 架构

```text
用户消息
  └─ AstrBot ProviderRequest
       └─ DecisionPlugin.on_llm_request
            └─ DecisionProvider.evaluate
                 └─ SystemOneProvider (aiohttp)
            ├─ 多个 Noul：逐个判断普通 Tool 是否推荐，并按开关决定是否过滤
            ├─ 多个 Noul：逐个判断 SubAgent 是否推荐，并按开关决定是否过滤
            └─ 修改本轮 ToolSet + 追加主 LLM system prompt routing hint
       └─ AstrBot 主 Agent / Tool Loop / SubAgent / Provider
```

插件配置中有两套可编辑提示词：`jev_pre_prompt` 发送给 Jev/Decision Model，`main_llm_post_prompt` 在 Jev 判断完成后追加到主 LLM 的 system prompt。后置提示词支持 `{tools}` 和 `{subagents}` 占位符；渲染只替换这两个占位符，不会破坏用户提示词中的其它大括号。Jev 不会收到 AstrBot 主 LLM 的完整 `system_prompt`。状态只包含当前请求、截断后的最近上下文、Tool 名称和简短描述、SubAgent 概览；不会发送 Tool 参数 JSON Schema。

## Tools 与 SubAgents 的过滤和推荐

普通 Tool 和每个 SubAgent 都有独立的 Noul 判断，尽量放在一个 SystemOne 请求中。候选只发送 `name + description`，不使用 Choice 选择一个唯一赢家。达到 `tool_noul_threshold` 的候选会进入主 LLM system prompt 中的推荐提示，因此推荐结果可以是 0 个、1 个或多个。两个过滤开关只控制候选是否从本轮 ToolSet 中移除，不会关闭 Jev 判断或推荐：

- `过滤普通 Tools` 默认开启。开启时主 LLM 只能看到 Jev 判断达到阈值的普通 Tool；关闭时所有普通 Tool 都保留，但推荐提示仍只列出 Jev 选中的 Tool。
- `过滤 SubAgents` 默认关闭。关闭时所有 SubAgent 都保留，但推荐提示仍只列出 Jev 选中的 SubAgent；开启时主 LLM 只能看到 Jev 判断达到阈值的 SubAgent。

通用配置页顶部的 `启用 Tools 与 SubAgent 决策（不影响主动对话）` 只控制这条 Tools/SubAgent 决策链。主动对话由单独的 `proactive_reply_enabled` 控制；关闭前者不会关闭主动回复，关闭后者也不会影响 Tools/SubAgent 判断。

若服务端返回当前已知的“混合题型 usage”400，Provider 会按 Noul、Choice、Score 题型拆成最多三路；同一题型（包括大量 Noul）仍保持单请求。

Always Keep 页面位于插件详情页的 `settings` Plugin Page。页面通过 AstrBot 官方 `window.AstrBotPluginPage` bridge 调用插件 Web API，读取并展示当前工具管理器中的全部插件/MCP Tool，同时调用 AstrBot 的 `iter_builtin_tools()` 展示原生内置 Tool。每个工具右侧会显示来源插件；同一来源的工具会排列在一起，`Astrbot内置工具` 默认排在列表底部。列表支持搜索、勾选和保存。每个普通 Tool 有两个独立选项：`始终保留` 和 `推荐给主 LLM`。第二项只有在第一项勾选后才可用；因此可以让一个 Tool 始终留在本轮 ToolSet 中，但不主动推荐它。Always Keep 只在本轮原本允许的 `req.func_tool` 中生效，因此不会突破 Persona 的 Tool 权限。SubAgent 的推荐由 Jev 逐个判断，页面不提供手工推荐勾选；内置 `jev_decide` 和 AstrBot 原生 Tool 首次打开页面时默认勾选始终保留、默认不勾选推荐，也可以像其它 Tool 一样取消勾选并保存。

普通 Tool 和 SubAgent 的推荐会在同一个可配置提示中分开列出。插件把这段提示追加到本次请求已有的 `req.system_prompt` 末尾，因此 AstrBot 原人格和其它插件已经追加的系统提示词都会保留；插件不会覆盖、清空或重建其它内容。提示明确允许主 LLM 选择零个、一个或多个候选；主 LLM 仍然可以忽略推荐并自行决定调用方式。

备用模型由 AstrBot 在同一个 `ProviderRequest` / Agent Runner 请求中切换。插件追加的内容位于该请求的 `system_prompt`，所以主模型失败并切换备用模型时，备用模型会继续收到同一段人格和 Decision Engine 推荐提示；插件不需要也不会为每个模型重复注入。

## `jev_decide`

插件使用 AstrBot 4.28.1 的 `Context.add_llm_tools()` 注册一个主 LLM 可主动调用的工具。它接受：

- `decision_type`: `noul`、`choice` 或 `score`
- `state`、`instructions`
- `choice_options`: `{key, description}` 数组
- `score_levels`: 有序字符串数组

它一次只评估一个问题，并返回简洁的结构化 JSON。它不会执行外部 Tool。

## 主动回复

主动回复默认关闭，而且严格采用白名单：只有 `proactive_whitelist` 中的群聊 ID、私聊用户 ID 或完整 `unified_msg_origin` 会话才会进入主动对话判断；群聊消息只匹配群 ID，私聊消息只匹配发送者 ID，避免把一个私聊用户 ID 意外扩展成所有群聊。白名单为空或会话不匹配时，插件不做主动判断，AstrBot 原生聊天照常处理。它同时支持群聊和私聊。实现借鉴了 AngelHeart 的直接回复前缀、四状态思路、事件缓存和每会话串行处理，也吸收了 AstrBot 市场中高使用量主动回复插件的会话冷却、活跃度、去重和资源上限设计；没有复制它们的独立模型、人格提示词、安抚消息或上下文接管。

每个白名单会话维护一个有上限的消息队列和状态：`不在场`、`被呼唤`、`混脸熟`、`观测中`。复读和密集对话只用于状态提示，真正是否介入仍由 Jev 判断。普通消息用一次 Jev/SystemOne 请求逐项评估五个 Noul 维度：是否指向机器人、是否适合介入、是否能提供相关价值、时机是否合适、回复是否能自然延续对话。加权结果和直接指向/自然介入阈值共同决定是否回复，异常或缺字段时保持不回复。

`direct_reply_prefixes` 默认是 `/` 和 `@`。普通前缀命中时跳过主动对话 Jev；特殊值 `@` 只匹配消息链中真正的 `At` 机器人节点，不会把文本中的 `@用户名` 当作直接回复。命中后仅设置 AstrBot 原生的 wake 标记，由原生 Agent 回复，因此不会改变人格提示词，也不会接管上下文。这个正常的 Agent 请求随后仍会经过 Tools/SubAgent 的独立 Jev 过滤与推荐链，这是有意保留的一次能力决策；直接回复只跳过主动对话判断，不跳过 Tools/SubAgent 判断。`force_reply_when_summoned` 和 `proactive_alias` 可分别控制真正 @/回复机器人的行为和文本昵称识别；文本昵称本身仍交给 Jev 判断。

每个会话有独立的异步锁、判断间隔、回复冷却、窗口上限、失败退避和最大历史；达到会话上限时清理最久未访问的空闲会话。`max_replies_per_window` 设为 0 时可在保留消息观测的同时禁止主动回复。Jev 的最大重试次数和重试间隔是全局设置，Tools、SubAgent 和主动对话共用；默认重试 1 次、间隔 0 秒。主动对话连续失败时，重试间隔还会作为下一次判断的基础退避时间。若 AstrBot 内置 `active_reply` 已开启，插件会停用自己的 ambient 判断并记录 warning；不会修改全局配置。某些平台只有 @Bot 消息才会推送给 Bot，插件无法判断未收到的群聊消息。

主动对话使用 `event.request_llm()` 进入 AstrBot 原生 Agent 流程，故人格、原生上下文、备用模型和其他插件的 system prompt 处理仍由 AstrBot 负责。主动判断在 Jev 失败时保持 fail-closed；普通 Tool Filter 在失败时保留原始 Tool，保证主聊天能力不被决策服务故障拖垮。

## 安装和配置

AstrBot 要求 `>=4.28.1`。将运行时文件放入 `data/plugins/astrbot_plugin_decision/`，在 WebUI 配置。运行时配置由 AstrBot 保存到 `data/config/astrbot_plugin_decision_config.json`，插件目录只包含代码、Schema、页面和静态资源，更新插件不会覆盖配置：

1. `base_url`: 必填，配置页默认留空，不预置第三方地址
2. `systemone_path`: 默认 `/v1/systemone`
3. `api_key`: SystemOne Bearer Key，配置 schema 使用 `secret: true`（仅遮罩，不是加密）
4. `model`: 默认 `jev-latest`
5. 通用配置页中，`[Tools 与 SubAgent] 启用 Tools 与 SubAgent 决策（不影响主动对话）` 控制决策链总开关；在插件详情页 `settings` 中设置 `过滤普通 Tools`（默认开）和 `过滤 SubAgents`（默认关）
6. 在插件详情页 `settings` 中分别勾选普通 Tool 的 `始终保留` 和 `推荐给主 LLM`
7. 如需主动回复，先填写 `proactive_whitelist`（群聊 ID 或私聊用户 ID），关闭 AstrBot 自带 `provider_ltm_settings.active_reply`，再开启插件 `proactive_reply_enabled`

其余配置包括默认 5 秒超时、最大重试 1 次、重试间隔 0 秒、两套可编辑提示词、Tool 描述截断、历史消息限制、Always Keep、主动回复冷却/窗口和调试日志，均定义在 `_conf_schema.json`。通用配置页用 `[全局设置]`、`[Tools 与 SubAgent]`、`[主动对话]` 副标题区分适用范围；两套提示词、两个过滤开关和两个 Always Keep 列表使用隐藏 Schema 字段保存，只在插件详情页的 `settings` 页面编辑。运行时配置由 AstrBot 保存到 `data/config/astrbot_plugin_decision_config.json`。

管理员诊断命令：

```text
/decision status
/decision test
/decision tools
```

命令不会显示 API Key 或 Authorization。

## 安全和故障处理

API Key 不写入日志、README、测试或 Git。错误信息会去掉请求头并截断。SystemOne Provider 使用可复用的异步 `aiohttp.ClientSession`，处理 timeout、DNS/连接错误、401/403、429、5xx、非 JSON 和协议字段错误；大问题数请求只有在服务端明确拒绝 payload 时才回退为小块并发请求，若服务端限制比默认分块大小更小，会继续二分被拒绝的分块。

本插件不依赖 TypeSafe Skill，不创建 `SKILL.md` 或 `skills/`，也不修改 AstrBot Core。

## 开发检查

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

`tests/`、Git 元数据、CI 文件和开发脚本只留在源码仓库，不应复制到服务器插件运行目录。

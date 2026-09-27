# astrbot_plugin_decision

AstrBot 的 Tools、SubAgent 与主动对话决策插件。每次主 LLM 请求前，插件调用配置的 Jev/SystemOne 决策服务，对候选能力逐项判断；它只负责结构化判断、过滤和推荐，最终回答、参数生成、工具执行与上下文管理仍由 AstrBot 主 Agent 完成。

## 页面与配置边界

AstrBot 原生插件配置页只保留 13 项：

- 全局服务：决策服务、服务地址、API 路径、API 密钥、Jev 模型、请求超时、最大重试次数、重试间隔、上下文上限、上下文消息数、上下文字符数。
- 两个总开关：Tools/SubAgent 决策、主动对话。

过滤阈值、Always Keep、两套提示词、主动对话白名单和主动对话的全部节流/状态参数，都在插件详情页的 WebUI 中配置。WebUI 使用 Vue 3 CDN 和 GooseHyperGlassCDN 的 `liquid-glass` 组件，玻璃层只负责环境光与层次；带文字的按钮使用不透明实色背景，保证可读性。

配置由 AstrBot 写入：

`data/config/astrbot_plugin_decision_config.json`

插件目录只保存代码、Schema、页面和静态资源。更新插件不会覆盖配置。

## Tools 与 SubAgents

```text
ProviderRequest
  └─ DecisionPlugin.on_llm_request
       ├─ 收集当前请求可用的普通 Tools 与 SubAgents
       ├─ 为每个候选建立一个 Noul 问题
       ├─ 按全局 Token 上限拆成若干 Jev 请求并合并答案
       ├─ 按阈值得到 0、1 或多个推荐
       ├─ 按两个过滤开关更新本轮 ToolSet
       └─ 把推荐提示追加到现有 system_prompt
            └─ AstrBot 主 Agent / Tool Loop / SubAgent
```

普通 Tool 和每个 SubAgent 都独立判断，不使用 Choice 选出唯一赢家。候选只发送名称和完整描述，不限制单个 Tool 的字符数；完整请求估算 Token 超过 `model_context_tokens` 时，按总量平均拆分为多批请求。

- **过滤 Tools** 默认开启：主 LLM 只看到 Jev 达到阈值的普通 Tool。
- **过滤 SubAgents** 默认关闭：主 LLM 默认仍看到全部 SubAgent；打开后只看到达到阈值的候选。
- 无论过滤开关状态，达到阈值的候选都会进入推荐提示。
- WebUI 中每个普通 Tool 都有“始终保留”和“推荐给主 LLM”两个选项。只有先勾选“始终保留”后，第二项才可用；因此可以始终保留一个 Tool，但不主动推荐它。
- SubAgent 的推荐始终由 Jev 逐个判断，WebUI 不提供手工推荐勾选。
- AstrBot 内置 Tool 与 `jev_decide` 首次打开页面时默认勾选“始终保留”，默认不勾选“推荐给主 LLM”。内置 Tool 在列表底部按来源分组显示。

过滤失败时保持原始 ToolSet，避免决策服务故障破坏正常聊天；如果 `jev_decide` 未被勾选始终保留，它会和其它 Tool 一样被移除。

## 两套提示词与备用模型

WebUI 提供：

- **Jev 判断前提示词**：只注入决策请求，不发送 AstrBot 主 LLM 的人格 system prompt。
- **判断后注入主 LLM 的提示词**：支持 `{tools}`、`{subagents}` 占位符，追加在本次请求现有 `system_prompt` 的末尾。

追加操作只在当前请求的 system prompt 上进行，保留 AstrBot 人格和其他插件已追加的内容。主模型失败后由 AstrBot 在同一请求链切换备用模型时，这段 system prompt 会随请求继续传递，不需要插件为每个模型重复注入。

## `jev_decide` 工具

插件通过 `Context.add_llm_tools()` 注册 `jev_decide`，供主 LLM 按需调用。参数支持：

- `decision_type`：`noul`、`choice` 或 `score`
- `state`、`instructions`
- `choice_options`：`{key, description}` 数组
- `score_levels`：有序字符串数组

该工具只返回结构化判断，不执行外部 Tool。

## 主动对话

主动对话总开关在 AstrBot 原生配置页，详细策略在 WebUI。它严格采用白名单：

- 群聊只匹配群聊 ID；
- 私聊只匹配发送者用户 ID；
- 也支持完整 `unified_msg_origin`；
- 白名单为空或不匹配时，不做主动判断，AstrBot 原生聊天照常处理。

白名单会话的普通消息会进入主动判断，但有几个节流点：每个会话有异步锁、判断间隔、回复冷却、窗口次数上限和最大历史；判断间隔内的新消息只记录，不重复请求 Jev。默认判断间隔为 3 秒，所以不是每条消息都发起网络请求。复读、密集对话和会话状态只作为 Jev 判断上下文，最终是否主动回复仍由 Jev 决定。

主动判断一次请求中同时评估五个 Noul 维度：是否指向机器人、是否适合介入、是否有相关价值、时机是否合适、是否能自然延续。综合阈值或“直接指向 + 自然介入”双阈值满足时，插件通过 `event.request_llm()` 进入 AstrBot 原生 Agent。

`direct_reply_prefixes` 默认是 `/` 和 `@`：

- 普通前缀命中后，跳过主动对话 Jev，直接唤醒 AstrBot 原生 Agent；
- 特殊值 `@` 只匹配消息链中的真正 At 机器人节点，不把普通文本中的 `@` 当作唤醒；
- 直接唤醒产生的正常 Agent 请求仍会经过 Tools/SubAgent 决策链，所以不会跳过能力过滤。

主动判断失败时保持不主动回复；Tools/SubAgent 判断失败时保持原始能力集合。Jev 的超时默认 5 秒，失败后默认立即重试 1 次；最大重试次数和重试间隔是全局设置，三条决策链共用。内置主动回复 `provider_ltm_settings.active_reply` 已开启时，插件不再重复执行 ambient 主动判断。某些平台只有 @Bot 消息才会把群聊消息推送给 Bot，插件无法判断未收到的消息。

## 安装与运行配置

AstrBot 要求 `>=4.28.1`。将运行时文件放入：

`data/plugins/astrbot_plugin_decision/`

在 AstrBot 原生配置页填写服务地址、API 路径、API 密钥和模型；需要能力决策时打开 Tools/SubAgent 总开关，需要主动回复时打开主动对话总开关，然后进入插件 WebUI 完成详细配置。

服务地址默认留空，不预置第三方地址。API 密钥字段使用 `secret: true` 仅做界面遮罩，AstrBot 负责保存，不写入日志、README 或 Git。

控制台只输出两类正常决策摘要：

- `因为……所以主动对话/不主动对话`
- `共筛选出如下工具……`

Jev 请求失败、重试和最终错误仍会输出，便于排查。

管理员诊断命令：

```text
/decision status
/decision test
/decision tools
```

## 开发检查

```bash
python -m pytest -q
ruff check .
ruff format --check .
git diff --check
```

`tests/`、Git 元数据、CI 文件和开发脚本只留在源码仓库，不复制到服务器插件运行目录。

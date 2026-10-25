# JEV 综合审计追踪清单

更新时间：2026-10-25（Asia/Shanghai）

本清单把本次审计要求映射到真实代码入口、行为测试和可执行证据。带“本地模拟”的项目只证明插件逻辑和失败降级；它不代表已经调用真实模型、真实平台或产生真实付费任务。

| 要求 | 实现文件与真实入口 | 行为测试 | 当前证据与边界 |
| --- | --- | --- | --- |
| GIF/APNG/动态 WebP 多帧，兼容原生动图且不重复转换 | `decision/multimodal/preprocess.py` `prepare_image`/`animated_montage`；`runtime.py` `sanitize_request_images` | `tests/test_multimodal_integration.py` 动图拼图、静态图片不重编码 | 本地生成 GIF 通过；未在生产 Provider 上发送真实动图 |
| 视频短片抽帧、长片多轮 Agentic、音轨融合 | `decision/multimodal/runtime.py` `_run_video`；`preprocess.py` 视频/音频工具 | `tests/test_multimodal_integration.py::test_long_video_runs_real_rounds` | Provider 调用是 stub；需要密钥和视频权限才能做外部实测 |
| Gemini 原生视频与官方 Agentic 路线 | `decision/multimodal/native_video.py` Files API、`generateContent`、`/v1beta/interactions` | `tests/test_multimodal_documents.py` 原生上传清理、Interactions 路由、能力标注 | 代码使用官方接口；本环境无 Gemini 密钥，未作付费/网络实测 |
| PDF 按页/混合扫描 OCR，Tesseract/PaddleOCR 互补 | `preprocess.py` PDF 渲染；`runtime.py` `_pdf_sparse_pages`/`_pdf_image_regions`/`_ocr_document`；`ocr.py` | `tests/test_multimodal_documents.py` 混合页图片区域、OCR 后端选择与降级；集成测试覆盖取消清理 | 混合页即使原生文本超过 200 字也会检查嵌入图片区域；融合结果按行去重；实际 OCR 准确率依赖主机安装与字体 |
| DOCX/PPTX 结构、整页渲染、嵌入图片和复杂页面视觉材料 | `decision/multimodal/office.py` `render_pages`；`runtime.py` `_materialize_office_visuals` | `tests/test_multimodal_documents.py` 文档/幻灯片/嵌图/整页渲染 | 本地渲染函数会把完整 PDF 页面转为 PNG；真实 Provider 未调用，缺少 LibreOffice 时明确安全回退 |
| XLSX 物理坐标精读、10 万行以后的按需读取 | `office.py` `_xlsx_sheet_window`/`_xlsx_target_window`/`read_target` | `test_xlsx_far_targets_stream_real_coordinates_and_metadata`、`test_xlsx_dense_workbook_after_row_100000_has_bounded_index`，原有单列/单行/空行回归 | 真实生成 100,005 行密集表和含第 1,048,576 行的稀疏表；概览与精读均流式读取，范围外列不混入；隐藏行、合并区域、公式及缓存值保留；源文件变化/失效、取消/超时/输出截断明确提示，未读取不冒充空值 |
| Provider 从 AstrBot 列表选择，UI 能力与真实路线一致 | `adapters.py` `provider_catalog`/`classify_kind`；`main.py` WebUI API | `tests/test_multimodal_core.py`、`test_multimodal_ui.py` | 本地 Provider double 通过；VPS 使用实际 `https://api.yiyinfaith.com/v1/chat/completions` + `gemini-3.8-flash` 纯文本探针返回 `DECISION_REAL_API_OK`；宿主版本测试因未配置 `DECISION_REAL_ASTRBOT_SOURCE` 跳过 |
| 自动理解异步、默认空白名单、不唤醒不回复 | `main.py` `multimodal_observe_message`；`config.py` 默认值；`runtime.py` `observe` | `tests/test_multimodal_integration.py`、`test_multimodal_core.py` | 本地事件链通过；无真实平台外发 |
| 工具独立开关、定位当前/历史附件、自定义指令/精读 | `decision/multimodal/tools.py`、`runtime.py`、`cache.py` | `tests/test_multimodal_core.py` 工具/缓存/会话隔离 | 本地调用链通过 |
| 会话隔离、SingleFlight、自动/工具双模式协作与增量缓存 | `cache.py`；`runtime.py` `observe`/`understand`/`inject` | `tests/test_multimodal_runtime.py` 自动关闭仍索引、通用指令复用、特定问题增量调用、并发合并、目标隔离、媒体版本 | 本地验证实际 Provider 调用次数；不同会话、媒体版本、文件 target、模型、策略和分析深度严格隔离 |
| JEV 与主 LLM 临时注入，不污染永久历史 | `main.py` `_media_placeholder`/`multimodal_prepare_request`；`runtime.py` `_append_text`；`life_agent.py` 历史保存 | `tests/test_life_native.py`、`test_multimodal_integration.py` | 本地请求与历史断言通过 |
| SSRF、大小、超时、取消、临时文件清理 | `multimodal/media.py` `_public_resolution`/`_PinnedResolver`；`preprocess.py` `run_command`；runtime 清理 | 集成 SSRF 测试、固定 DNS 解析器测试、FFmpeg 取消测试、原生上传取消测试 | 请求发送前以已校验地址绑定连接，禁用代理和重定向；本地公网/私网/DNS rebinding/取消通过；未对外部 URL 做真实下载 |
| 手动发起遵守会话类型开关、名单、免打扰、额度、冷却、Jev | `life_runtime.py` `_attempt_unsolicited`/`can_commit`；`virtual_life/core/proactive.py`；`engine.py` `/生活 立即` | `tests/test_life_native.py`、`test_life_sleep_quiet.py`、`test_proactive_policy.py` | 本地真实事件 double 通过；未向生产会话发送 |
| 日程生成不阻塞消息钩子，任务合并/取消 | `life_runtime.py` `record_incoming`/`_queue_ensure_scheduled`/`terminate` | `test_record_incoming_does_not_wait_for_schedule_generation` | 本地慢模型事件链通过 |
| 主动媒体只使用明确计费账户 | `life_media.py` `_billing_event`；`life_config.py`；schema/README | `test_life_billing_fallback.py`、`test_life_media_billing.py` | 未配置账户安全拒绝；Decision 仅把固定身份传给 AIGen；真实金额、价格表达式、余额和视频实际秒数由 AIGen `billed_call` 统一结算；未产生真实付费媒体 |
| 主动媒体每日额度按实际媒体单位且不超限 | `life_media.py` reservation/reconciliation；AIGen `aigen_completed_media_units` 协作扩展 | `test_life_media_capture.py` 预占超限、批量实际单位、部分失败、重载取消 | 先按本次最大请求原子预占，完成后按 AIGen 报告的实际照片张数/视频段数/音频段数结算；旧 AIGen 或替身无扩展时安全回退到请求数 |
| Jev、模型、媒体、发送、平台离线故障分域 | `life_agent.py` 异常类型；`life_runtime.py` `_attempt_unsolicited` | `test_jev_and_platform_failures_have_distinct_runtime_state` | 本地故障注入通过 |
| 睡眠免打扰豁免默认关闭、跨午夜状态 | `life_runtime.py` `_quiet_exempt`；`virtual_life/schema.json`；持久化模型 | `test_life_sleep_quiet.py`、`test_life_midnight_split.py` | 本地状态/默认值通过 |
| 重复触发、部分分段失败、会话更新/人格变化失效 | `life_runtime.py` delivery key；`life_agent.py` guarded send/history | `tests/test_life_native.py` 重复、部分发送、人格变化；`test_life_persistence.py` | 本地事件链通过 |
| 生命周期恢复、插件热更新清理、重复定时任务避免 | `life_runtime.py` `initialize`/`terminate`；`virtual_life/core/storage.py` 原子保存 | `test_initialize_restores_cache_schedules_and_stops_only_own_jobs` | 本地生命周期通过；VPS 使用 AstrBot 单插件 reload 接口返回 HTTP 200，之后只读检查确认单个 AstrBot screen/Python 进程与服务端口正常 |
| 全量回归与代码质量 | 全仓库 | `python -m pytest -q`、`python -m ruff check .`、`git diff --check` | Decision 当前：649 passed, 1 skipped, 12 subtests passed；`ruff check .` clean；常规回归中的 1 项为原生宿主集成入口跳过，另在 VPS 隔离目录中使用实际 AstrBot Builder/Runner 单独运行该套集成，43 passed。AIGen `tests/`：381 passed, 1 skipped, 230 subtests passed；其目标生成/计费回归集 104 passed, 42 subtests passed；AIGen 仓库的全量 `ruff` 仍有既有基线告警，未将其冒充为 clean |
| GitHub/VPS 同提交、备份、健康与回滚 | Git、部署脚本/命令与本清单 | 发布后补充 Commit、远端路径、备份路径、热重载和健康结果 | Decision `ba856c3` 已合并并推送 GitHub `main`，AIGen 协作提交 `c90707e` 已推送其 `main`；Decision `/root/yiyin/AstrBot/data/plugins/astrbot_plugin_decision/main.py` SHA-256 `8f579b68a2f68de71600b51da84867da74f74c6fdcb97b00a2de77df4553f1a7`，AIGen `/root/yiyin/AstrBot/data/plugins/astrbot_plugin_aigen/main.py` SHA-256 `3dfa506c9adc9520e72373060d492967dd7abe37be6f1541158b53030e35bfd2`，与本地 `main` 一致；写入前备份为 `/root/astrbot_backups/decision_final_20261024T165000Z/`，包含两个插件代码和各自 `plugin_data`。两个插件均通过 AstrBot `/api/plugin/reload` 单插件接口返回 HTTP 200；screen PID `568866`、AstrBot Python PID `568938`、端口 6185/8088 均保持正常，插件列表确认 Decision 与 AIGen 均 activated。 |

## 外部依赖边界

### 2026-10-25 用户保存模型后的真实复验与渠道对照

本节为最新外部验收状态，覆盖后文首次盘点时的模型空值、Files 被 WAF 拒绝等历史边界。功能代码基线为 `535d83aaacd30832154b9dd3989077393035342c`；以下请求使用已部署代码及隔离缓存，没有修改生产模型、渠道、余额、价格或发送真实平台消息。

- 用户已将默认及图片、视频、音频、文件模型全部保存为 `吟枢苑API/gemini-3.8-flash`。从实际 Provider Source 与模型条目合并配置，使用真实 AstrBot OpenAI Provider 重新运行九项验收：静态图片、三帧 GIF、24 帧视频和静音音轨、英语音频转录、混合 PDF、整页 PPTX、DOCX 第二页、XLSX A401:A403、XLSX B401:B403，九项均通过，各一次真实 HTTP 请求。XLSX 两列独有内容没有交叉，DOCX 没有返回第一页独有文字。
- `https://api.yiyinfaith.com/v1beta/models/gemini-3.8-flash:generateContent` 直接传入完整 MP4 的 `inline_data` 成功，使用量包含 VIDEO 和 AUDIO，最终结果描述花朵绽放。这与抽帧验证分开记录。
- WAF 放行后，站点 `/upload/v1beta/files` 返回 HTTP 200、`text/html`、标题 New API，缺少 `X-Goog-Upload-URL`，不能视为上传成功；`/v1beta/interactions` 仍返回 `Invalid URL` 404。网关实际为官方镜像 `v1.0.0-rc.46`，镜像 revision 为 `7d8b91268dab69d7150f345c448b5e250163146a`。镜像对应公开路由与插件 API 已核对；宿主项目目录的 Git 源码版本较旧，不能据此代替运行镜像版本。

真实上游对照不经过 New API 站点，不能用来宣称站点已打通相同能力：

| 渠道与请求 | 实际结果 | 验收结论 |
| --- | --- | --- |
| 330 Gemini Developer API，最初按名称选取，用户随后明确目标为 331 | 官方 `generativelanguage.googleapis.com` 上传初始化、完整 MP4 上传、ACTIVE、文件引用 generateContent、文字 Interactions、后台视频 Agentic Interactions 均成功 | 证明现有 Decision 原生实现可与真实官方服务完成调用；不代表用户指定的 331 或站点路由成功 |
| 330 远端资源清理 | Files DELETE 200；删除后 GET 403（不存在或无权限）；同一凭据完整 LIST 200、0 个文件。两条测试 Interactions DELETE 均为 200 | 通过实际删除响应与完整认证列表确认文件不再存在，不能把 Google 删除后 403 误报为删除失败 |
| 331，Google Agent platform，类型 41，启用状态 | `aiplatform.googleapis.com` 普通 `/v1/publishers/google/models/gemini-3.8-flash:generateContent` 200，返回校验文字；按 2026-10-22 官方文档的 Express 路径 `/v1beta1/locations/global/interactions`，文字与完整视频均 404 | 普通模型调用成功；指定渠道的 Interactions 远端成功未验证。Developer API 的 `/upload/v1beta/files` 及 `/v1beta/interactions` 也不能直接套到该域名 |
| 229，CPA 反重力渠道，启用状态，沿用模型映射 `gemini-3.8-flash-high` | 直接请求同一 CPA 容器，Files 初始化 404；Interactions 文字与完整 MP4 请求 200/completed；返回 ID 的 GET 与 DELETE 404；generateContent 文字对照 200 | 存在单次 Interactions 兼容入口，未证明原生后台执行、查询、取消和文件管理能力 |
| 229 实际视频内容对照 | 与官方测试视频大小及 SHA256 相同；实际画面为红/橙红花瓣，回答却描述为粉红到乳白渐变 | 传输和返回格式可用，但本次视觉理解质量验收失败，不能仅凭 HTTP 200 或答案含“花”记为通过 |

331 与 229 均按用户明确指定重新验收，渠道启用状态保持原样。330 的既有停用状态也未修改。所有测试凭据仅在受保护的 SSH/进程内传递，没有写入公开证据。

New API 插件可行性：rc.46 `meta.routes` 支持自定义原生路由、任务归属与计费，并可热加载；其解码器只接收 JSON、表单和 multipart 引用，原始二进制不暴露给 JS；原生 presenter 返回 JSON，不提供完整 HTTP 响应头/状态的透明代理接口，`utils.fetch` 只允许 GET/HEAD。因此可以实现需要客户端适配的多模态任务入口，不能把普通 JS 任务插件直接等同于完整 Google resumable Files 与 Interactions 协议代理。特别是兼容转换不能凭空补出上游缺少的 Agentic 执行能力。当前未修改 New API 源码、安装新插件或变更 Nginx 路由。

证据在 AstrBot VPS 的 `/root/astrbot_backups/decision_full_audit_20261025/followup-current-models/`（`current-config-summary.json`、各 `live-*.json`、`native-inline-video.json`、`native-endpoints*.json`、`official-direct-*.json`、`official-cleanup.json`、`vertex331-*.json`）；229 的独立结果在 API VPS `/root/decision-native-probes/`。以上目录含私有测试记录，不上传 GitHub。用户要求跳过纯文字 `send_voice` 实测，继续保留原有参考音频克隆配置。

### 2026-10-25 全量审计与真实调用追加记录

以下为本轮新增证据；上表和后面的 2026-10-24 记录保留为历史，不代表当前仍未进行外部验证。审计使用最新 main `69ef953` 起点，无回退、无宿主升级、无生产会话外发。

本轮修复覆盖：

- 多模态并发理解把 UMO 与处理策略同时纳入任务兼容判定；独立任务受插件生命周期管理，一个等待者取消不会取消其他调用者，最后一个等待者取消及热重载会清理正在运行的任务。
- 通用指令换说法复用概览；短的具体问题不再被当成概览，缺失问题采用包含已有理解的增量请求，补充结果保留概览；普通 Provider 对象、思考字段、工具元数据不会被字符串化为答案。
- 缓存加载与写入取消竞态、只有媒体索引的会话清理不落盘、错误 JSON 索引结构；已知 Provider Key 在异常截断前脱敏。
- SSRF 特殊 IPv4/IPv6、NAT64/过渡地址和 IDN Host 一致性；直接保留组件的 HTTP 引用，防止 AstrBot 便捷转换器提前发出未验证下载；下载遵守配置大小上限并检查 Content-Length；FFmpeg/ffprobe 仅开放本地协议，禁止恶意播放列表从解码器二次访问内网。
- 无扩展名的签名附件下载沿用平台原始文件名格式；Office 压缩包展开体积与本地文件大小约束；长文本首块不再因超出预算而整块丢弃。
- Office 转换使用 bubblewrap 网络/PID/文件系统命名空间、只读系统库与 LibreOffice 系统配置、独立 profile、宏安全等级 3、CPU/内存/描述符限制和取消清理。真实 PPTX/DOCX 页图通过 LibreOffice 与 PyMuPDF 得到，降级资源不再冒充完整页面。指定章节按 PDF 书签映射，无法映射时明确降级。
- PPTX 数字页码先于标题匹配；DOCX 物理页与页范围正确处理；定向渲染不额外附带其他页的内嵌图片。PDF 巨大范围改为惰性并按实际页数限制，首个长文本页保留内容，整页与图片区域渲染约束像素预算。
- 长视频各轮保留用户问题和抽帧上限，传递同一音轨结论；实测发现静音音轨被模型编造声音后，加入数字静音检测与“声音未验证”约束，并实际重新验证。
- Gemini GenerateContent 排除 thought；Interactions 依据官方 `steps/model_output.content` schema 只接受 completed 的最终输出，空的最终输出不回用中间答案；incomplete、客户端工具要求和未知状态明确失败并取消。Key 不写入错误响应体、日志和 URL。
- 多模态调用显式使用宿主 `request_max_retries=1`（支持该参数的 Provider），由 `mm_retries` 统一控制尝试次数，避免宿主默认五次与插件重试相乘。
- OutputPro 以 UMO 与运行实例隔离状态，热重载清理；私聊 OutContext 保持空 group_id，避免被撤回/分段步骤当成群消息；初始化失败与单步骤退出失败不阻碍其他资源清理。Jev 拒绝 NaN 分数。
- AIGen 协作修复：Gemini 等不接受上游 `n` 的生图路线仍能处理逻辑数量 1–10；按单张调用现有 `_generate -> billed_call`，只应用一次工具冷却，实际成功数量累加，失败停止后续提交。原生支持批量的路线继续一次请求。

真实验收材料与结果：

| 材料/链路 | 实际输入与断言 | 结果 |
| --- | --- | --- |
| 静态图片、公开照片 | 校准图红色三角形/蓝圆/7429；真实 MDN 花朵视频截帧 | 正确识别；使用实际 AstrBot OpenAI Provider 与 `gemini-3.8-flash` |
| GIF | 3 个独立 RGB 帧，红→蓝→绿与帧号 | 实际请求包含 3 张图，顺序正确 |
| 短视频及多轮精读 | 公共 MDN 5.055 秒花朵视频，6 帧短片与独立精读窗口 | 正确描述开花变化；静音修复后不再虚构音乐/水声 |
| 音频 | 公开恐龙吼声；FFmpeg flite 英语语音，含 7429 | 识别非语言声音、准确转录语音 |
| 混合 PDF | 同页超过 200 字原生文本 CEDAR-9624 + 扫描图片 ORCHID-5831 | 实际模型收到完整页与图片区域并读到两个编码；真实 Tesseract 本地 OCR 也读到两者，0 次 Provider 调用 |
| PPTX | 原生箭头、两色方框、橙色叠加椭圆、Q1/Q2/Q3 图表 10/20/35 | 实际请求含 1080×810 完整幻灯片图，箭头方向/重叠对象/数值正确 |
| DOCX | 第二页 LILAC-2847 与表格 Orchid/17 | 实际请求只附第二页 918×1188 整页图，未返回第一页独有文字 |
| XLSX | A401:A403 与 B401:B403 具有不同独有值 | 两次实际请求精确隔离列；原有百万物理行/空行/隐藏行/公式测试保留 |
| 双模式与缓存 | 自动关闭仍索引/工具可读；自动开启并发工具；通用换说法；缺失事实追问；重复追问；跨 UMO | 按实际 HTTP 计数检查；并发一次、通用复用零次、增量一次、重复零次；其他会话不能召回 |
| SSRF 网络对照 | 自有监听器仅绑定 127.0.0.1；DNS 首次公网、随后内网 | 修复路径内网收到 0 请求；旧式校验后重解析对照收到 1 请求；下载无临时文件泄漏 |
| Office 隔离 | 沙箱内尝试访问宿主 /root/yiyin 与 dashboard | 目录不可见、网络不通；取消后转换目录清理 |
| Jev | 实际 jev-latest 同次 noul/choice/score | 1 次实际调用；noul 0.88、choice red、score 1.96。最初探针遗漏 instructions 的 400 是验收脚本错误，已纠正 |
| 原生日程/关系生成 | 周一高中人格、校服、不运动/不喝咖啡；一个同班好友 | 实际 Gemini 生成通过结构校验，日程满足角色约束，关系草稿通过严格验证 |
| 原生 Agent | 真实 AstrBot Builder/Runner + 真实 Gemini，捕获平台发送 | 回答 NATIVE-AGENT-7429、注入人格并保存历史；未向平台外发 |
| 生图协作 | Decision→真实 AIGen→gemini-3.1-flash-image→Gemini 理解，绿色方块/黄色星形 | 实际返回并解码验证 30042 字节 JPEG，实际媒体单位 1；再次真实理解识别绿色正方形与黄色五角星。财务验证使用生产账本私有副本，该模型未产生内部收费订单，没有改生产资金；不据此宣称生产扣费已验证 |
| WebUI | 真浏览器读取生产 settings/tools/life/models，点击分区与选择 OCR 策略 | 9 个下拉框浅色文字/深色背景；标题正确；更改后未保存，刷新恢复生产设置 |

验收目录：`/root/astrbot_backups/decision_full_audit_20261025/`，证据在 `evidence/`；生产发布与逐文件哈希以同目录的 release manifest 和部署报告为准。备份含私有配置/账本，不应公开上传。

最终回归：VPS 隔离副本完整 pytest 为 **691 passed, 1 skipped, 12 subtests passed**，另以真实 AstrBot 源单独运行原生集成为 **43 passed**，真实 Provider 原生 Agent 验收为 **1 passed**。Decision 全量 ruff 与 diff whitespace 检查通过。Windows 全套为 690 passed（宿主入口与缺少 ffprobe 的实进程检查跳过）；ffprobe 内网阻断检查已在有实际二进制的 VPS 通过，且真实视频探测、抽帧、音轨抽取/静音检测通过。AIGen 全套为 **384 passed, 1 skipped, 230 subtests passed**，变更文件 ruff 通过；其跳过项是本地宿主配置加载器源码依赖。常规测试的 Provider/发送替身和本节明确列出的真实 HTTP/真实模型验收分别记录，不混为一类。

边界保持明确：该 API 站点对 Gemini Files 初始化上传返回 HTTP 403，因此原生 Google 上传/Interactions 的远端成功未验证；实际通用视频链路已通过。PaddleOCR 未安装，使用并实测 Tesseract。AIGen 当前文字生音频自动路由没有唯一有效路线，真实尝试被本地校验阻止，未提交/扣费。生产多模态默认模型目前留空，未擅自修改配置；真实多模态验证使用隔离的明确模型配置。真实平台发送、生产余额扣费、所有上游模型组合与真实视频生成不宣称已通过。

2026-10-24 追加验证：删除重复的自动理解后端总开关，配置默认值、校验、归一化、读取/保存及运行状态均不再保留该设置；自动理解使用 `mm_enabled`、媒体类型开关和会话名单，主动工具继续独立。旧客户端提交的已删除字段会被忽略。更新后的本地全量回归为 **649 passed, 1 skipped, 12 subtests passed**，ruff 与 `git diff --check` 通过。新增测试检查实际 Provider 替身调用次数、指定 XLSX 区域传入模型的内容、重复区域缓存命中和取消信号；不代表真实模型理解质量。对比修复前后的代码，旧 `read_target` 对同一真实文件的有界概览会把 A100001 误报为空，新实现从源文件读到独有文字；旧自动理解依赖隐藏开关，新实现只需现有总开关和媒体类型开关。

VPS 环境回归额外修正了测试边界：人际网络服务单测显式注入请求/响应替身，避免依赖真实 Web 请求 ContextVar；PDF 测试实际断言空白页错误及两页独有文字；原生集成夹具适配异步 Provider 选择、现有媒体 kind 参数、工具迁移路径和 AIGen 相对导入。原生测试同时发现并修复 `life_agent.py` 向新宿主 `MainAgentBuildConfig.fallback_provider_ids` 传递现有备用模型列表的遗漏；使用真实 AstrBot 框架和 Provider/平台替身验证，43 项全部通过。以上不修改 AstrBot 核心，也不产生真实模型调用、外发消息或付费媒体。

本次真实 Provider 仅执行了一次低成本纯文本探针：VPS 已配置的 `gemini-3.8-flash` 返回 `DECISION_REAL_API_OK`。没有执行真实图片/视频/音频多模态请求、真实媒体扣费或真实群聊/私聊发送；因此多模态模型理解质量、Office 视觉效果和 AIGen 生产账单仍以本地行为测试与生产运行边界为准。部署保持现有自动发送配置，不主动触发测试消息；若宿主未提供兼容 AstrBot 源，宿主原生测试继续明确标为 skipped。

## 生产盘点与边界

- 盘点到的 AstrBot 运行目录为 `/root/yiyin/AstrBot`，插件目录为 `/root/yiyin/AstrBot/data/plugins/astrbot_plugin_decision`，服务单元为 `astrbot.service`，启动方式是 `/root/.local/bin/uv run --project /root/yiyin/AstrBot /root/yiyin/AstrBot/main.py`，由 detached `screen` 托管。
- 写入前备份了插件代码、AstrBot 配置目录、插件持久化目录、虚拟生活计划/会话、媒体缓存和人际网络 SQLite；部署后已恢复插件目录内原有的 `data/cmd_config.json`、`data/skills.json` 和模板，哈希与旧目录一致。
- 部署早期曾按原流程重启 AstrBot 服务，随后插件更新通过 AstrBot 单插件 reload 接口返回 HTTP 200；这是已发生的操作记录，不代表本次收尾又重启。用户要求后续插件更新使用热重载，本次收尾只执行只读核验。插件元数据继续使用最低兼容版本范围，没有新增 AstrBot 精确版本锁；宿主原有 `pyproject.toml`/`uv.lock` 未修改，盘点时宿主 Git 工作树已有大量未提交改动。
- 最新只读健康检查：AstrBot Python 进程数为 1、screen 数为 1，`astrbot.service` 为 `ActiveState=active`/`SubState=exited`（oneshot + `RemainAfterExit` 设计），端口 6185、8088 继续监听；reload 前后 PID 保持为 screen `568866`、Python `568938`。没有执行生产会话外发、真实模型调用或付费媒体任务。
- 回滚：从 `/root/astrbot_backups/decision_audit_20261023T164301Z/plugin_previous` 恢复插件代码，保留当前配置和全部持久化目录，再调用 AstrBot 单插件 reload 接口并复核状态；不要为插件回滚停止或重启整个 AstrBot 服务。配置和持久化归档保持原路径不变。

# JEV 综合审计追踪清单

更新时间：2026-10-24（Asia/Shanghai）

本清单把本次审计要求映射到真实代码入口、行为测试和可执行证据。带“本地模拟”的项目只证明插件逻辑和失败降级；它不代表已经调用真实模型、真实平台或产生真实付费任务。

| 要求 | 实现文件与真实入口 | 行为测试 | 当前证据与边界 |
| --- | --- | --- | --- |
| GIF/APNG/动态 WebP 多帧，兼容原生动图且不重复转换 | `decision/multimodal/preprocess.py` `prepare_image`/`animated_montage`；`runtime.py` `sanitize_request_images` | `tests/test_multimodal_integration.py` 动图拼图、静态图片不重编码 | 本地生成 GIF 通过；未在生产 Provider 上发送真实动图 |
| 视频短片抽帧、长片多轮 Agentic、音轨融合 | `decision/multimodal/runtime.py` `_run_video`；`preprocess.py` 视频/音频工具 | `tests/test_multimodal_integration.py::test_long_video_runs_real_rounds` | Provider 调用是 stub；需要密钥和视频权限才能做外部实测 |
| Gemini 原生视频与官方 Agentic 路线 | `decision/multimodal/native_video.py` Files API、`generateContent`、`/v1beta/interactions` | `tests/test_multimodal_documents.py` 原生上传清理、Interactions 路由、能力标注 | 代码使用官方接口；本环境无 Gemini 密钥，未作付费/网络实测 |
| PDF 按页/混合扫描 OCR，Tesseract/PaddleOCR 互补 | `preprocess.py` PDF 渲染；`runtime.py` `_pdf_sparse_pages`/`_pdf_image_regions`/`_ocr_document`；`ocr.py` | `tests/test_multimodal_documents.py` 混合页图片区域、OCR 后端选择与降级；集成测试覆盖取消清理 | 混合页即使原生文本超过 200 字也会检查嵌入图片区域；融合结果按行去重；实际 OCR 准确率依赖主机安装与字体 |
| DOCX/PPTX 结构、整页渲染、嵌入图片和复杂页面视觉材料 | `decision/multimodal/office.py` `render_pages`；`runtime.py` `_materialize_office_visuals` | `tests/test_multimodal_documents.py` 文档/幻灯片/嵌图/整页渲染 | 本地渲染函数会把完整 PDF 页面转为 PNG；真实 Provider 未调用，缺少 LibreOffice 时明确安全回退 |
| XLSX 超 400 行、空行/隐藏行、区域精读 | `office.py` `_xlsx`/`read_target` | `test_xlsx_indexes_rows_beyond_400_and_supports_range_reads`、`test_xlsx_ranges_use_physical_rows_when_blank_rows_are_present` | 本地稀疏 450 行工作簿通过；行索引有 100,000 行上限并保留物理坐标 |
| Provider 从 AstrBot 列表选择，UI 能力与真实路线一致 | `adapters.py` `provider_catalog`/`classify_kind`；`main.py` WebUI API | `tests/test_multimodal_core.py`、`test_multimodal_ui.py` | 本地 Provider double 通过；宿主版本测试因未配置 `DECISION_REAL_ASTRBOT_SOURCE` 跳过 |
| 自动理解异步、默认空白名单、不唤醒不回复 | `main.py` `multimodal_observe_message`；`config.py` 默认值；`runtime.py` `observe` | `tests/test_multimodal_integration.py`、`test_multimodal_core.py` | 本地事件链通过；无真实平台外发 |
| 工具独立开关、定位当前/历史附件、自定义指令/精读 | `decision/multimodal/tools.py`、`runtime.py`、`cache.py` | `tests/test_multimodal_core.py` 工具/缓存/会话隔离 | 本地调用链通过 |
| 会话隔离、SingleFlight、自动/工具双模式协作与增量缓存 | `cache.py`；`runtime.py` `observe`/`understand`/`inject` | `tests/test_multimodal_runtime.py` 自动关闭仍索引、通用指令复用、特定问题增量调用、并发合并、目标隔离、媒体版本 | 本地验证实际 Provider 调用次数；不同会话、媒体版本、文件 target、模型、策略和分析深度严格隔离 |
| JEV 与主 LLM 临时注入，不污染永久历史 | `main.py` `_media_placeholder`/`multimodal_prepare_request`；`runtime.py` `_append_text`；`life_agent.py` 历史保存 | `tests/test_life_native.py`、`test_multimodal_integration.py` | 本地请求与历史断言通过 |
| SSRF、大小、超时、取消、临时文件清理 | `multimodal/media.py` `_public_resolution`/`_PinnedResolver`；`preprocess.py` `run_command`；runtime 清理 | 集成 SSRF 测试、固定 DNS 解析器测试、FFmpeg 取消测试、原生上传取消测试 | 请求发送前以已校验地址绑定连接，禁用代理和重定向；本地公网/私网/DNS rebinding/取消通过；未对外部 URL 做真实下载 |
| 手动发起遵守会话类型开关、名单、免打扰、额度、冷却、Jev | `life_runtime.py` `_attempt_unsolicited`/`can_commit`；`virtual_life/core/proactive.py`；`engine.py` `/生活 立即` | `tests/test_life_native.py`、`test_life_sleep_quiet.py`、`test_proactive_policy.py` | 本地真实事件 double 通过；未向生产会话发送 |
| 日程生成不阻塞消息钩子，任务合并/取消 | `life_runtime.py` `record_incoming`/`_queue_ensure_scheduled`/`terminate` | `test_record_incoming_does_not_wait_for_schedule_generation` | 本地慢模型事件链通过 |
| 主动媒体只使用明确计费账户 | `life_media.py` `_billing_event`；`life_config.py`；schema/README | `test_life_billing_fallback.py`、`test_life_media_billing.py` | 未配置账户安全拒绝；未产生真实付费媒体 |
| Jev、模型、媒体、发送、平台离线故障分域 | `life_agent.py` 异常类型；`life_runtime.py` `_attempt_unsolicited` | `test_jev_and_platform_failures_have_distinct_runtime_state` | 本地故障注入通过 |
| 睡眠免打扰豁免默认关闭、跨午夜状态 | `life_runtime.py` `_quiet_exempt`；`virtual_life/schema.json`；持久化模型 | `test_life_sleep_quiet.py`、`test_life_midnight_split.py` | 本地状态/默认值通过 |
| 重复触发、部分分段失败、会话更新/人格变化失效 | `life_runtime.py` delivery key；`life_agent.py` guarded send/history | `tests/test_life_native.py` 重复、部分发送、人格变化；`test_life_persistence.py` | 本地事件链通过 |
| 生命周期恢复、插件热更新清理、重复定时任务避免 | `life_runtime.py` `initialize`/`terminate`；`virtual_life/core/storage.py` 原子保存 | `test_initialize_restores_cache_schedules_and_stops_only_own_jobs` | 本地生命周期通过；VPS 使用 AstrBot 单插件 reload 接口返回 HTTP 200，之后只读检查确认单个 AstrBot screen/Python 进程与服务端口正常 |
| 全量回归与代码质量 | 全仓库 | `python -m pytest -q`、`python -m ruff check .`、`git diff --check` | 当前：612 passed, 2 skipped, 12 subtests passed；ruff clean。2 项为未配置真实 AstrBot 源的宿主集成跳过 |
| GitHub/VPS 同提交、备份、健康与回滚 | Git、部署脚本/命令与本清单 | 发布后补充 Commit、远端路径、备份路径、热重载和健康结果 | 本轮已部署 `ed90efd8856236233b37f2e88403962e03241a6a`；VPS 插件路径为 `/root/yiyin/AstrBot/data/plugins/astrbot_plugin_decision`，`main.py` SHA-256 为 `1af37c8bd24f1a43ed4ffb3de702757212e380e3bb676f75dffefd407fd0689f`；备份归档为 `/root/astrbot_backups/decision_audit_20261023T164301Z/decision_state.tar.gz`，SHA-256 为 `e8ec13b3e159f0ab455a90760ec684e8d39f590e183110948c623ebda3e5c361`；AstrBot 单插件 reload 接口返回 HTTP 200。只读复核时 screen PID 为 `568866`、AstrBot Python PID 为 `568938`，各一个；服务为 `active/exited`（oneshot + `RemainAfterExit`），启动时间 `2026-10-24 01:02:04 CST`，端口 6185、8088 正常监听；当前插件、备份归档及保留的两个配置文件 SHA-256 与部署记录一致。旧插件目录和运行数据已保存在同一备份目录。 |

## 外部依赖边界

本次没有 Gemini/其它模型密钥、管理员指定测试会话或生产平台外发授权，因此没有执行真实模型请求、真实媒体扣费、真实群聊/私聊发送。部署只复制代码并保持现有自动发送配置，不主动触发测试消息；若宿主未提供兼容 AstrBot 源，宿主原生测试继续明确标为 skipped。

## 生产盘点与边界

- 盘点到的 AstrBot 运行目录为 `/root/yiyin/AstrBot`，插件目录为 `/root/yiyin/AstrBot/data/plugins/astrbot_plugin_decision`，服务单元为 `astrbot.service`，启动方式是 `/root/.local/bin/uv run --project /root/yiyin/AstrBot /root/yiyin/AstrBot/main.py`，由 detached `screen` 托管。
- 写入前备份了插件代码、AstrBot 配置目录、插件持久化目录、虚拟生活计划/会话、媒体缓存和人际网络 SQLite；部署后已恢复插件目录内原有的 `data/cmd_config.json`、`data/skills.json` 和模板，哈希与旧目录一致。
- 部署早期曾按原流程重启 AstrBot 服务，随后插件更新通过 AstrBot 单插件 reload 接口返回 HTTP 200；这是已发生的操作记录，不代表本次收尾又重启。用户要求后续插件更新使用热重载，本次收尾只执行只读核验。插件元数据继续使用最低兼容版本范围，没有新增 AstrBot 精确版本锁；宿主原有 `pyproject.toml`/`uv.lock` 未修改，盘点时宿主 Git 工作树已有大量未提交改动。
- 最新只读健康检查：AstrBot Python 进程数为 1、screen 数为 1，`astrbot.service` 为 `ActiveState=active`/`SubState=exited`（oneshot + `RemainAfterExit` 设计），端口 6185、8088 继续监听。没有执行生产会话外发、真实模型调用或付费媒体任务。当前 PID 不能单独证明热重载前后的 PID 一致；本记录仅写入可观察到的 reload HTTP 200 和此刻进程/端口状态。
- 回滚：从 `/root/astrbot_backups/decision_audit_20261023T164301Z/plugin_previous` 恢复插件代码，保留当前配置和全部持久化目录，再调用 AstrBot 单插件 reload 接口并复核状态；不要为插件回滚停止或重启整个 AstrBot 服务。配置和持久化归档保持原路径不变。

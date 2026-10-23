# JEV 综合审计追踪清单

更新时间：2026-10-24（Asia/Shanghai）

本清单把本次审计要求映射到真实代码入口、行为测试和可执行证据。带“本地模拟”的项目只证明插件逻辑和失败降级；它不代表已经调用真实模型、真实平台或产生真实付费任务。

| 要求 | 实现文件与真实入口 | 行为测试 | 当前证据与边界 |
| --- | --- | --- | --- |
| GIF/APNG/动态 WebP 多帧，兼容原生动图且不重复转换 | `decision/multimodal/preprocess.py` `prepare_image`/`animated_montage`；`runtime.py` `sanitize_request_images` | `tests/test_multimodal_integration.py` 动图拼图、静态图片不重编码 | 本地生成 GIF 通过；未在生产 Provider 上发送真实动图 |
| 视频短片抽帧、长片多轮 Agentic、音轨融合 | `decision/multimodal/runtime.py` `_run_video`；`preprocess.py` 视频/音频工具 | `tests/test_multimodal_integration.py::test_long_video_runs_real_rounds` | Provider 调用是 stub；需要密钥和视频权限才能做外部实测 |
| Gemini 原生视频与官方 Agentic 路线 | `decision/multimodal/native_video.py` Files API、`generateContent`、`/v1beta/interactions` | `tests/test_multimodal_documents.py` 原生上传清理、Interactions 路由、能力标注 | 代码使用官方接口；本环境无 Gemini 密钥，未作付费/网络实测 |
| PDF 按页/混合扫描 OCR，Tesseract/PaddleOCR 互补 | `preprocess.py` PDF 渲染；`runtime.py` `_pdf_sparse_pages`/`_ocr_document`；`ocr.py` | `tests/test_multimodal_documents.py` OCR 后端选择与降级；集成测试覆盖取消清理 | 本地后端状态和失败降级通过；实际 OCR 准确率依赖主机安装与字体 |
| DOCX/PPTX 结构、嵌入图片和复杂页面视觉材料 | `decision/multimodal/office.py`；`runtime.py` `_materialize_office_visuals` | `tests/test_multimodal_documents.py` 文档/幻灯片/嵌图 | 本地生成 Office 文件通过；未向真实视觉 Provider 发请求 |
| XLSX 超 400 行、区域/分块精读 | `office.py` `_xlsx`/`read_target` | `test_xlsx_indexes_rows_beyond_400_and_supports_range_reads` | 本地 450 行工作簿通过 |
| Provider 从 AstrBot 列表选择，UI 能力与真实路线一致 | `adapters.py` `provider_catalog`/`classify_kind`；`main.py` WebUI API | `tests/test_multimodal_core.py`、`test_multimodal_ui.py` | 本地 Provider double 通过；宿主版本测试因未配置 `DECISION_REAL_ASTRBOT_SOURCE` 跳过 |
| 自动理解异步、默认空白名单、不唤醒不回复 | `main.py` `multimodal_observe_message`；`config.py` 默认值；`runtime.py` `observe` | `tests/test_multimodal_integration.py`、`test_multimodal_core.py` | 本地事件链通过；无真实平台外发 |
| 工具独立开关、定位当前/历史附件、自定义指令/精读 | `decision/multimodal/tools.py`、`runtime.py`、`cache.py` | `tests/test_multimodal_core.py` 工具/缓存/会话隔离 | 本地调用链通过 |
| 会话隔离、SingleFlight、历史文字追问召回 | `cache.py`；`runtime.py` `understand`/`inject` | `test_multimodal_core.py`、`test_multimodal_integration.py` | 本地并发/缓存通过 |
| JEV 与主 LLM 临时注入，不污染永久历史 | `main.py` `_media_placeholder`/`multimodal_prepare_request`；`runtime.py` `_append_text`；`life_agent.py` 历史保存 | `tests/test_life_native.py`、`test_multimodal_integration.py` | 本地请求与历史断言通过 |
| SSRF、大小、超时、取消、临时文件清理 | `multimodal/media.py` `assert_public_url`；`preprocess.py` `run_command`；runtime 清理 | 集成 SSRF 测试、FFmpeg 取消测试、原生上传取消测试 | 本地私网/DNS rebinding/取消通过；未对外部 URL 做真实下载 |
| 手动发起遵守会话类型开关、名单、免打扰、额度、冷却、Jev | `life_runtime.py` `_attempt_unsolicited`/`can_commit`；`virtual_life/core/proactive.py`；`engine.py` `/生活 立即` | `tests/test_life_native.py`、`test_life_sleep_quiet.py`、`test_proactive_policy.py` | 本地真实事件 double 通过；未向生产会话发送 |
| 日程生成不阻塞消息钩子，任务合并/取消 | `life_runtime.py` `record_incoming`/`_queue_ensure_scheduled`/`terminate` | `test_record_incoming_does_not_wait_for_schedule_generation` | 本地慢模型事件链通过 |
| 主动媒体只使用明确计费账户 | `life_media.py` `_billing_event`；`life_config.py`；schema/README | `test_life_billing_fallback.py`、`test_life_media_billing.py` | 未配置账户安全拒绝；未产生真实付费媒体 |
| Jev、模型、媒体、发送、平台离线故障分域 | `life_agent.py` 异常类型；`life_runtime.py` `_attempt_unsolicited` | `test_jev_and_platform_failures_have_distinct_runtime_state` | 本地故障注入通过 |
| 睡眠免打扰豁免默认关闭、跨午夜状态 | `life_runtime.py` `_quiet_exempt`；`virtual_life/schema.json`；持久化模型 | `test_life_sleep_quiet.py`、`test_life_midnight_split.py` | 本地状态/默认值通过 |
| 重复触发、部分分段失败、会话更新/人格变化失效 | `life_runtime.py` delivery key；`life_agent.py` guarded send/history | `tests/test_life_native.py` 重复、部分发送、人格变化；`test_life_persistence.py` | 本地事件链通过 |
| 重启恢复、热更新清理、重复定时任务避免 | `life_runtime.py` `initialize`/`terminate`；`virtual_life/core/storage.py` 原子保存 | `test_initialize_restores_cache_schedules_and_stops_only_own_jobs` | 本地生命周期通过；VPS 重启验证在部署阶段记录 |
| 全量回归与代码质量 | 全仓库 | `python -m pytest -q`、`python -m ruff check .`、`git diff --check` | 当前：602 passed, 2 skipped, 12 subtests passed；ruff clean。2 项为未配置真实 AstrBot 源的宿主集成跳过 |
| GitHub/VPS 同提交、备份、健康与回滚 | Git、部署脚本/命令与本清单 | 发布后补充 Commit、远端路径、备份路径、服务 PID/健康结果 | 外部部署证据在发布完成后填写；未把代码测试冒充生产验证 |

## 外部依赖边界

本次没有 Gemini/其它模型密钥、管理员指定测试会话或生产平台外发授权，因此没有执行真实模型请求、真实媒体扣费、真实群聊/私聊发送。部署只复制代码并保持现有自动发送配置，不主动触发测试消息；若宿主未提供兼容 AstrBot 源，宿主原生测试继续明确标为 skipped。

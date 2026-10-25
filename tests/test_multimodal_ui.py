"""WebUI structure and the non-text Jev switch semantics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "dashboard" / "index.html"
MAIN = ROOT / "main.py"


def page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_flow_enhancement_has_two_sub_tabs():
    html = page_text()
    assert 'aria-label="对话流增强模式"' in html
    assert 'flowTab = \'input\'' in html
    assert 'flowTab = \'output\'' in html
    assert "flowTab: 'input'" in html, "input must be the default sub-tab"


def test_each_flow_card_belongs_to_one_sub_tab():
    html = page_text()
    # Input group: the original analysis card plus the new multi-modal card.
    assert html.count('v-show="flowTab === \'input\'" class="glass-card"') == 2
    # Output group: the original output card plus the output pipeline card.
    assert 'v-show="flowTab === \'output\'" class="glass-card"' in html
    assert 'class="glass-card output-config-card"' in html


def test_original_input_and_output_settings_are_preserved():
    html = page_text()
    for key in (
        "conversation_flow_analysis_enabled",
        "conversation_flow_window",
        "conversation_flow_scope",
        "conversation_flow_scope_blacklist",
        "output_enhancement_enabled",
        "output_scope",
        "output_scope_blacklist",
        "output_pipeline",
    ):
        assert key in html, key


def test_flow_cards_keep_top_tabs_and_name_left_card_correctly():
    html = page_text()
    assert "<h2>对话流分析</h2>" in html
    assert "<h2>多模态理解</h2>" in html
    # The two top-level mode tabs retain their original labels.
    assert "对话流分析与多模态理解" in html
    assert "OutputPro 输出管道与阶梯" in html


def test_multimodal_section_is_under_input_enhancement():
    html = page_text()
    assert "多模态理解" in html
    for key in (
        "mm_enabled",
        "mm_tools_enabled",
        "mm_stream_enabled",
        "mm_gateway_openai_stream",
        "mm_auto_image",
        "mm_auto_video",
        "mm_auto_audio",
        "mm_auto_file",
        "mm_tool_image",
        "mm_tool_video",
        "mm_tool_audio",
        "mm_tool_file",
        "mm_default_model",
        "mm_model_image",
        "mm_model_video",
        "mm_model_audio",
        "mm_model_file",
        "mm_auto_scope",
    ):
        assert key in html, key
    # There is one multimodal master; automatic understanding uses the media
    # type switches and session scope without a second hidden master.
    assert 'v-model="mm.multimodal.mm_auto_enabled"' not in html
    assert '<strong>启用多模态理解</strong>' in html


def test_input_selects_use_readable_dark_glass_theme():
    html = page_text()
    assert '.dialogue-flow-page select:not([multiple])' in html
    assert 'appearance: none' in html
    assert 'color-scheme: dark' in html
    assert '.dialogue-flow-page select:not([multiple]) option' in html
    assert '#eaf3ff' in html


def test_model_dropdowns_are_populated_from_the_provider_catalog():
    html = page_text()
    assert "mmModels" in html
    assert "重新加载模型列表" in html
    assert "reloadModels" in html
    # The catalog comes from the plugin's own endpoint, not a hardcoded list.
    assert "multimodal/models" in html
    assert "multimodal/models" in MAIN.read_text(encoding="utf-8")


def test_non_text_switch_defaults_to_off_and_gates_the_judgement():
    main = MAIN.read_text(encoding="utf-8")
    assert '"non_text_proactive_enabled": False' in main
    # Inside the pure-media branch the switch is still the gate.
    block = main[main.index("        if not text:") : main.index("        if not text:") + 700]
    assert 'self._bool("non_text_proactive_enabled", False)' in block
    assert "_media_placeholder" in block


def test_media_detection_does_not_depend_on_multimodal_understanding():
    """The switch must work with mm_enabled off, so detection cannot rely on refs."""

    main = MAIN.read_text(encoding="utf-8")
    assert "async def _media_refs_for" in main
    body = main[main.index("async def _media_refs_for") :][:900]
    assert "extract_media(event)" in body, "must fall back to direct extraction"
    assert "decision_mm_refs" in body, "reuse the refs when understanding already ran"


def test_placeholder_reads_the_cache_through_the_public_api():
    main = MAIN.read_text(encoding="utf-8")
    body = main[main.index("async def _media_placeholder") :][:1600]
    assert "cache._entries" not in body, "must not poke the private cache dict"
    assert "get_for_media" in body
    assert "recent_understandings" in body


def test_multimodal_does_not_run_inside_on_llm_request_only():
    """Auto understanding needs a message-level hook, not just the LLM one."""

    main = MAIN.read_text(encoding="utf-8")
    assert "async def multimodal_observe_message" in main
    hook = main.index("async def multimodal_observe_message")
    segment = main[hook : hook + 2000]
    assert "event_message_type" in main[hook - 200 : hook]
    assert "runtime.observe(event" in segment


def test_tools_are_registered_through_the_shared_tool_list():
    main = MAIN.read_text(encoding="utf-8")
    assert "_sync_multimodal_tools" in main
    assert "self.context.add_llm_tools(tool)" in main
    assert "self._life_tools.append(tool)" in main

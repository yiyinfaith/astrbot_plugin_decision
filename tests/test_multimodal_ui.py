"""WebUI structure and the non-text Jev switch semantics."""

from __future__ import annotations

import re
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


def test_multimodal_section_is_under_input_enhancement():
    html = page_text()
    assert "多模态理解" in html
    for key in (
        "mm_enabled",
        "mm_auto_enabled",
        "mm_tools_enabled",
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
    # The pure-text early return must consult the switch and the media refs.
    gate = re.search(
        r"media_refs = event\.get_extra\(\"decision_mm_refs\"\) or \[\]\n"
        r"\s+if not text:.*?"
        r'if not self\._bool\("non_text_proactive_enabled", False\)',
        main,
        re.S,
    )
    assert gate, "pure media messages must be gated by the new switch"


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

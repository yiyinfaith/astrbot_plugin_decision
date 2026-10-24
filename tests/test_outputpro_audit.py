"""Behavioral checks for output conversation isolation and lifecycle cleanup."""

from types import ModuleType, SimpleNamespace

import pytest

from decision.outputpro_runtime import OutputPipelineRuntime


@pytest.fixture
def model_boundary(monkeypatch):
    import sys

    groups = {}

    class StateManager:
        _groups = groups

        @classmethod
        def get_group(cls, key):
            return groups.setdefault(key, SimpleNamespace(msg_queue=[], name_to_qq={}))

    module = ModuleType("decision.outputpro.core.model")
    module.StateManager = StateManager
    module.StepName = SimpleNamespace(AT="at")
    module.OutContext = SimpleNamespace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return groups


def event(umo, sender, message):
    return SimpleNamespace(
        unified_msg_origin=umo,
        get_group_id=lambda: "123", get_sender_id=lambda: sender,
        get_self_id=lambda: "bot", get_sender_name=lambda: "同名用户",
        message_obj=SimpleNamespace(message_id=message),
    )


async def test_output_state_is_scoped_to_umo_and_cleared_on_close(tmp_path, model_boundary):
    runtime = OutputPipelineRuntime(None, tmp_path)
    runtime._plugin_config = SimpleNamespace(
        reply=SimpleNamespace(threshold=1), at=SimpleNamespace(at_str=False),
        pipeline=SimpleNamespace(is_enabled_step=lambda step: True),
    )
    await runtime.prepare_message(event("qq:GroupMessage:123", "qq-person", "qq-message"))
    await runtime.prepare_message(event("telegram:GroupMessage:123", "telegram-person", "telegram-message"))
    assert len(model_boundary) == 2
    assert {tuple(group.msg_queue) for group in model_boundary.values()} == {("qq-message",), ("telegram-message",)}
    assert {group.name_to_qq["同名用户"] for group in model_boundary.values()} == {"qq-person", "telegram-person"}
    await runtime.close()
    assert model_boundary == {}


async def test_private_output_keeps_empty_group_id_for_platform_sending(tmp_path, model_boundary):
    runtime = OutputPipelineRuntime(None, tmp_path)
    captured = []

    async def run(context):
        captured.append(context)

    runtime._pipeline = SimpleNamespace(run=run)
    private = event("qq:FriendMessage:123", "123", "message")
    private.get_group_id = lambda: ""
    private.get_result = lambda: SimpleNamespace(chain=["original"], is_llm_result=lambda: True, get_plain_text=lambda: "original")
    await runtime.run(private)
    assert len(captured) == 1
    assert captured[0].gid == "" and captured[0].uid == "123"

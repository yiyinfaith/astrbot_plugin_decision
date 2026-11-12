"""Private QQ chats reject At segments (retcode 1400), so output steps must never add them."""

from collections import OrderedDict, deque
from types import SimpleNamespace

import pytest

components = pytest.importorskip("astrbot.core.message.components")
aiocqhttp = pytest.importorskip(
    "astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event"
)

from decision.outputpro.core.model import OutContext  # noqa: E402
from decision.outputpro.core.step.at import AtStep  # noqa: E402
from decision.outputpro.core.step.reply import ReplyStep  # noqa: E402

At, Plain, Reply = components.At, components.Plain, components.Reply


class QQEvent(aiocqhttp.AiocqhttpMessageEvent):
    def __init__(self):  # skip platform wiring; only the fields the steps read
        self.message_obj = SimpleNamespace(message_id="m1")

    def get_platform_name(self):
        return "aiocqhttp"

    def get_sender_id(self):
        return "42"

    def get_sender_name(self):
        return "小明"


def context(chain, gid):
    group = SimpleNamespace(
        msg_queue=deque(["m1", "m2", "m3"]),
        last_reply_mark_msg_id=None,
        name_to_qq=OrderedDict(),
    )
    return OutContext(
        event=QQEvent(), chain=chain, is_llm=True, plain="", gid=gid,
        uid="42", bid="bot", group=group, timestamp=0,
    )


def reply_step():
    cfg = SimpleNamespace(reply=SimpleNamespace(threshold=1, include_at=True))
    return ReplyStep(cfg)


def at_step(prob):
    return AtStep(SimpleNamespace(at=SimpleNamespace(at_str=False, at_prob=prob)))


async def test_smart_reply_quotes_without_at_in_private_chat():
    ctx = context([Plain("在呀")], gid="")
    await reply_step().handle(ctx)
    assert isinstance(ctx.chain[0], Reply)
    assert not any(isinstance(c, At) for c in ctx.chain)
    assert [c.text for c in ctx.chain if isinstance(c, Plain)] == ["在呀"]


async def test_smart_reply_still_mentions_in_group_chat():
    ctx = context([Plain("在呀")], gid="123")
    await reply_step().handle(ctx)
    assert isinstance(ctx.chain[0], Reply) and isinstance(ctx.chain[1], At)


async def test_at_step_never_adds_and_strips_at_in_private_chat():
    ctx = context([Plain("[at:42] 在呀")], gid="")
    await at_step(1.0).handle(ctx)
    assert [type(c) for c in ctx.chain] == [Plain] and ctx.chain[0].text == "在呀"

    ctx = context([At(qq="42"), Plain("​"), Plain("你好")], gid="")
    await at_step(1.0).handle(ctx)
    assert [c.text for c in ctx.chain] == ["你好"]


async def test_at_step_group_behaviour_unchanged():
    ctx = context([Plain("在呀")], gid="123")
    await at_step(1.0).handle(ctx)
    assert isinstance(ctx.chain[0], At)

    ctx = context([Plain("[at:42] 在呀")], gid="123")
    await at_step(0.0).handle(ctx)
    assert isinstance(ctx.chain[0], At)  # at_prob 0 skips smart removal, fake @ resolved

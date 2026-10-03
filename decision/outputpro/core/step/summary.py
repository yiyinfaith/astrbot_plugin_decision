import json
import random
from pathlib import Path

from astrbot.api import logger
from astrbot.core.message.components import Image
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)

from ..config import PluginConfig
from ..model import OutContext, StepName, StepResult
from .base import BaseStep


class SummaryStep(BaseStep):
    name = StepName.SUMMARY

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.cfg = config.summary
        self.quotes = self._load_all_quotes()

    def _load_all_quotes(self) -> list[str]:
        """
        把 summary.quotes 与 summary.quotes_files 里的金句全部合并成一个 list
        """
        quotes: list[str] = list(self.cfg.quotes)

        for file_path in self.cfg.quotes_files:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"金句文件不存在，已跳过：{path}")
                continue
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        quotes.extend(data)
                    else:
                        logger.warning(f"金句文件内容不是 list，已跳过：{path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("读取金句文件失败 %s: %s", path, e)
        return quotes

    async def handle(self, ctx: OutContext) -> StepResult:
        """图片外显（直接发送并中断流水线）"""
        if (
            isinstance(ctx.event, AiocqhttpMessageEvent)
            and len(ctx.chain) == 1
            and isinstance(ctx.chain[0], Image)
        ):
            obmsg = await ctx.event._parse_onebot_json(MessageChain(ctx.chain))
            quote = random.choice(self.quotes)
            obmsg[0]["data"]["summary"] = quote

            await ctx.event.bot.send(ctx.event.message_obj.raw_message, obmsg)  # type: ignore
            ctx.event.should_call_llm(True)
            ctx.chain.clear()

            return StepResult(abort=True, msg=f"已给图片附加外显金句：{quote}")

        return StepResult()

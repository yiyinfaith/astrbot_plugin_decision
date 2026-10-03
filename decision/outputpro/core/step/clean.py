import re
from collections import defaultdict

import emoji

from astrbot.core.message.components import Plain

from ..config import PluginConfig
from ..model import OutContext, StepName, StepResult
from .base import BaseStep


class CleanStep(BaseStep):
    name = StepName.CLEAN

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.cfg = config.clean

    async def handle(self, ctx: OutContext) -> StepResult:
        removed: dict[str, list[str]] = defaultdict(list)

        for seg in ctx.chain:
            if not isinstance(seg, Plain):
                continue

            if len(seg.text) >= self.cfg.text_threshold:
                continue

            # 中括号
            if self.cfg.bracket:
                matches = re.findall(r"\[.*?\]", seg.text)
                if matches:
                    removed["中括号内容"].extend(matches)
                    seg.text = re.sub(r"\[.*?\]", "", seg.text)

            # 圆括号
            if self.cfg.parenthesis:
                matches = re.findall(r"[（(].*?[）)]", seg.text)
                if matches:
                    removed["圆括号内容"].extend(matches)
                    seg.text = re.sub(r"[（(].*?[）)]", "", seg.text)

            # 情绪标签
            if self.cfg.emotion_tag:
                matches = re.findall(r"&&.*?&&", seg.text)
                if matches:
                    removed["情绪标签"].extend(matches)
                    seg.text = re.sub(r"&&.*?&&", "", seg.text)

            # Emoji
            if self.cfg.emoji:
                emojis = [c for c in seg.text if c in emoji.EMOJI_DATA]
                if emojis:
                    removed["Emoji"].extend(emojis)
                    seg.text = emoji.replace_emoji(seg.text, replace="")

            # 前缀
            if self.cfg.lead:
                for s in self.cfg.lead:
                    if seg.text.startswith(s):
                        removed["前缀"].append(s)
                        seg.text = seg.text[len(s) :]
                        break

            # 后缀
            if self.cfg.tail:
                for s in self.cfg.tail:
                    if seg.text.endswith(s):
                        removed["后缀"].append(s)
                        seg.text = seg.text[: -len(s)]
                        break

            # 标点
            if self.cfg.punctuation:
                matches = re.findall(self.cfg.punctuation, seg.text)
                if matches:
                    removed["标点字符"].extend(matches)
                    seg.text = re.sub(self.cfg.punctuation, "", seg.text)

        return StepResult(msg=self._build_msg(removed))

    def _build_msg(self, removed: dict[str, list[str]]) -> str:
        """消息构建（记录删了什么）"""
        if not removed:
            return ""

        parts: list[str] = []

        for k, items in removed.items():
            uniq = list(dict.fromkeys(items))  # 去重但保序
            if len(uniq) == 1:
                parts.append(f"{k}: {uniq[0]}")
            else:
                preview = "、".join(uniq[:3])
                more = f" 等{len(uniq)}项" if len(uniq) > 3 else ""
                parts.append(f"{k}: {preview}{more}")

        return "文本清理：" + "；".join(parts)

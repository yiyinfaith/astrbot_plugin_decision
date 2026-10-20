"""Persona-bound appearance assets and AIGen's public tool adapter."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import inspect
import io
import json
import re
import time
import uuid
from pathlib import Path

from .virtual_life.core.storage import JsonRepository
from .virtual_life.core.utils import format_outfit, timeline_item_at


class AppearanceStore:
    def __init__(self, root: Path):
        self.root = root
        self.repo = JsonRepository(root / "appearance.json", {"personas": {}})
        self.write_lock = asyncio.Lock()

    def folder(self, persona_id):
        return self.root / "appearance" / hashlib.sha256(persona_id.encode()).hexdigest()[:24]

    async def list(self, persona_id, *, previews=False):
        manifest = await self.repo.load()
        items = manifest.get("personas", {}).get(persona_id, [])
        result = []
        for item in items:
            if not isinstance(item, dict) or not re.fullmatch(
                r"[a-f0-9]{32}\.(?:png|jpg|webp)", str(item.get("id", ""))
            ):
                continue
            path = self.folder(persona_id) / item["id"]
            if path.is_file():
                result.append(
                    {
                        **item,
                        **(
                            {
                                "preview": "data:"
                                + item["mime"]
                                + ";base64,"
                                + base64.b64encode(path.read_bytes()).decode()
                            }
                            if previews
                            else {}
                        ),
                    }
                )
        return result

    async def upload(self, persona_id, name, data_url):
        async with self.write_lock:
            return await self._upload(persona_id, name, data_url)

    async def _upload(self, persona_id, name, data_url):
        from PIL import Image

        if (
            not isinstance(data_url, str)
            or len(data_url) > 12_000_000
            or ";base64," not in data_url
        ):
            raise ValueError("请上传不超过 8MB 的 PNG、JPEG 或 WebP 图片")
        raw = base64.b64decode(data_url.split(";base64,", 1)[1], validate=True)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("参考图片不能超过 8MB")
        with Image.open(io.BytesIO(raw)) as picture:
            if (
                picture.format not in {"PNG", "JPEG", "WEBP"}
                or picture.width * picture.height > 25_000_000
            ):
                raise ValueError("参考图格式或尺寸不支持")
            mime = Image.MIME[picture.format]
            picture.verify()
        items = await self.list(persona_id)
        if len(items) >= 5:
            raise ValueError("每个人格最多上传 5 张参考图，请先删除旧图")
        folder = self.folder(persona_id)
        folder.mkdir(parents=True, exist_ok=True)
        ident = (
            uuid.uuid4().hex
            + {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[mime]
        )
        path = folder / ident
        path.write_bytes(raw)
        item = {"id": ident, "name": Path(str(name)).name[:100], "mime": mime}
        manifest = await self.repo.load()
        manifest.setdefault("personas", {})[persona_id] = [*items, item]
        try:
            await self.repo.save(manifest)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return item

    async def remove(self, persona_id, ident):
        async with self.write_lock:
            return await self._remove(persona_id, ident)

    async def _remove(self, persona_id, ident):
        if not re.fullmatch(r"[a-f0-9]{32}\.(?:png|jpg|webp)", str(ident)):
            raise ValueError("参考图标识无效")
        manifest = await self.repo.load()
        items = manifest.get("personas", {}).get(persona_id, [])
        if not any(item["id"] == ident for item in items):
            raise ValueError("参考图不存在或不属于当前人格")
        manifest["personas"][persona_id] = [item for item in items if item["id"] != ident]
        await self.repo.save(manifest)
        (self.folder(persona_id) / ident).unlink(missing_ok=True)


class ReferenceEvent:
    """A request-local view: never mutate the user's or another tool's event."""

    def __init__(self, event, paths, prompt):
        from astrbot.api.message_components import Image, Plain

        self._event = event
        self.message_obj = copy.copy(event.message_obj)
        self.message_obj.message = [
            Plain(prompt),
            *[Image.fromFileSystem(str(path)) for path in paths],
        ]
        self.message_obj.message_str = prompt
        self.message_obj.raw_message = None
        self.message_str = prompt

    def get_messages(self):
        return self.message_obj.message

    def get_message_str(self):
        return self.message_str

    def __getattr__(self, key):
        return getattr(self._event, key)


class LifeMedia:
    def __init__(self, owner, root):
        self.owner = owner
        self.appearance = AppearanceStore(root)
        self.usage = JsonRepository(root / "media_usage.json", {"usage": {}})
        self.locks: dict[str, asyncio.Lock] = {}
        self.last_call: dict[str, float] = {}
        self.usage_lock = asyncio.Lock()

    def aigen(self):
        getter = getattr(self.owner.context, "get_all_stars", None)
        if not callable(getter):
            return None
        matches = [
            star
            for star in getter()
            if getattr(star, "name", "") == "astrbot_plugin_aigen"
            and getattr(star, "activated", False)
        ]
        if len(matches) != 1:
            return None
        instance = getattr(matches[0], "star_cls", None)
        return (
            instance
            if instance
            and callable(getattr(instance, "generate_image", None))
            and callable(getattr(instance, "generate_video", None))
            else None
        )

    async def capture(self, event, *, video=False, scene=""):
        runtime = await self.owner._get_life_runtime()
        settings = runtime.config["media_settings"]
        if not settings["enable"]:
            return "拍照和拍视频已关闭。"
        if not settings["daily_limit"]:
            return "本会话今日拍摄额度已用完。"
        aigen = self.aigen()
        if aigen is None:
            return "aigen 插件未安装、未启用或接口不兼容，尚未提交生成任务。"
        umo = str(event.unified_msg_origin)
        async with self.locks.setdefault(umo, asyncio.Lock()):
            now = time.monotonic()
            if now - self.last_call.get(umo, -1e9) < settings["cooldown_seconds"]:
                return "拍摄冷却中，请稍后再试。"
            persona = await runtime.personas.resolve(umo)
            refs = await self.appearance.list(persona.id)
            if not refs:
                return "当前人格尚未上传外貌参考图，请在 WebUI 的主动发起页面上传后再拍摄。"
            reference_persona = persona.id
            persona, plan = await runtime._ensure_plan_for_umo(umo)
            if persona.id != reference_persona:
                return "准备期间人格已变更，尚未提交拍摄任务。"
            if not plan.has_outfit:
                return "今日穿搭尚未生成；请开启每日穿搭并在 WebUI 生成今日内容后再拍摄。"
            item = timeline_item_at(plan, runtime._now())
            prompt = settings["prompt"]
            values = {
                "kind": "短视频" if video else "照片",
                "appearance": settings["appearance"],
                "outfit": format_outfit(plan),
                "activity": item.activity if item else "自由活动",
                "scene": str(scene or "自然地分享此刻的生活")[:4000],
            }
            for name, value in values.items():
                prompt = prompt.replace("{" + name + "}", str(value))
            if (await runtime.personas.resolve(umo)).id != persona.id:
                return "准备期间人格已变更，尚未提交拍摄任务。"
            if runtime.closed:
                return "设置正在重载，尚未提交拍摄任务。"
            if event.get_extra("decision_life_proactive", False) and not runtime.can_commit(
                umo, event.get_extra("decision_life_generation", -1)
            ):
                return "主动候选已过期，尚未提交拍摄任务。"
            proxy = ReferenceEvent(
                event, [self.appearance.folder(persona.id) / ref["id"] for ref in refs], prompt
            )
            method = aigen.generate_video if video else aigen.generate_image
            kwargs = {
                "event": proxy,
                "prompt": prompt,
                "aspect_ratio": settings["video_aspect_ratio" if video else "image_aspect_ratio"],
            }
            if video:
                kwargs.update(
                    duration=settings["video_duration"],
                    reference_mode="reference",
                    use_message_media=True,
                )
            signature = inspect.signature(method)
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
            if self.aigen() is not aigen:
                return "aigen 正在重载或已停用，尚未提交拍摄任务。"
            # Reserve before calling a potentially paid backend. Never retry
            # automatically after ambiguous transport or platform failures.
            async with self.usage_lock:
                usage = await self.usage.load()
                key = runtime._now().date().isoformat() + "::" + umo
                count = usage["usage"].get(key, 0)
                if count >= settings["daily_limit"]:
                    return "本会话今日拍摄额度已用完。"
                usage["usage"] = {
                    k: v
                    for k, v in usage["usage"].items()
                    if k.startswith(runtime._now().date().isoformat())
                }
                usage["usage"][key] = count + 1
                await self.usage.save(usage)
                self.last_call[umo] = time.monotonic()
            results = []
            try:
                if runtime.closed:
                    return "生活设置已重载，已取消尚未提交的拍摄尝试。"
                outcome = method(**kwargs)
                if inspect.isasyncgen(outcome):
                    async for value in outcome:
                        if isinstance(value, str) and value.strip():
                            results.append(value)
                        elif value is not None:
                            # Older AIGen releases may yield a direct result.
                            await event.send(value)
                elif inspect.isawaitable(outcome):
                    value = await outcome
                    if isinstance(value, str):
                        results.append(value)
            except Exception as exc:
                return f"aigen 拍摄失败：{str(exc)[:180]}。未自动重试，请先确认任务状态。"
            return "\n".join(results) or json.dumps(
                {
                    "status": "submitted",
                    "media_type": "video" if video else "image",
                    "backend": "aigen",
                }
            )

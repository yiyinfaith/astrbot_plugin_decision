"""Persona-bound appearance assets, AIGen bridge and proactive media choice."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import inspect
import io
import json
import random
import re
import time
import uuid
from pathlib import Path

from .virtual_life.core.storage import JsonRepository
from .virtual_life.core.utils import format_outfit, timeline_item_at

# Media kinds understood by the generic capture tools.  ``subject`` only
# influences prompt wording; every kind still goes through the same AIGen call.
SUBJECTS: dict[str, str] = {
    "self": "自己",
    "other": "别人",
    "scene": "眼前的景色",
    "object": "身边的物品",
    "screenshot": "手机屏幕内容（截图或录屏）",
}
MEDIA_KINDS = ("photo", "video", "audio")
KIND_LABELS = {"photo": "照片", "video": "短视频", "audio": "音频"}
KIND_TOOLS = {"photo": "take_photo", "video": "take_video", "audio": "send_voice"}


def media_settings(runtime) -> dict:
    """Return the media group of a live runtime with defaults applied."""

    settings = (runtime.config or {}).get("media_settings", {}) or {}
    return settings if isinstance(settings, dict) else {}


def clamp_count(settings: dict, kind: str, value: object) -> int:
    """Clamp a model-supplied count into the configured inclusive range."""

    low = max(0, min(10, int(settings.get(f"{kind}_count_min", 1))))
    high = max(low, min(10, int(settings.get(f"{kind}_count_max", low))))
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = low
    return max(low, min(high, count))


def enabled_kinds(settings: dict) -> list[str]:
    """Return the media kinds that are both enabled and reachable."""

    if not settings.get("enable", True):
        return []
    return [kind for kind in MEDIA_KINDS if settings.get(f"{kind}_enable", False)]


def choose_proactive_media(settings: dict) -> tuple[str, int, int] | None:
    """Pick at most one media kind for one proactive delivery.

    Each enabled kind is rolled independently against its own probability; when
    several pass, exactly one is drawn with those probabilities as weights.  The
    returned range is the configured inclusive count range for the winner.
    """

    candidates: list[tuple[str, float]] = []
    for kind in enabled_kinds(settings):
        try:
            probability = float(settings.get(f"{kind}_probability", 0.0) or 0.0)
        except (TypeError, ValueError):
            probability = 0.0
        if probability <= 0.0:
            continue
        if random.random() < min(1.0, probability):
            candidates.append((kind, probability))
    if not candidates:
        return None
    if len(candidates) == 1:
        kind = candidates[0][0]
    else:
        kind = random.choices(
            [item[0] for item in candidates], weights=[item[1] for item in candidates], k=1
        )[0]
    low = max(0, min(10, int(settings.get(f"{kind}_count_min", 1))))
    high = max(low, min(10, int(settings.get(f"{kind}_count_max", low))))
    return kind, low, high


def media_instruction(kind: str, low: int, high: int) -> str:
    """Describe one allowed media attachment for the proactive prompt."""

    tool = KIND_TOOLS[kind]
    label = KIND_LABELS[kind]
    if high <= 0:
        return ""
    if kind == "audio":
        # Each call speaks one text, so the configured range is a per-run budget
        # that the model spends by calling the tool again with new content.
        quantity = (
            f"最多 {low} 段" if low == high else f"最多 {high} 段（至少 {low} 段）"
        )
        return (
            f"本次主动发起可以附带{label}。若要开口，请调用 {tool} 工具，每次生成一段，"
            f"本次共{quantity}，多段时每段内容或情绪应有所不同，由你根据场景自行决定。"
            "若此刻说话不如文字自然，也可以不发。"
        )
    unit = "张" if kind == "photo" else "段"
    if low == high:
        quantity = f"恰好 {low} {unit}"
    else:
        quantity = f"{low}~{high} {unit}"
    return (
        f"本次主动发起可以附带{label}。若要分享，请调用 {tool} 工具生成，"
        f"数量{quantity}，由你根据场景自行决定；若此刻不适合分享，也可以不发。"
    )


class AppearanceStore:
    def __init__(self, root: Path):
        self.root = root
        self.repo = JsonRepository(root / "appearance.json", {"personas": {}, "descriptions": {}})
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

    async def get_description(self, persona_id: str) -> str:
        manifest = await self.repo.load()
        descriptions = manifest.get("descriptions", {})
        value = descriptions.get(str(persona_id), "") if isinstance(descriptions, dict) else ""
        return str(value or "")[:4000]

    async def set_description(self, persona_id: str, description: str) -> str:
        value = str(description or "").strip()[:4000]
        async with self.write_lock:
            manifest = await self.repo.load()
            descriptions = manifest.setdefault("descriptions", {})
            if not isinstance(descriptions, dict):
                descriptions = {}
                manifest["descriptions"] = descriptions
            if value:
                descriptions[str(persona_id)] = value
            else:
                descriptions.pop(str(persona_id), None)
            await self.repo.save(manifest)
        return value

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

    def aigen(self, kind: str | None = None):
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
        if instance is None:
            return None
        if kind in MEDIA_KINDS:
            return instance if callable(getattr(instance, {"photo": "generate_image", "video": "generate_video", "audio": "generate_audio"}[kind], None)) else None
        return instance if any(callable(getattr(instance, method, None)) for method in ("generate_image", "generate_video", "generate_audio")) else None

    async def capture(
        self, event, *, kind="photo", scene="", subject="self", count=None, audio_text=""
    ):
        """Generate one media kind through AIGen.

        ``kind`` is ``photo``, ``video`` or ``audio``.  Every kind shares the
        same guardrails (master switch, per-kind switch, cooldown, daily quota,
        reference images, daily outfit) so a single tool surface cannot bypass
        them by choosing a different kind.
        """

        runtime = await self.owner._get_life_runtime()
        settings = media_settings(runtime)
        if not settings.get("enable", True):
            return "多媒体生成已关闭。"
        if kind not in MEDIA_KINDS:
            return "未知的多媒体类型。"
        if not settings.get(f"{kind}_enable", False):
            return f"{KIND_LABELS[kind]}生成已关闭。"
        if not settings.get("daily_limit"):
            return "本会话今日生成额度已用完。"
        aigen = self.aigen(kind)
        if aigen is None:
            return "aigen 插件未安装、未启用或接口不兼容，尚未提交生成任务。"
        audio_method = getattr(aigen, "generate_audio", None)
        if kind == "audio":
            if not str(audio_text or "").strip():
                return "生成音频需要提供要朗读或表达的内容。"
            if not callable(audio_method):
                return "当前 aigen 版本不提供 generate_audio，尚未提交生成任务。"
        umo = str(event.unified_msg_origin)
        async with self.locks.setdefault(umo, asyncio.Lock()):
            now = time.monotonic()
            if now - self.last_call.get(umo, -1e9) < settings["cooldown_seconds"]:
                return "多媒体生成冷却中，请稍后再试。"
            persona = await runtime.personas.resolve(umo)
            uses_persona = str(subject or "self") == "self"
            refs = await self.appearance.list(persona.id) if uses_persona else []
            appearance_text = await self.appearance.get_description(persona.id) if uses_persona else ""
            if not refs and kind != "audio" and uses_persona:
                return "当前人格尚未上传外貌参考图，请在 WebUI 的主动发起页面上传后再拍摄。"
            reference_persona = persona.id
            plan = None
            if uses_persona and kind != "audio":
                persona, plan = await runtime._ensure_plan_for_umo(umo)
                if persona.id != reference_persona:
                    return "准备期间人格已变更，尚未提交生成任务。"
            if uses_persona and kind != "audio" and not plan.has_outfit:
                appearance = (runtime.config or {}).get("appearance_settings", {}) or {}
                if not appearance.get("use_reference_outfit", True):
                    return (
                        "今日穿搭尚未生成；请开启每日穿搭并在 WebUI 生成今日内容后再拍摄，"
                        "或开启“无今日穿搭时沿用参考图穿搭”。"
                    )
            item = timeline_item_at(plan, runtime._now()) if plan is not None else None
            # A proactive run receives one preselected media kind.  Enforce it
            # here as well as in the prompt so a tool loop cannot bypass the
            # single-kind and quantity budget by calling another wrapper.
            proactive = event.get_extra("decision_life_proactive", False)
            allowed_kind = event.get_extra("decision_life_media_kind", "")
            if proactive and allowed_kind and kind != allowed_kind:
                return f"本次主动发起只允许发送{KIND_LABELS.get(allowed_kind, allowed_kind)}。"
            try:
                used = max(0, int(event.get_extra("decision_life_media_used", 0) or 0))
            except (TypeError, ValueError):
                used = 0
            max_allowed = event.get_extra("decision_life_media_max", None)
            try:
                max_allowed = None if max_allowed is None else max(0, int(max_allowed))
            except (TypeError, ValueError):
                max_allowed = None
            if proactive and max_allowed is not None and used >= max_allowed:
                return "本次主动发起的多媒体数量已达到上限。"
            resolved_count = clamp_count(settings, kind, 1 if count is None else count)
            # One ``send_voice`` call reads exactly one text, so asking aigen for
            # several clips in one call would only duplicate the same utterance.
            # Audio therefore always produces one clip per call, and the
            # configured range acts as a per-run budget that the model spends by
            # calling the tool again with different content.
            if kind == "audio":
                resolved_count = 1
            if proactive and max_allowed is not None:
                resolved_count = min(resolved_count, max(0, int(max_allowed) - used))
            if resolved_count <= 0:
                return "本次生成数量为 0，未提交生成任务。"
            prompt = self._render_prompt(
                settings,
                runtime,
                kind=kind,
                subject=subject,
                scene=scene,
                audio_text=audio_text,
                plan=plan,
                item=item,
                count=resolved_count,
                appearance_text=appearance_text,
            )
            if (await runtime.personas.resolve(umo)).id != persona.id:
                return "准备期间人格已变更，尚未提交生成任务。"
            if runtime.closed:
                return "设置正在重载，尚未提交生成任务。"
            if event.get_extra("decision_life_proactive", False) and not runtime.can_commit(
                umo, event.get_extra("decision_life_generation", -1)
            ):
                return "主动候选已过期，尚未提交生成任务。"
            proxy = (
                ReferenceEvent(
                    event, [self.appearance.folder(persona.id) / ref["id"] for ref in refs], prompt
                )
                if uses_persona and refs
                else event
            )
            if kind == "photo":
                method = aigen.generate_image
            elif kind == "video":
                method = aigen.generate_video
            else:
                method = audio_method
            kwargs = self._call_kwargs(settings, kind, proxy, prompt, resolved_count, audio_text)
            signature = inspect.signature(method)
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
            if self.aigen(kind) is not aigen:
                return "aigen 正在重载或已停用，尚未提交生成任务。"
            # Reserve before calling a potentially paid backend. Never retry
            # automatically after ambiguous transport or platform failures.
            async with self.usage_lock:
                usage = await self.usage.load()
                key = runtime._now().date().isoformat() + "::" + umo
                quota = usage["usage"].get(key, 0)
                if quota >= settings["daily_limit"]:
                    return "本会话今日生成额度已用完。"
                usage["usage"] = {
                    k: v
                    for k, v in usage["usage"].items()
                    if k.startswith(runtime._now().date().isoformat())
                }
                usage["usage"][key] = quota + 1
                await self.usage.save(usage)
                self.last_call[umo] = time.monotonic()
            # ``n`` lets one image request return several pictures. Video and
            # audio have no batch parameter: video issues one fresh generation
            # per item, while audio is already pinned to a single clip above.
            batched = "n" in signature.parameters
            calls = 1 if batched else resolved_count
            results = []
            try:
                if runtime.closed:
                    return "生活设置已重载，已取消尚未提交的生成尝试。"
                for index in range(calls):
                    if index and runtime.closed:
                        break
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
                if not results:
                    return f"aigen 生成失败：{str(exc)[:180]}。未自动重试，请先确认任务状态。"
                results.append(f"aigen 部分生成失败：{str(exc)[:180]}")
            if proactive:
                try:
                    event.set_extra("decision_life_media_used", used + resolved_count)
                except AttributeError:
                    pass
            return "\n".join(results) or json.dumps(
                {
                    "status": "submitted",
                    "media_type": kind,
                    "count": resolved_count,
                    "backend": "aigen",
                }
            )

    def _render_prompt(
        self,
        settings,
        runtime,
        *,
        kind,
        subject,
        scene,
        audio_text,
        plan,
        item,
        count,
        appearance_text="",
    ) -> str:
        template = str(settings.get(f"{kind}_prompt", "") or "")
        appearance = appearance_text or self._appearance_text(runtime, settings)
        has_outfit = bool(getattr(plan, "has_outfit", False))
        outfit = (
            format_outfit(plan)
            if has_outfit
            else "今日尚未生成穿搭，请沿用外貌参考图中的服装，不要自行发明新的服装。"
        )
        values = {
            "kind": KIND_LABELS[kind],
            "subject": SUBJECTS.get(str(subject or "self"), SUBJECTS["self"]),
            "subject_desc": self._subject_description(subject),
            "appearance": appearance,
            "outfit": outfit,
            "activity": item.activity if item else "自由活动",
            "scene": str(scene or "自然地分享此刻的生活")[:4000],
            "style": str(scene or "自然地表达").strip()[:2000],
            "text": str(audio_text or "").strip()[:4000],
            "count": str(count),
        }
        for name, value in values.items():
            template = template.replace("{" + name + "}", str(value))
        return template

    @staticmethod
    def _subject_description(subject) -> str:
        key = str(subject or "self")
        if key == "self":
            return "当前人格本人"
        return SUBJECTS.get(key, SUBJECTS["self"])

    @staticmethod
    def _appearance_text(runtime, settings) -> str:
        life = getattr(runtime, "config", {}) or {}
        appearance = (life.get("appearance_settings", {}) or {}).get("description", "")
        text = str(appearance or "").strip()
        return text or "未填写外貌补充描述，请完全依据参考图。"

    def _call_kwargs(self, settings, kind, event, prompt, count, audio_text=""):
        if kind == "photo":
            return {
                "event": event,
                "prompt": prompt,
                "aspect_ratio": settings["image_aspect_ratio"],
                "n": count,
            }
        if kind == "video":
            return {
                "event": event,
                "prompt": prompt,
                "aspect_ratio": settings["video_aspect_ratio"],
                "duration": settings["video_duration"],
                "reference_mode": "reference",
                "use_message_media": True,
            }
        # AIGen reads ``text`` literally in 朗读 mode, so the spoken content must
        # be the caller's text rather than the descriptive template.
        mode = str(settings.get("audio_mode", "朗读") or "朗读")
        spoken = str(audio_text or "").strip()
        kwargs = {
            "event": event,
            "text": spoken if mode == "朗读" and spoken else prompt,
            "mode": mode,
        }
        voice = str(settings.get("audio_voice", "") or "").strip()
        if voice:
            kwargs["voice"] = voice
        return kwargs

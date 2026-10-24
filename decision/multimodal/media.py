"""Discover media attachments on an incoming AstrBot message.

Kept separate from the pipeline so both the auto-understanding hook and the LLM
tools can look at the same message in exactly the same way.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import socket
import time
from pathlib import Path
from typing import Any

from aiohttp.abc import AbstractResolver
from astrbot import logger

from .models import MediaIndexEntry, MediaRef

# Component class names in AstrBot's message model.
_KIND_BY_CLASS = {
    "Image": "image",
    "Video": "video",
    "Record": "audio",
    "File": "file",
}


def _kind_of(component: Any) -> str:
    name = type(component).__name__
    if name in _KIND_BY_CLASS:
        return _KIND_BY_CLASS[name]
    declared = getattr(component, "type", None)
    value = getattr(declared, "value", declared)
    if isinstance(value, str):
        for candidate, kind in _KIND_BY_CLASS.items():
            if value.lower().startswith(candidate.lower()):
                return kind
    return ""


async def _resolve_reference(component: Any) -> str:
    """Turn a component into something the provider layer can read."""

    kind = _kind_of(component)
    # Preserve HTTP references for our pinned, bounded downloader. AstrBot's
    # convenience converters may fetch them eagerly through an unvalidated
    # connection, bypassing this module's SSRF boundary entirely.
    for attribute in ("path", "url", "file", "file_"):
        reference = getattr(component, attribute, None)
        if isinstance(reference, str) and (
            reference.lower().startswith(("https://", "http://")) or is_local_file(reference)
        ):
            return reference
    if kind == "file":
        getter = getattr(component, "get_file", None)
        if callable(getter):
            return str(await getter() or "")
        return str(getattr(component, "file_", "") or getattr(component, "url", "") or "")
    converter = getattr(component, "convert_to_file_path", None)
    if callable(converter):
        return str(await converter())
    return str(
        getattr(component, "path", "")
        or getattr(component, "url", "")
        or getattr(component, "file", "")
        or ""
    )


def _component_name(component: Any, kind: str) -> str:
    if kind == "file":
        return str(getattr(component, "name", "") or "")
    reference = str(
        getattr(component, "path", "")
        or getattr(component, "url", "")
        or getattr(component, "file", "")
        or ""
    )
    base = os.path.basename(reference.split("?", 1)[0])
    return base[:120]


def guess_mime(reference: str, kind: str) -> str:
    suffix = Path(reference.split("?", 1)[0]).suffix.lower()
    guessed = mimetypes.guess_type(f"x{suffix}")[0]
    if guessed:
        return guessed
    return {
        "image": "image/*",
        "video": "video/*",
        "audio": "audio/*",
        "file": "application/octet-stream",
    }.get(kind, "application/octet-stream")


async def extract_media(event: Any, *, limit: int = 16) -> list[MediaRef]:
    """Return every media attachment of the message, in order.

    A message may carry several attachments; each becomes its own reference so the
    caller can understand them one by one.
    """

    getter = getattr(event, "get_messages", None)
    components = []
    if callable(getter):
        try:
            components = list(getter() or [])
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Decision Engine 读取消息组件失败：%s", str(exc)[:120])
            return []
    if not components:
        message_obj = getattr(event, "message_obj", None)
        components = list(getattr(message_obj, "message", None) or [])

    umo = str(getattr(event, "unified_msg_origin", "") or "")
    message_id = str(getattr(event, "message_obj", None) and getattr(event.message_obj, "message_id", "") or "")
    if not message_id:
        message_id = f"local-{int(time.time() * 1000)}"
    received_at = time.time()

    refs: list[MediaRef] = []
    for component in components:
        kind = _kind_of(component)
        if not kind:
            continue
        if len(refs) >= limit:
            logger.info("Decision Engine 一条消息中的媒体超过 %s 个，忽略其余部分", limit)
            break
        try:
            reference = await _resolve_reference(component)
        except Exception as exc:
            logger.warning(
                "Decision Engine 无法读取%s附件：%s", kind, str(exc)[:160]
            )
            continue
        if not reference:
            continue
        index = len(refs)
        refs.append(
            MediaRef(
                kind=kind,
                reference=reference,
                urn=f"{umo}::{message_id}::{index}",
                umo=umo,
                message_id=message_id,
                name=_component_name(component, kind),
                mime=guess_mime(reference, kind),
                index=index,
                received_at=received_at,
            )
        )
    return refs


def index_rows(refs: list[MediaRef], sender_id: str = "") -> list[MediaIndexEntry]:
    return [
        MediaIndexEntry(
            umo=ref.umo,
            message_id=ref.message_id,
            kind=ref.kind,
            index=ref.index,
            name=ref.name,
            reference=ref.reference,
            content_key=ref.content_key,
            received_at=ref.received_at,
            sender_id=sender_id,
        )
        for ref in refs
    ]


def is_local_file(reference: str) -> bool:
    try:
        return bool(reference) and Path(reference).is_file()
    except OSError:
        return False


async def ensure_local(reference: str, *, timeout: float = 60.0, max_bytes: int = 200 * 1024 * 1024) -> str:
    """Return a local path for a reference, downloading http(s) sources safely.

    Only http(s) with a public-looking host is attempted; the download is bounded
    and written to a temporary file that the caller removes.
    """

    if is_local_file(reference):
        return reference
    if not reference.lower().startswith(("http://", "https://")):
        raise ValueError("媒体引用不是本地文件也不是可下载链接")
    return await asyncio.wait_for(_download(reference, max_bytes=max_bytes), timeout=timeout)


_PRIVATE_HOSTS = {"localhost", "localhost.localdomain", "metadata", "metadata.google.internal"}
_PRIVATE_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")


def _is_public_ip(ip: str) -> bool:
    import ipaddress

    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    # ``is_global`` is deliberately stricter than just ``not private``.  It
    # rejects documentation/test ranges, benchmarking ranges and other
    # special-use networks which must never be reachable from a media fetcher.
    if not address.is_global or any((
        address.is_multicast, address.is_reserved, address.is_loopback,
        address.is_link_local, address.is_unspecified,
        getattr(address, "is_site_local", False),
    )):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return _is_public_ip(str(address.ipv4_mapped))
        # Transition/translation endpoints can route an otherwise global IPv6
        # literal to private IPv4. Refuse these special-use transports.
        if any(address in ipaddress.ip_network(prefix) for prefix in (
            "64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32",
        )):
            return False
    return True


def _public_resolution(url: str) -> tuple[str, int, list[dict[str, Any]]]:
    """Resolve *url* once and return a connector-pinned address list.

    The returned addresses are the exact socket endpoints that the HTTP
    connector will use.  Keeping this list separate from the URL hostname
    preserves HTTPS Host/SNI/certificate validation while removing the DNS
    rebinding window between validation and connect.
    """

    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise ValueError("只允许 http/https 媒体链接")
    if parts.username or parts.password:
        raise ValueError("媒体链接不允许携带用户凭据")
    host = (parts.hostname or "").strip().lower()
    if not host:
        raise ValueError("媒体链接缺少主机名")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("媒体链接主机名无效") from exc
    if host in _PRIVATE_HOSTS or host.endswith(_PRIVATE_SUFFIXES):
        raise ValueError("拒绝访问内网或元数据地址")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("媒体链接端口无效") from exc
    try:
        literal = __import__("ipaddress").ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and not _is_public_ip(str(literal)):
        raise ValueError("拒绝访问内网地址")
    try:
        infos = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except OSError as exc:
        raise ValueError(f"无法解析媒体主机：{exc}") from exc
    pinned: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    for family, _socktype, proto, _, sockaddr in infos:
        address = str(sockaddr[0])
        if not _is_public_ip(address):
            raise ValueError("拒绝访问内网地址")
        key = (family, address, int(sockaddr[1]))
        if key in seen:
            continue
        seen.add(key)
        pinned.append(
            {
                "hostname": host,
                "host": address,
                "port": int(sockaddr[1]),
                "family": family,
                "proto": proto or socket.IPPROTO_TCP,
                "flags": 0,
            }
        )
    if not pinned:
        raise ValueError("媒体主机没有可用公网地址")
    return host, port, pinned


async def assert_public_url(url: str) -> None:
    """Reject URLs that could reach the local network (SSRF guard).

    The host is resolved and every resulting address must be public, which also
    covers DNS rebinding to a private address. Redirects are refused at the HTTP
    layer, so a public URL cannot bounce to an internal one.
    """

    await asyncio.to_thread(_public_resolution, url)


def _peername(response: Any) -> str:
    """IP the response is actually connected to, when aiohttp exposes it."""

    try:
        connection = getattr(response, "connection", None)
        transport = getattr(connection, "transport", None)
        if transport is None:
            return ""
        peername = transport.get_extra_info("peername")
        if not peername:
            return ""
        host = str(peername[0])
        # Strip IPv6 scope/mapped prefixes so the address parses.
        if host.startswith("::ffff:"):
            host = host[len("::ffff:") :]
        return host
    except Exception:  # pragma: no cover - defensive
        return ""


class _PinnedResolver(AbstractResolver):
    """Small aiohttp resolver adapter backed by one validated DNS answer."""

    def __init__(self, host: str, pinned: list[dict[str, Any]]) -> None:
        self.host = host.rstrip(".").lower()
        self.pinned = list(pinned)

    async def resolve(
        self, requested_host: str, requested_port: int = 0, family: int = 0
    ) -> list[dict[str, Any]]:
        if requested_host.rstrip(".").lower() != self.host:
            raise OSError("请求主机与已校验主机不一致")
        matches = [item for item in self.pinned if not family or item["family"] == family]
        if not matches:
            raise OSError("已校验主机没有匹配的地址族")
        return matches

    async def close(self) -> None:
        return None


async def _download(url: str, *, max_bytes: int = 200 * 1024 * 1024) -> str:
    import aiohttp
    host, port, pinned = await asyncio.to_thread(_public_resolution, url)
    suffix = Path(url.split("?", 1)[0]).suffix[:8] or ".bin"
    handle = None

    try:
        import tempfile

        handle = tempfile.NamedTemporaryFile(
            "wb", prefix="mm-dl-", suffix=suffix, delete=False
        )
        connector = aiohttp.TCPConnector(
            resolver=_PinnedResolver(host, pinned),
            use_dns_cache=False,
            # trust_env=False prevents an unvalidated HTTP proxy from changing
            # the destination after our pinned resolution.
            ssl=True,
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=connector,
            trust_env=False,
        ) as session:
            async with session.get(url, allow_redirects=False, proxy=None) as response:
                if response.status != 200:
                    raise ValueError(f"下载失败：HTTP {response.status}")
                if response.content_length is not None and response.content_length > max_bytes:
                    raise ValueError("媒体体积超过下载上限")
                total = 0
                async for chunk in response.content.iter_chunked(65536):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("媒体体积超过下载上限")
                    handle.write(chunk)
        handle.close()
        return handle.name
    except BaseException:
        if handle is not None and not handle.closed:
            handle.close()
        if handle is not None:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
        raise

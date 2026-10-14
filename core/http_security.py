"""Shared defensive helpers for local HTTP APIs."""

from __future__ import annotations

import ipaddress
import re
import time
from urllib.parse import urlsplit


MAX_CHAT_MESSAGES = 256


class HTTPRequestError(ValueError):
    """A malformed or unsupported HTTP request body."""


class RequestBodyTooLarge(HTTPRequestError):
    """The request body exceeds the endpoint's configured size limit."""


def is_loopback_host(host: str) -> bool:
    """Treat only localhost literals and loopback IP addresses as local.

    Unknown hostnames fail closed: they require API authentication because
    their eventual network destination cannot be established from the name.
    """
    normalized = str(host or "").strip().strip("[]").rstrip(".").casefold()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def is_allowed_local_browser_origin(origin: str, port: int) -> bool:
    """Allow only HTTP origins hosted on loopback at the expected local port.

    Requests without an Origin header are non-browser/API-client requests and
    remain eligible for normal authentication. This is a CSRF defense, not a
    substitute for API-key authentication when binding beyond loopback.
    """
    value = str(origin or "").strip()
    if not value:
        return True
    try:
        parsed = urlsplit(value)
        effective_port = parsed.port if parsed.port is not None else (80 if parsed.scheme == "http" else 443)
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname is not None
        and is_loopback_host(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and effective_port == int(port)
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def read_limited_request_body(
    handler,
    max_bytes: int,
    *,
    empty_body: bytes = b"",
    read_timeout_seconds: float = 60.0,
) -> bytes:
    """Read a bounded Content-Length body with unambiguous framing and a deadline.

    BaseHTTPRequestHandler does not decode chunked bodies. Reject them rather
    than accidentally treating chunk framing as JSON or leaving unread bytes
    on a persistent connection. Reject duplicate lengths, malformed lengths,
    truncated bodies, and clients that hold a handler open while drip-feeding.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes 不能为负数。")
    if read_timeout_seconds <= 0:
        raise ValueError("read_timeout_seconds 必须大于 0。")

    headers = handler.headers
    transfer_encodings = [
        str(value).strip()
        for value in (headers.get_all("Transfer-Encoding", []) or [])
        if str(value).strip()
    ]
    if transfer_encodings:
        raise HTTPRequestError("当前 API 不支持 Transfer-Encoding 请求体。")

    lengths = headers.get_all("Content-Length", []) or []
    if len(lengths) > 1:
        raise HTTPRequestError("请求包含重复的 Content-Length。")
    if not lengths:
        return empty_body

    raw_length = str(lengths[0]).strip()
    if not re.fullmatch(r"[0-9]+", raw_length, flags=re.ASCII):
        raise HTTPRequestError("Content-Length 格式无效。")
    # Avoid expensive conversions of attacker-controlled megabyte-sized integers.
    if len(raw_length) > 20:
        raise RequestBodyTooLarge("Content-Length 数值超过安全范围。")
    length = int(raw_length, 10)
    if length > max_bytes:
        raise RequestBodyTooLarge(f"请求体超过安全上限（{max_bytes} 字节）。")
    if length == 0:
        return empty_body

    connection = getattr(handler, "connection", None)
    gettimeout = getattr(connection, "gettimeout", None)
    settimeout = getattr(connection, "settimeout", None)
    original_timeout = gettimeout() if callable(gettimeout) else None
    deadline = time.monotonic() + float(read_timeout_seconds)
    chunks: list[bytes] = []
    remaining = length
    try:
        while remaining:
            if time.monotonic() >= deadline:
                raise HTTPRequestError("接收请求体超时。")
            if callable(settimeout):
                settimeout(max(0.05, deadline - time.monotonic()))
            try:
                chunk = handler.rfile.read(min(64 * 1024, remaining))
            except (TimeoutError, OSError) as exc:
                raise HTTPRequestError("接收请求体超时或连接中断。") from exc
            if not chunk:
                raise HTTPRequestError("请求体不完整。")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        if callable(settimeout):
            try:
                settimeout(original_timeout)
            except OSError:
                pass
    return b"".join(chunks)


def require_auth_for_non_loopback_bind(
    host: str,
    api_key: str,
    *,
    api_name: str,
    api_key_env: str,
    host_env: str,
) -> None:
    """Refuse remotely reachable API binds without an explicitly configured key."""
    if str(api_key or "").strip() or is_loopback_host(host):
        return
    raise RuntimeError(
        f"拒绝在非本机地址上启动未配置 API Key 的 {api_name}；"
        f"请设置 {api_key_env}，或将 {host_env} 设为 127.0.0.1。"
    )

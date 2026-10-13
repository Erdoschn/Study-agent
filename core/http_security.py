"""Shared startup checks for local HTTP API bind security."""
from __future__ import annotations

import ipaddress


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

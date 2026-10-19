"""Local-only browser environment diagnostics.

This module writes a small report for support/debugging. It never reads browser
cookies, page storage, request headers, prompts, or site-generated identifiers.
The installation ID is an app-local correlation key and is not sent to websites.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .storage_security import ensure_private_directory


_DEFAULT_REPORT_DIRECTORY = Path("data") / "browser-environment"
_ID_FILENAME = "installation_id"
_REPORT_FILENAME = "report.json"
_MAX_ID_BYTES = 80


def _safe_label(value: Any, *, fallback: str = "unknown", limit: int = 100) -> str:
    text = "".join(char for char in str(value or "") if char.isprintable()).strip()
    return text[:limit] if text else fallback


def _profile_label(path_value: str | Path) -> str:
    # Normalize both Windows and POSIX separators without returning the full path.
    raw = str(path_value or "").rstrip("/\\").replace("\\", "/")
    label = raw.split("/")[-1] if raw else ""
    return _safe_label(label, fallback="browser-profile", limit=80)


def _validate_private_file(path: Path, *, max_bytes: int) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PermissionError("无法检查浏览器环境诊断文件。") from exc

    if stat.S_ISLNK(status.st_mode):
        raise PermissionError("浏览器环境诊断文件禁止使用符号链接。")
    if os.name == "nt" and getattr(status, "st_file_attributes", 0) & 0x400:
        raise PermissionError("浏览器环境诊断文件禁止使用 Windows reparse point。")
    if not stat.S_ISREG(status.st_mode) or status.st_size > max_bytes:
        raise PermissionError("浏览器环境诊断路径不是有效的普通文件。")
    if getattr(status, "st_nlink", 1) > 1:
        raise PermissionError("浏览器环境诊断文件禁止使用硬链接。")
    return True


def _read_installation_id(path: Path) -> str | None:
    if not _validate_private_file(path, max_bytes=_MAX_ID_BYTES):
        return None
    try:
        raw = path.read_text(encoding="ascii").strip()
        parsed = uuid.UUID(raw)
    except (OSError, UnicodeError, ValueError, AttributeError) as exc:
        raise PermissionError("浏览器环境 installation ID 文件无效。") from exc
    return str(parsed)


def _get_or_create_installation_id(path: Path) -> str:
    existing = _read_installation_id(path)
    if existing is not None:
        return existing

    identifier = str(uuid.uuid4())
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        # Another process may have won the create race; validate its file.
        existing = _read_installation_id(path)
        if existing is None:
            raise PermissionError("无法安全创建浏览器环境 installation ID。")
        return existing
    except OSError as exc:
        raise PermissionError("无法安全创建浏览器环境 installation ID。") from exc

    try:
        payload = (identifier + "\n").encode("ascii")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PermissionError("无法写入浏览器环境 installation ID。") from exc
    finally:
        os.close(descriptor)

    if os.name != "nt":
        os.chmod(path, 0o600)
    return identifier


def _atomic_write_report(path: Path, report: dict[str, Any]) -> None:
    _validate_private_file(path, max_bytes=2 * 1024 * 1024)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        payload = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        if os.name != "nt":
            os.chmod(path, 0o600)
    except OSError as exc:
        raise PermissionError("无法安全保存浏览器环境诊断报告。") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _package_version(name: str) -> str | None:
    try:
        return _safe_label(importlib.metadata.version(name), limit=80)
    except importlib.metadata.PackageNotFoundError:
        return None


def _target_origin(value: str) -> str | None:
    try:
        parts = urlsplit(str(value or ""))
        hostname = parts.hostname
        if not parts.scheme or not hostname:
            return None
        host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        port = parts.port
        authority = f"{host}:{port}" if port is not None else host
        return f"{parts.scheme.lower()}://{authority}"
    except ValueError:
        return None


def _signature(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def write_browser_environment_report(
    *,
    user_data_dir: str | Path,
    browser_channel: str | None,
    target_url: str,
    browser_version: str | None = None,
    chat_input_ready: bool | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist sanitized, local-only browser/runtime metadata and return the report."""
    if chat_input_ready is not None and not isinstance(chat_input_ready, bool):
        raise ValueError("chat_input_ready must be a boolean or None")
    root = Path(project_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    report_directory = ensure_private_directory(root / _DEFAULT_REPORT_DIRECTORY)
    installation_id = _get_or_create_installation_id(report_directory / _ID_FILENAME)

    profile_label = _profile_label(user_data_dir)
    channel = _safe_label(browser_channel, fallback="default", limit=80)
    actual_browser_version = _safe_label(browser_version, fallback="", limit=80) or None
    runtime = {
        "os": _safe_label(platform.system(), fallback="unknown", limit=80),
        "os_release": _safe_label(platform.release(), fallback="unknown", limit=80),
        "architecture": _safe_label(platform.machine(), fallback="unknown", limit=80),
        "python": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
        "playwright": _package_version("playwright"),
    }
    browser = {"channel": channel, "version": actual_browser_version}
    settings = {
        "browser_channel": channel,
        "profile_label": profile_label,
        "headless": False,
        "persistent_profile": True,
    }
    settings_signature = _signature(settings)
    environment_signature = _signature({
        "settings": settings,
        "browser": browser,
        "runtime": runtime,
    })

    profile_path = Path(user_data_dir).expanduser()
    try:
        profile_exists = profile_path.is_dir()
        profile_writable = profile_exists and os.access(profile_path, os.W_OK)
    except OSError:
        profile_exists = False
        profile_writable = False

    report: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "installation_id": installation_id,
        "settings_signature": settings_signature,
        "environment_signature": environment_signature,
        "browser": browser,
        "profile": {
            "label": profile_label,
            "exists": bool(profile_exists),
            "writable": bool(profile_writable),
        },
        "runtime": runtime,
        "target_origin": _target_origin(target_url),
        "chat_input_ready": chat_input_ready,
    }
    _atomic_write_report(report_directory / _REPORT_FILENAME, report)
    return report

"""Canonical user-profile identifiers and isolated persistent storage paths.

A username is a profile selector, not an authentication credential. Networked
multi-user deployments must bind this identifier to an authenticated identity.
"""
from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from pathlib import Path

from .storage_security import ensure_private_directory

DEFAULT_USERNAME = "local"
MAX_USERNAME_LENGTH = 64
DEFAULT_MAX_USER_PROFILES = 256


class UserProfileLimitError(ValueError):
    """The server's configured limit for persistent user profiles was reached."""


def normalize_username(value: object) -> str:
    """Return a stable case-insensitive username safe for profile lookup."""
    if not isinstance(value, str):
        raise ValueError("username 必须是字符串。")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise ValueError("username 不能为空。")
    if len(normalized) > MAX_USERNAME_LENGTH:
        raise ValueError(f"username 不能超过 {MAX_USERNAME_LENGTH} 个字符。")
    if normalized in {".", ".."}:
        raise ValueError("username 格式无效。")
    if any(not (char.isalnum() or char in "._-") for char in normalized):
        raise ValueError("username 只允许字母、数字、点、下划线和连字符。")
    return normalized.casefold()


def _profile_limit(value: int | None) -> int:
    if value is None:
        try:
            value = int(os.getenv("STUDY_AGENT_MAX_USER_PROFILES", str(DEFAULT_MAX_USER_PROFILES)))
        except (TypeError, ValueError):
            value = DEFAULT_MAX_USER_PROFILES
    if not 1 <= int(value) <= 100_000:
        raise ValueError("最大用户 Profile 数必须在 1 到 100000 之间。")
    return int(value)


def user_graph_path(
    username: object,
    *,
    data_root: str | Path,
    max_profiles: int | None = None,
) -> Path:
    """Return a per-user SQLite graph path using a non-reversible directory key.

    User input is never interpolated into a filesystem path. The directory
    creation and SQLite file are guarded by storage_security.
    """
    key = normalize_username(username)
    limit = _profile_limit(max_profiles)
    users_root = ensure_private_directory(Path(data_root).expanduser() / "users")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    profile_dir = users_root / digest

    try:
        status = profile_dir.lstat()
    except FileNotFoundError:
        status = None

    if status is None:
        existing_profiles = 0
        for child in users_root.iterdir():
            try:
                child_status = child.lstat()
            except OSError:
                continue
            if stat.S_ISDIR(child_status.st_mode) and not stat.S_ISLNK(child_status.st_mode):
                # Only count profile directories created by this implementation.
                if len(child.name) == 64 and all(ch in "0123456789abcdef" for ch in child.name):
                    existing_profiles += 1
        if existing_profiles >= limit:
            raise UserProfileLimitError(
                f"已达到用户 Profile 数量上限（{limit}）；请调整 STUDY_AGENT_MAX_USER_PROFILES。"
            )

    private_profile = ensure_private_directory(profile_dir)
    return private_profile / "knowledge_graph.sqlite3"

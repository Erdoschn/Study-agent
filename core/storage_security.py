"""Filesystem guards shared by persistent SQLite stores."""

from __future__ import annotations

import os
import stat
from pathlib import Path


MAX_SQLITE_FILE_BYTES = 256 * 1024 * 1024
_SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


def _validate_regular_file(path: Path, *, max_bytes: int) -> bool:
    """Return whether a file exists, rejecting links, special files and oversize files."""
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PermissionError(f"无法检查 SQLite 文件：{path.name}") from exc

    if stat.S_ISLNK(status.st_mode):
        raise PermissionError(f"SQLite 文件禁止使用符号链接：{path.name}")
    if not stat.S_ISREG(status.st_mode):
        raise PermissionError(f"SQLite 路径必须是普通文件：{path.name}")
    if os.name == "nt" and getattr(status, "st_file_attributes", 0) & 0x400:
        raise PermissionError(f"SQLite 文件禁止使用 Windows reparse point：{path.name}")
    if status.st_size > max_bytes:
        raise PermissionError(f"SQLite 文件超过 {max_bytes} 字节安全上限：{path.name}")
    return True


def ensure_safe_sqlite_path(path: str | Path) -> Path:
    """Create a private DB file atomically when absent and reject unsafe DB/sidecar paths.

    SQLite otherwise follows a pre-existing symlink at its database path. WAL and
    rollback-journal sidecars are checked too, because they are written alongside
    the database file. This is defense in depth; it does not replace OS access control.
    """
    db_path = Path(path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not _validate_regular_file(db_path, max_bytes=MAX_SQLITE_FILE_BYTES):
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(db_path, flags, 0o600)
        except FileExistsError:
            # Another writer created the database first; validate its actual file type.
            _validate_regular_file(db_path, max_bytes=MAX_SQLITE_FILE_BYTES)
        except OSError as exc:
            raise PermissionError(f"无法安全创建 SQLite 文件：{db_path.name}") from exc
        else:
            os.close(descriptor)

    _validate_regular_file(db_path, max_bytes=MAX_SQLITE_FILE_BYTES)
    if os.name != "nt":
        try:
            os.chmod(db_path, 0o600)
        except OSError as exc:
            raise PermissionError(f"无法限制 SQLite 文件权限：{db_path.name}") from exc

    for suffix in _SQLITE_SIDECARS:
        _validate_regular_file(
            Path(str(db_path) + suffix),
            max_bytes=MAX_SQLITE_FILE_BYTES,
        )
    return db_path

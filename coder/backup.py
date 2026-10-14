from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
import uuid
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .filesystem import WorkspaceFS, WorkspaceSecurityError


@dataclass(frozen=True)
class BackupSnapshot:
    generation: int
    archive: Path
    file_count: int


class CoderBackupStore:
    """Stores one immutable initial snapshot and one replaceable last-known-good snapshot."""

    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_LOCKS_GUARD = threading.Lock()

    INITIAL_ARCHIVE_NAME = "initial.zip"
    ARCHIVE_NAME = "latest.zip"
    MANIFEST_NAME = "backup-manifest.json"
    EDITABLE_SUFFIXES = frozenset({".py", ".pyi", ".ipynb"})
    MAX_ARCHIVE_FILES = 10_000
    MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
    MAX_ARCHIVE_BYTES = 768 * 1024 * 1024
    MAX_MANIFEST_BYTES = 16 * 1024 * 1024
    SKIP_DIRS = frozenset({
        ".git", ".venv", "__pycache__", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", ".idea",
    })

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        backup_path = self.workspace / ".coder-backup"

        # Inspect the lexical path before resolving it. Resolving first would
        # hide a symlink that points to another directory inside the workspace.
        try:
            status = backup_path.lstat()
        except FileNotFoundError:
            try:
                backup_path.mkdir(mode=0o700)
                status = backup_path.lstat()
            except OSError as exc:
                raise WorkspaceSecurityError("无法创建 Coder backup 目录。") from exc
        except OSError as exc:
            raise WorkspaceSecurityError("无法检查 Coder backup 目录。") from exc

        if not os.path.isdir(backup_path) or backup_path.is_symlink():
            raise WorkspaceSecurityError("备份目录必须是普通目录，禁止使用符号链接。")
        if os.name == "nt" and getattr(status, "st_file_attributes", 0) & 0x400:
            raise WorkspaceSecurityError("备份目录禁止使用 Windows reparse point。")

        self.root = backup_path.resolve()
        try:
            self.root.relative_to(self.workspace)
        except ValueError as exc:
            raise WorkspaceSecurityError("Coder backup 必须位于 workspace 内。") from exc

        if self.root.parent != self.workspace or self.root.name != ".coder-backup":
            raise WorkspaceSecurityError("Coder backup 目录解析结果不符合预期。")
        if os.name != "nt":
            try:
                os.chmod(self.root, 0o700)
            except OSError as exc:
                raise WorkspaceSecurityError("无法限制 Coder backup 目录权限。") from exc

        with self._ROOT_LOCKS_GUARD:
            self._snapshot_lock = self._ROOT_LOCKS.setdefault(
                str(self.root).casefold(), threading.RLock()
            )

    @classmethod
    def _iter_code_files(cls, workspace: Path) -> Iterable[tuple[Path, Path]]:
        for current, dirs, files in os.walk(workspace, topdown=True, followlinks=False):
            current_path = Path(current)
            dirs[:] = [
                name for name in dirs
                if name.casefold() not in cls.SKIP_DIRS
                and not name.casefold().startswith(".coder-")
            ]
            for name in files:
                source = current_path / name
                try:
                    if source.is_symlink() or source.suffix.casefold() not in cls.EDITABLE_SUFFIXES:
                        continue
                    relative = source.relative_to(workspace)
                    if any(part.casefold().startswith(".coder-") for part in relative.parts):
                        continue
                    if source.stat().st_size > WorkspaceFS.MAX_FILE_BYTES:
                        continue
                except (FileNotFoundError, OSError):
                    continue
                yield source, relative

    @staticmethod
    def _read_regular_source(source: Path) -> bytes | None:
        """Read one bounded regular file without following a leaf link where supported."""
        descriptor: int | None = None
        try:
            before = source.lstat()
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_size > WorkspaceFS.MAX_FILE_BYTES
            ):
                return None
            if os.name == "nt" and getattr(before, "st_file_attributes", 0) & 0x400:
                return None

            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(source, flags)
            stream = os.fdopen(descriptor, "rb")
            descriptor = None  # the file object now owns the descriptor
            with stream:
                opened = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_size > WorkspaceFS.MAX_FILE_BYTES
                    or (
                        os.name == "nt"
                        and getattr(opened, "st_file_attributes", 0) & 0x400
                    )
                ):
                    return None
                data = stream.read(WorkspaceFS.MAX_FILE_BYTES + 1)
            return data if len(data) <= WorkspaceFS.MAX_FILE_BYTES else None
        except OSError:
            return None
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _build_archive(self, archive_path: Path, generation: int) -> int:
        entries: list[dict[str, object]] = []
        total_bytes = 0
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for source, relative in self._iter_code_files(self.workspace):
                if len(entries) >= self.MAX_ARCHIVE_FILES:
                    raise WorkspaceSecurityError("备份文件数量超过安全上限。")
                data = self._read_regular_source(source)
                if data is None:
                    continue

                total_bytes += len(data)
                if total_bytes > self.MAX_ARCHIVE_TOTAL_BYTES:
                    raise WorkspaceSecurityError("备份内容总大小超过 512 MiB 安全上限。")

                arcname = relative.as_posix()
                archive.writestr(arcname, data)
                entries.append({
                    "path": arcname,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                })

            entries.sort(key=lambda item: str(item["path"]))
            archive.writestr(
                self.MANIFEST_NAME,
                json.dumps(
                    {
                        "generation": int(generation),
                        "file_count": len(entries),
                        "files": entries,
                    },
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
            )
        return len(entries)

    def _assert_safe_archive_file(self, archive: Path) -> bool:
        """Reject archive symlinks/special files before any manifest or entry read."""
        try:
            status = archive.lstat()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise WorkspaceSecurityError("无法检查 Coder backup archive。") from exc
        if stat.S_ISLNK(status.st_mode):
            raise WorkspaceSecurityError("备份 archive 禁止使用符号链接。")
        if not stat.S_ISREG(status.st_mode):
            raise WorkspaceSecurityError("备份 archive 必须是普通文件。")
        if os.name == "nt" and getattr(status, "st_file_attributes", 0) & 0x400:
            raise WorkspaceSecurityError("备份 archive 禁止使用 Windows reparse point。")
        if status.st_size > self.MAX_ARCHIVE_BYTES:
            raise WorkspaceSecurityError("备份 archive 超过 768 MiB 安全上限。")
        return True

    def _read_archive_manifest(self, archive: Path) -> dict:
        if not self._assert_safe_archive_file(archive):
            return {}
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                info = zf.getinfo(self.MANIFEST_NAME)
                if info.file_size > self.MAX_MANIFEST_BYTES:
                    raise RuntimeError("Coder backup manifest 超过 16 MiB 安全上限。")
                data = json.loads(zf.read(info).decode("utf-8"))
        except RuntimeError:
            raise
        except (OSError, KeyError, ValueError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Coder backup archive 损坏：{archive.name}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("files", []), list):
            raise RuntimeError(f"Coder backup manifest 格式非法：{archive.name}")
        if len(data.get("files", [])) > self.MAX_ARCHIVE_FILES:
            raise RuntimeError("Coder backup manifest 文件条目超过安全上限。")
        return data

    def _snapshot(self, archive: Path, generation: int, *, immutable: bool) -> BackupSnapshot:
        generation = int(generation)
        if generation < 0:
            raise ValueError("backup generation 不能为负数。")

        # Serialize snapshots across store instances targeting the same root.
        # Otherwise an older generation that finishes late can overwrite a
        # newer last-known-good archive even when each individual replace is
        # atomic.
        with self._snapshot_lock:
            if self._assert_safe_archive_file(archive):
                current = self._read_archive_manifest(archive)
                current_generation = int(current.get("generation", -1))
                if immutable or current_generation > generation:
                    return BackupSnapshot(
                        generation=current_generation,
                        archive=archive,
                        file_count=int(current.get("file_count", 0)),
                    )

            archive_tmp = self.root / f".{archive.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                file_count = self._build_archive(archive_tmp, generation)
                if immutable and self._assert_safe_archive_file(archive):
                    try:
                        archive_tmp.unlink()
                    except FileNotFoundError:
                        pass
                    return self._snapshot(archive, generation, immutable=True)
                os.replace(archive_tmp, archive)
                return BackupSnapshot(
                    generation=generation,
                    archive=archive,
                    file_count=file_count,
                )
            except Exception:
                try:
                    archive_tmp.unlink()
                except FileNotFoundError:
                    pass
                raise

    def ensure_initial_snapshot(self, generation: int = 0) -> BackupSnapshot:
        return self._snapshot(
            self.root / self.INITIAL_ARCHIVE_NAME,
            generation,
            immutable=True,
        )

    def snapshot(self, generation: int) -> BackupSnapshot:
        return self._snapshot(
            self.root / self.ARCHIVE_NAME,
            generation,
            immutable=False,
        )

    def read_manifest(self, *, initial: bool = False) -> dict:
        archive = self.root / (
            self.INITIAL_ARCHIVE_NAME if initial else self.ARCHIVE_NAME
        )
        return self._read_archive_manifest(archive)

    def contains_text(self, relative_path: str, expected: str, *, initial: bool = False) -> bool:
        archive = self.root / (
            self.INITIAL_ARCHIVE_NAME if initial else self.ARCHIVE_NAME
        )
        manifest = self.read_manifest(initial=initial)
        allowed = {
            str(item.get("path"))
            for item in manifest.get("files", [])
            if isinstance(item, dict)
        }
        normalized = relative_path.replace("\\", "/")
        if normalized not in allowed or not archive.is_file():
            return False
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                actual = zf.read(normalized).decode("utf-8")
        except (KeyError, UnicodeDecodeError, zipfile.BadZipFile):
            return False
        return actual == expected

    def has_initial_snapshot(self) -> bool:
        archive = self.root / self.INITIAL_ARCHIVE_NAME
        return archive.is_file() and bool(self._read_archive_manifest(archive))

    def has_latest_snapshot(self) -> bool:
        archive = self.root / self.ARCHIVE_NAME
        return archive.is_file() and bool(self._read_archive_manifest(archive))

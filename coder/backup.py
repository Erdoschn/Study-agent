from __future__ import annotations

import hashlib
import json
import os
import zipfile
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

    INITIAL_ARCHIVE_NAME = "initial.zip"
    ARCHIVE_NAME = "latest.zip"
    MANIFEST_NAME = "backup-manifest.json"
    EDITABLE_SUFFIXES = frozenset({".py", ".pyi"})
    SKIP_DIRS = frozenset({
        ".git", ".venv", "__pycache__", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", ".idea",
    })

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.root = (self.workspace.parent / f"{self.workspace.name}.coder-backup").resolve()
        try:
            self.root.relative_to(self.workspace)
        except ValueError:
            pass
        else:
            raise WorkspaceSecurityError("Coder backup 必须位于 workspace 同级或其外部。")

        self.root.mkdir(parents=True, exist_ok=True)
        try:
            status = self.root.lstat()
        except OSError as exc:
            raise WorkspaceSecurityError("无法检查 Coder backup 目录。") from exc
        if status.st_mode & 0o170000 == 0o120000:
            raise WorkspaceSecurityError("备份目录禁止使用符号链接。")
        if os.name == "nt" and getattr(status, "st_file_attributes", 0) & 0x400:
            raise WorkspaceSecurityError("备份目录禁止使用 Windows reparse point。")

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

    def _build_archive(self, archive_path: Path, generation: int) -> int:
        entries: list[dict[str, object]] = []
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for source, relative in self._iter_code_files(self.workspace):
                arcname = relative.as_posix()
                data = source.read_bytes()
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

    def _read_archive_manifest(self, archive: Path) -> dict:
        if not archive.is_file():
            return {}
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                data = json.loads(zf.read(self.MANIFEST_NAME).decode("utf-8"))
        except (OSError, KeyError, ValueError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Coder backup archive 损坏：{archive.name}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Coder backup manifest 格式非法：{archive.name}")
        return data

    def _snapshot(self, archive: Path, generation: int, *, immutable: bool) -> BackupSnapshot:
        generation = int(generation)
        if generation < 0:
            raise ValueError("backup generation 不能为负数。")
        if immutable and archive.exists():
            data = self._read_archive_manifest(archive)
            return BackupSnapshot(
                generation=int(data.get("generation", generation)),
                archive=archive,
                file_count=int(data.get("file_count", 0)),
            )

        archive_tmp = self.root / f".{archive.name}.{os.getpid()}.tmp"
        try:
            file_count = self._build_archive(archive_tmp, generation)
            if immutable and archive.exists():
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

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
    """Stores the immutable initial snapshot and the latest known-good snapshot."""

    INITIAL_ARCHIVE_NAME = "initial.zip"
    INITIAL_MANIFEST_NAME = "initial_manifest.json"
    ARCHIVE_NAME = "latest.zip"
    MANIFEST_NAME = "manifest.json"

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

    def _build_snapshot_files(self, archive_path: Path, generation: int) -> int:
        entries: list[dict[str, object]] = []
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for source, relative in self._iter_code_files(self.workspace):
                arcname = relative.as_posix()
                archive.write(source, arcname=arcname)
                data = source.read_bytes()
                entries.append({
                    "path": arcname,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                })

        entries.sort(key=lambda item: str(item["path"]))
        return len(entries)

    def _write_manifest(self, path: Path, generation: int, file_count: int) -> None:
        entries: list[dict[str, object]] = []
        archive = self.root / (
            self.INITIAL_ARCHIVE_NAME
            if path.name == self.INITIAL_MANIFEST_NAME
            else self.ARCHIVE_NAME
        )
        with zipfile.ZipFile(archive, "r") as zf:
            for name in sorted(zf.namelist()):
                info = zf.getinfo(name)
                if name.endswith("/"):
                    continue
                data = zf.read(name)
                entries.append({
                    "path": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": info.file_size,
                })
        path.write_text(
            json.dumps(
                {"generation": generation, "file_count": file_count, "files": entries},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
            newline="",
        )

    def ensure_initial_snapshot(self, generation: int = 0) -> BackupSnapshot:
        archive = self.root / self.INITIAL_ARCHIVE_NAME
        manifest = self.root / self.INITIAL_MANIFEST_NAME
        if archive.exists() or manifest.exists():
            if not archive.is_file() or not manifest.is_file():
                raise RuntimeError("Coder initial backup 不完整，已拒绝继续。")
            data = self._read_manifest(manifest)
            return BackupSnapshot(
                generation=int(data.get("generation", generation)),
                archive=archive,
                file_count=int(data.get("file_count", 0)),
            )

        archive_tmp = self.root / f"{self.INITIAL_ARCHIVE_NAME}.{os.getpid()}.tmp"
        manifest_tmp = self.root / f"{self.INITIAL_MANIFEST_NAME}.{os.getpid()}.tmp"
        try:
            file_count = self._build_snapshot_files(archive_tmp, generation)
            entries = []
            with zipfile.ZipFile(archive_tmp, "r") as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith("/"):
                        continue
                    info = zf.getinfo(name)
                    data = zf.read(name)
                    entries.append({
                        "path": name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": info.file_size,
                    })
            manifest_tmp.write_text(
                json.dumps(
                    {"generation": generation, "file_count": file_count, "files": entries},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
                newline="",
            )
            os.replace(archive_tmp, archive)
            os.replace(manifest_tmp, manifest)
            return BackupSnapshot(
                generation=generation,
                archive=archive,
                file_count=file_count,
            )
        except Exception:
            for path in (archive_tmp, manifest_tmp):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def snapshot(self, generation: int) -> BackupSnapshot:
        generation = int(generation)
        if generation < 0:
            raise ValueError("backup generation 不能为负数。")

        archive = self.root / self.ARCHIVE_NAME
        manifest = self.root / self.MANIFEST_NAME
        archive_tmp = self.root / f"{self.ARCHIVE_NAME}.{os.getpid()}.tmp"
        manifest_tmp = self.root / f"{self.MANIFEST_NAME}.{os.getpid()}.tmp"
        try:
            file_count = self._build_snapshot_files(archive_tmp, generation)
            entries = []
            with zipfile.ZipFile(archive_tmp, "r") as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith("/"):
                        continue
                    info = zf.getinfo(name)
                    data = zf.read(name)
                    entries.append({
                        "path": name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": info.file_size,
                    })
            manifest_tmp.write_text(
                json.dumps(
                    {"generation": generation, "file_count": file_count, "files": entries},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
                newline="",
            )
            os.replace(archive_tmp, archive)
            os.replace(manifest_tmp, manifest)
            return BackupSnapshot(
                generation=generation,
                archive=archive,
                file_count=file_count,
            )
        except Exception:
            for path in (archive_tmp, manifest_tmp):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise

    @staticmethod
    def _read_manifest(path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Coder backup manifest 损坏：{path.name}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Coder backup manifest 格式非法：{path.name}")
        return data

    def read_manifest(self, *, initial: bool = False) -> dict:
        return self._read_manifest(
            self.root / (
                self.INITIAL_MANIFEST_NAME if initial else self.MANIFEST_NAME
            )
        ) if (
            self.root / (
                self.INITIAL_MANIFEST_NAME if initial else self.MANIFEST_NAME
            )
        ).is_file() else {}

    def contains_text(self, relative_path: str, expected: str, *, initial: bool = False) -> bool:
        manifest = self.read_manifest(initial=initial)
        allowed = {
            str(item.get("path"))
            for item in manifest.get("files", [])
            if isinstance(item, dict)
        }
        normalized = relative_path.replace("\\", "/")
        if normalized not in allowed:
            return False
        archive = self.root / (
            self.INITIAL_ARCHIVE_NAME if initial else self.ARCHIVE_NAME
        )
        if not archive.is_file():
            return False
        with zipfile.ZipFile(archive, "r") as zf:
            try:
                actual = zf.read(normalized).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                return False
        return actual == expected

    def has_initial_snapshot(self) -> bool:
        return (
            (self.root / self.INITIAL_ARCHIVE_NAME).is_file()
            and (self.root / self.INITIAL_MANIFEST_NAME).is_file()
        )

    def has_latest_snapshot(self) -> bool:
        return (
            (self.root / self.ARCHIVE_NAME).is_file()
            and (self.root / self.MANIFEST_NAME).is_file()
        )

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
    """Keeps one last-known-good Python snapshot outside the model workspace."""

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
        if self.root == self.workspace.parent.parent:
            raise WorkspaceSecurityError("Coder backup 路径非法。")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise WorkspaceSecurityError("备份目录禁止使用符号链接。")

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

    def snapshot(self, generation: int) -> BackupSnapshot:
        generation = int(generation)
        if generation < 0:
            raise ValueError("backup generation 不能为负数。")

        stage_dir = Path(
            tempfile.mkdtemp(prefix=".coder-backup-", dir=str(self.root.parent))
        )
        archive_tmp = self.root / f"{self.ARCHIVE_NAME}.{os.getpid()}.tmp"
        manifest_tmp = self.root / f"{self.MANIFEST_NAME}.{os.getpid()}.tmp"
        try:
            entries: list[dict[str, object]] = []
            with zipfile.ZipFile(
                archive_tmp,
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
            manifest = {
                "generation": generation,
                "file_count": len(entries),
                "files": entries,
            }
            manifest_tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
                newline="",
            )

            os.replace(archive_tmp, self.root / self.ARCHIVE_NAME)
            os.replace(manifest_tmp, self.root / self.MANIFEST_NAME)

            return BackupSnapshot(
                generation=generation,
                archive=self.root / self.ARCHIVE_NAME,
                file_count=len(entries),
            )
        except Exception:
            for path in (archive_tmp, manifest_tmp):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise
        finally:
            try:
                stage_dir.rmdir()
            except OSError:
                pass

    def read_manifest(self) -> dict:
        path = self.root / self.MANIFEST_NAME
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("Coder backup manifest 损坏。") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Coder backup manifest 格式非法。")
        return data

    def contains_text(self, relative_path: str, expected: str) -> bool:
        manifest = self.read_manifest()
        allowed = {
            str(item.get("path"))
            for item in manifest.get("files", [])
            if isinstance(item, dict)
        }
        if relative_path.replace("\\", "/") not in allowed:
            return False
        archive = self.root / self.ARCHIVE_NAME
        if not archive.is_file():
            return False
        with zipfile.ZipFile(archive, "r") as zf:
            try:
                actual = zf.read(relative_path.replace("\\", "/")).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                return False
        return actual == expected

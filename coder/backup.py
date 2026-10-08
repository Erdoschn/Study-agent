from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .filesystem import WorkspaceFS, WorkspaceSecurityError


@dataclass(frozen=True)
class BackupSnapshot:
    generation: int
    root: Path
    file_count: int


class CoderBackupStore:
    """Stores the last known-good Python generation outside the model workspace."""

    SNAPSHOT_DIR = "snapshot"
    MANIFEST_NAME = "manifest.json"
    EDITABLE_SUFFIXES = frozenset({".py", ".pyi"})

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace.parent / f"{self.workspace.name}.coder-backup"
        self.root = self.root.resolve()
        try:
            self.root.relative_to(self.workspace)
        except ValueError:
            pass
        else:
            raise WorkspaceSecurityError("Coder backup 必须位于 workspace 同级或其外部。")
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_no_reparse(self.root)

    @staticmethod
    def _ensure_no_reparse(path: Path) -> None:
        if path.is_symlink():
            raise WorkspaceSecurityError(f"备份路径禁止使用符号链接：{path}")

    @classmethod
    def _iter_code_files(cls, workspace: Path) -> Iterable[tuple[Path, Path]]:
        for current, dirs, files in os.walk(workspace, topdown=True, followlinks=False):
            current_path = Path(current)
            dirs[:] = [
                name for name in dirs
                if name.casefold() not in {
                    ".git", ".venv", "__pycache__", ".pytest_cache",
                    ".mypy_cache", ".ruff_cache", ".idea",
                }
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

        stage = Path(tempfile.mkdtemp(prefix=".coder-backup-stage-", dir=str(self.root.parent)))
        try:
            stage_snapshot = stage / self.SNAPSHOT_DIR
            stage_snapshot.mkdir(parents=True, exist_ok=True)
            entries = []
            for source, relative in self._iter_code_files(self.workspace):
                destination = stage_snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                entries.append(relative.as_posix())

            entries.sort()
            manifest = {
                "generation": generation,
                "file_count": len(entries),
                "files": entries,
            }
            (stage / self.MANIFEST_NAME).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
                newline="",
            )

            published = self.root / self.SNAPSHOT_DIR
            old = self.root / f"{self.SNAPSHOT_DIR}.old"
            old_manifest = self.root / f"{self.MANIFEST_NAME}.old"
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            if old_manifest.exists():
                old_manifest.unlink()

            if published.exists():
                os.replace(published, old)

            staged_manifest = stage / self.MANIFEST_NAME
            final_manifest = self.root / self.MANIFEST_NAME
            if final_manifest.exists():
                os.replace(final_manifest, old_manifest)

            os.replace(stage_snapshot, published)
            os.replace(staged_manifest, final_manifest)

            shutil.rmtree(old, ignore_errors=True)
            if old_manifest.exists():
                old_manifest.unlink()

            return BackupSnapshot(
                generation=generation,
                root=published,
                file_count=len(entries),
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)

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

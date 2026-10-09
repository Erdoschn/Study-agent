import json
import ntpath
import os
import stat
import uuid
import re
import threading
from pathlib import Path


class WorkspaceSecurityError(PermissionError):
    pass


class WorkspaceFS:
    """Fail-closed filesystem boundary for the fixed Coder workspace."""

    EDITABLE_EXTENSIONS = frozenset({".py", ".pyi", ".ipynb"})
    READABLE_EXTENSIONS = frozenset({
        ".py", ".pyi", ".txt", ".md", ".rst", ".json", ".toml",
        ".ini", ".cfg", ".yaml", ".yml", ".csv", ".tsv", ".xml", ".ipynb",
    })
    BLOCKED_NAMES = frozenset({
        ".env", ".env.local", ".env.production", ".git-credentials",
        "id_rsa", "id_ed25519",
    })
    BLOCKED_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".kdbx"})
    BLOCKED_PREFIXES = (".coder-sandbox-", ".coder-backup")
    # Keep a generous hard safety bound, while READ_FILE can stream a huge
    # source file to the model in line-addressed chunks.
    MAX_FILE_BYTES = 16 * 1024 * 1024
    WINDOWS_DEVICE_NAMES = frozenset(
        {"con", "prn", "aux", "nul"}
        | {f"com{i}" for i in range(1, 10)}
        | {f"lpt{i}" for i in range(1, 10)}
    )

    def __init__(self, root: str | Path):
        raw_root = str(root)
        # ntpath.isabs() also treats POSIX paths such as /tmp/workspace as
        # absolute. On non-Windows hosts, reject Windows drive/UNC paths while
        # still allowing the temporary POSIX workspaces used by tests and CI.
        if (
            os.name != "nt"
            and ntpath.isabs(raw_root)
            and not Path(raw_root).expanduser().is_absolute()
        ):
            raise WorkspaceSecurityError("Windows 风格的固定 Coder 工作空间只允许在 Windows 主机上使用。")
        self.root = Path(root).expanduser()
        if not self.root.is_absolute():
            self.root = self.root.resolve()
        if self.root.exists():
            self._reject_reparse(self.root)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        # Serialize writes through one workspace instance. Windows can return
        # PermissionError when concurrent os.replace calls target the same file.
        self._write_lock = threading.RLock()
        if self.root == Path(self.root.anchor):
            raise WorkspaceSecurityError("禁止把文件系统根目录作为 Coder workspace。")

    @staticmethod
    def _reject_reparse(path: Path) -> None:
        try:
            st = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(st.st_mode):
            raise WorkspaceSecurityError(f"禁止访问符号链接：{path}")
        if os.name == "nt" and getattr(st, "st_file_attributes", 0) & 0x400:
            raise WorkspaceSecurityError(f"禁止访问 Windows reparse point：{path}")

    def _relative(self, value: str, *, allow_empty: bool = False) -> Path:
        text = str(value or "").strip()
        if not text:
            if allow_empty:
                return Path(".")
            raise WorkspaceSecurityError("路径不能为空。")
        if "\0" in text or ":" in text:
            raise WorkspaceSecurityError("路径包含禁止字符。")
        if ntpath.isabs(text):
            raise WorkspaceSecurityError("禁止绝对路径访问 workspace 之外的文件。")
        parts = [p for p in text.replace("\\", "/").split("/") if p not in {"", "."}]
        if (
            not parts
            or any(p == ".." for p in parts)
            or any(p.startswith("-") for p in parts)
            or any(p.endswith((" ", ".")) for p in parts)
        ):
            raise WorkspaceSecurityError("路径包含禁止形式。")
        if os.name == "nt":
            for part in parts:
                stem = re.split(r"[.]", part, maxsplit=1)[0].casefold()
                if stem in self.WINDOWS_DEVICE_NAMES:
                    raise WorkspaceSecurityError("禁止访问 Windows 设备名。")
        return Path(*parts)

    def _target(self, value: str, *, allow_empty: bool = False) -> tuple[Path, Path]:
        rel = self._relative(value, allow_empty=allow_empty)
        target = self.root / rel
        current = self.root
        for part in rel.parts:
            current = current / part
            self._reject_reparse(current)
        resolved = target.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceSecurityError("解析后的路径越出 Coder workspace。") from exc
        return rel, target

    def _policy(self, rel: Path, *, write: bool = False, test: bool = False) -> None:
        name = rel.name.casefold()
        suffix = rel.suffix.casefold()
        if (
            name in self.BLOCKED_NAMES
            or suffix in self.BLOCKED_SUFFIXES
            or any(
                part.casefold().startswith(prefix)
                for part in rel.parts
                for prefix in self.BLOCKED_PREFIXES
            )
        ):
            raise WorkspaceSecurityError(f"禁止访问敏感文件：{rel}")
        allowed = self.EDITABLE_EXTENSIONS if write else self.READABLE_EXTENSIONS
        if suffix not in allowed:
            raise WorkspaceSecurityError(f"Coder 当前只允许 Python/安全文本文件：{rel}")
        if test and (not rel.parts or rel.parts[0].casefold() != "tests"):
            raise WorkspaceSecurityError("CREATE_TEST 只能写入 workspace/tests。")

    def read_text(self, path: str) -> str:
        rel, target = self._target(path)
        self._policy(rel)
        self._reject_reparse(target)
        if not target.is_file():
            raise FileNotFoundError(rel.as_posix())
        if target.stat().st_size > self.MAX_FILE_BYTES:
            raise WorkspaceSecurityError("文件超过安全读取上限（16 MiB）。")
        return target.read_text(encoding="utf-8")

    def read_text_range(
        self,
        path: str,
        start_line: int,
        end_line: int,
    ) -> dict[str, object]:
        """Read an inclusive line range without silently discarding the rest."""
        rel, target = self._target(path)
        self._policy(rel)
        self._reject_reparse(target)
        if not target.is_file():
            raise FileNotFoundError(rel.as_posix())
        if target.stat().st_size > self.MAX_FILE_BYTES:
            raise WorkspaceSecurityError("文件超过安全读取上限（16 MiB）。")
        start = int(start_line)
        end = int(end_line)
        if start < 1 or end < start:
            raise WorkspaceSecurityError("READ_FILE 行范围无效。")

        lines: list[str] = []
        total = 0
        with target.open("r", encoding="utf-8", newline="") as handle:
            for number, line in enumerate(handle, 1):
                total = number
                if start <= number <= end:
                    lines.append(line)
        return {
            "path": rel.as_posix(),
            "content": "".join(lines),
            "start_line": start,
            "end_line": min(end, total),
            "total_lines": total,
            "complete": start == 1 and end >= total,
        }

    def write_text(self, path: str, content: str, *, test: bool = False) -> None:
        # Keep validation, temporary-file creation and replacement in one
        # critical section so two threads cannot race on the same destination.
        with self._write_lock:
            rel, target = self._target(path)
            self._policy(rel, write=True, test=test)
            data = str(content)
            if len(data.encode("utf-8")) > self.MAX_FILE_BYTES:
                raise WorkspaceSecurityError("写入内容超过安全上限。")
            target.parent.mkdir(parents=True, exist_ok=True)
            self._reject_reparse(target.parent)
            if target.exists():
                self._reject_reparse(target)
            # Unique sibling temp files keep the target untouched until the
            # complete payload is ready; os.replace publishes it atomically.
            temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.coder-tmp")
            try:
                temp.write_text(data, encoding="utf-8", newline="")
                os.replace(temp, target)
            finally:
                try:
                    temp.unlink()
                except FileNotFoundError:
                    pass

    def validate_notebook(self, content: str) -> dict:
        try:
            value = json.loads(str(content))
        except json.JSONDecodeError as exc:
            raise WorkspaceSecurityError(f"Notebook JSON 无效：{exc.msg}") from exc
        if not isinstance(value, dict):
            raise WorkspaceSecurityError("Notebook 顶层必须是 JSON 对象。")
        try:
            nbformat = int(value.get("nbformat", 0))
        except (TypeError, ValueError) as exc:
            raise WorkspaceSecurityError("Notebook nbformat 必须是整数。") from exc
        if nbformat < 4:
            raise WorkspaceSecurityError("Notebook 仅支持 nbformat >= 4。")
        cells = value.get("cells")
        if not isinstance(cells, list):
            raise WorkspaceSecurityError("Notebook cells 必须是数组。")
        if not isinstance(value.get("metadata", {}), dict):
            raise WorkspaceSecurityError("Notebook metadata 必须是对象。")
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                raise WorkspaceSecurityError(f"Notebook cell[{index}] 必须是对象。")
            if cell.get("cell_type") not in {"code", "markdown", "raw"}:
                raise WorkspaceSecurityError(
                    f"Notebook cell[{index}] 的 cell_type 无效。"
                )
            if "source" in cell and not isinstance(cell["source"], (str, list)):
                raise WorkspaceSecurityError(
                    f"Notebook cell[{index}] 的 source 必须是字符串或数组。"
                )
        return value

    def write_notebook(self, path: str, content: str) -> None:
        rel, _ = self._target(path)
        if rel.suffix.casefold() != ".ipynb":
            raise WorkspaceSecurityError("WRITE_NOTEBOOK 目标必须是 .ipynb 文件。")
        value = self.validate_notebook(content)
        normalized = json.dumps(value, ensure_ascii=False, indent=1) + "\n"
        self.write_text(path, normalized)

    def write_uploaded_text(self, path: str, data: bytes) -> None:
        if len(data) > self.MAX_FILE_BYTES:
            raise WorkspaceSecurityError("上传文件超过安全大小上限。")
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceSecurityError("Coder 仅接受 UTF-8 文本/代码文件上传。") from exc
        rel, _ = self._target(path)
        self._policy(rel, write=True)
        if rel.suffix.casefold() == ".ipynb":
            self.validate_notebook(content)
        self.write_text(path, content)

    def patch_text(self, path: str, old_text: str, new_text: str) -> None:
        current = self.read_text(path)
        old = str(old_text)
        if not old:
            raise WorkspaceSecurityError("PATCH_FILE 的 old_text 不能为空。")
        count = current.count(old)
        if count != 1:
            raise WorkspaceSecurityError(
                f"PATCH_FILE 要求 old_text 恰好出现 1 次，实际 {count} 次。"
            )
        self.write_text(path, current.replace(old, str(new_text), 1))

    @staticmethod
    def _cell_source(cell: dict) -> str:
        source = cell.get("source", "")
        if isinstance(source, list):
            return "".join(str(part) for part in source)
        return str(source or "")

    def patch_notebook(
        self,
        path: str,
        cell_index: int,
        old_source: str,
        new_source: str,
    ) -> None:
        rel, _ = self._target(path)
        if rel.suffix.casefold() != ".ipynb":
            raise WorkspaceSecurityError("PATCH_NOTEBOOK 目标必须是 .ipynb 文件。")
        try:
            index = int(cell_index)
        except (TypeError, ValueError) as exc:
            raise WorkspaceSecurityError("PATCH_NOTEBOOK.cell_index 必须是整数。") from exc
        if index < 0:
            raise WorkspaceSecurityError("PATCH_NOTEBOOK.cell_index 不能为负数。")
        old = str(old_source)
        if not old:
            raise WorkspaceSecurityError("PATCH_NOTEBOOK 的 old_source 不能为空。")
        value = self.validate_notebook(self.read_text(path))
        cells = value["cells"]
        if index >= len(cells):
            raise WorkspaceSecurityError(
                f"PATCH_NOTEBOOK.cell_index 超出范围：{index}，当前只有 {len(cells)} 个 cell。"
            )
        current = self._cell_source(cells[index])
        # Models sometimes serialize notebook source with CRLF while the
        # notebook stores LF, or provide only the exact snippet to change.
        # Normalize line endings first; preserve strict stale-write protection
        # by allowing a partial patch only when the snippet occurs once.
        normalize = lambda text: str(text).replace("\r\n", "\n").replace("\r", "\n")
        current_normalized = normalize(current)
        old_normalized = normalize(old)
        replacement = str(new_source)
        if current == old:
            cells[index]["source"] = replacement
        elif current_normalized == old_normalized:
            cells[index]["source"] = replacement
        elif old_normalized and current_normalized.count(old_normalized) == 1:
            cells[index]["source"] = current_normalized.replace(
                old_normalized, normalize(replacement), 1
            )
        else:
            preview = current[:1200]
            raise WorkspaceSecurityError(
                "PATCH_NOTEBOOK 无法安全匹配 old_source：它既不是目标 cell 的完整 source，"
                "也不是其中唯一出现的片段。请先 READ_FILE 重新读取该 cell，再重试。"
                f" cell_index={index}, current_source_length={len(current)}, "
                f"current_source_preview={preview!r}"
            )
        self.write_notebook(path, json.dumps(value, ensure_ascii=False, indent=1))

    def exists(self, path: str) -> bool:
        rel, target = self._target(path)
        self._policy(rel)
        self._reject_reparse(target)
        return target.is_file()

    def list_files(self) -> list[str]:
        output = []
        for path in self.root.rglob("*"):
            try:
                self._reject_reparse(path)
            except WorkspaceSecurityError:
                continue
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(self.root)
                self._policy(rel)
            except (ValueError, WorkspaceSecurityError):
                continue
            output.append(rel.as_posix())
        return sorted(output)

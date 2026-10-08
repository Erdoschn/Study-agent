import ntpath
import os
import stat
import re
from pathlib import Path


class WorkspaceSecurityError(PermissionError):
    pass


class WorkspaceFS:
    """Fail-closed filesystem boundary for the fixed Coder workspace."""

    EDITABLE_EXTENSIONS = frozenset({".py", ".pyi"})
    READABLE_EXTENSIONS = frozenset({
        ".py", ".pyi", ".txt", ".md", ".rst", ".json", ".toml",
        ".ini", ".cfg", ".yaml", ".yml", ".csv", ".tsv", ".xml",
    })
    BLOCKED_NAMES = frozenset({
        ".env", ".env.local", ".env.production", ".git-credentials",
        "id_rsa", "id_ed25519",
    })
    BLOCKED_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".kdbx"})
    BLOCKED_PREFIXES = (".coder-sandbox-",)
    MAX_FILE_BYTES = 1_048_576
    WINDOWS_DEVICE_NAMES = frozenset(
        {"con", "prn", "aux", "nul"}
        | {f"com{i}" for i in range(1, 10)}
        | {f"lpt{i}" for i in range(1, 10)}
    )

    def __init__(self, root: str | Path):
        raw_root = str(root)
        if os.name != "nt" and ntpath.isabs(raw_root):
            raise WorkspaceSecurityError("固定 Coder 工作空间只允许在 Windows 主机上使用。")
        self.root = Path(root).expanduser()
        if not self.root.is_absolute():
            self.root = self.root.resolve()
        if self.root.exists():
            self._reject_reparse(self.root)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
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
            or any(name.startswith(prefix) for prefix in self.BLOCKED_PREFIXES)
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
            raise WorkspaceSecurityError("文件超过安全读取上限。")
        return target.read_text(encoding="utf-8")

    def write_text(self, path: str, content: str, *, test: bool = False) -> None:
        rel, target = self._target(path)
        self._policy(rel, write=True, test=test)
        data = str(content)
        if len(data.encode("utf-8")) > self.MAX_FILE_BYTES:
            raise WorkspaceSecurityError("写入内容超过安全上限。")
        target.parent.mkdir(parents=True, exist_ok=True)
        self._reject_reparse(target.parent)
        if target.exists():
            self._reject_reparse(target)
        temp = target.with_name(target.name + ".coder-tmp")
        if temp.exists():
            self._reject_reparse(temp)
            temp.unlink()
        temp.write_text(data, encoding="utf-8", newline="")
        os.replace(temp, target)

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

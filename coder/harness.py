from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any

from .backup import CoderBackupStore
from .filesystem import WorkspaceFS
from .state import CoderGoal, CoderState


class CoderHarness:
    ALLOWED_ACTIONS = frozenset({
        "SEARCH", "LIST_FILES", "READ_FILE", "WRITE_FILE", "PATCH_FILE",
        "CREATE_TEST", "RUN_PYTHON", "RUN_PYTEST", "READ_DIFF", "VERIFY_GOAL",
    })
    ARGUMENT_KEYS = {
        "SEARCH": frozenset({"query"}),
        "LIST_FILES": frozenset(),
        "READ_FILE": frozenset({"path"}),
        "WRITE_FILE": frozenset({"path", "content"}),
        "PATCH_FILE": frozenset({"path", "old_text", "new_text"}),
        "CREATE_TEST": frozenset({"path", "content"}),
        "RUN_PYTHON": frozenset({"script_path"}),
        "RUN_PYTEST": frozenset({"paths"}),
        "READ_DIFF": frozenset(),
        "VERIFY_GOAL": frozenset(),
    }
    _SECRET_PATTERNS = (
        re.compile(r"(?i)(?:password|passwd|api[_ -]?key|secret|token|private[_ -]?key)\s*[:=]"),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"(?<![A-Za-z0-9])(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})(?![A-Za-z0-9])"),
        re.compile(r"(?<![A-Za-z0-9])(AIza[0-9A-Za-z_-]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16})(?![A-Za-z0-9])"),
    )
    _PATH_EXFIL_PATTERNS = (
        re.compile(r"(?i)(?:[A-Za-z]:[\\/]|\\\\|/home/|/users/|/workspace/|file://)"),
        re.compile(r"(?<!\w)(?:\.{1,2}[\\/])"),
    )
    MAX_SEARCH_QUERY_BYTES = 240

    def __init__(self, workspace: str, *, search_router=None, sandbox=None, backup=None):
        self.fs = WorkspaceFS(workspace)
        self.search_router = search_router
        self.sandbox = sandbox
        if self.sandbox is None:
            from .sandbox import DockerPythonSandbox
            self.sandbox = DockerPythonSandbox(self.fs.root)
        self.backup = backup or CoderBackupStore(self.fs.root)
        self._baseline: dict[str, str | None] = {}
        self._write_count = 0
        self._test_count = 0
        self.MAX_WRITES = 200
        self.MAX_TEST_RUNS = 100

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {"name": "SEARCH", "description": "Search configured sources.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "LIST_FILES", "description": "List readable files under the workspace.", "parameters": {"type": "object", "properties": {}}},
            {"name": "READ_FILE", "description": "Read one allowed text file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "WRITE_FILE", "description": "Create or replace one Python file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "PATCH_FILE", "description": "Replace exactly one matching Python fragment.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "CREATE_TEST", "description": "Create one pytest file under workspace/tests.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "RUN_PYTHON", "description": "Run one Python script inside the isolated sandbox.", "parameters": {"type": "object", "properties": {"script_path": {"type": "string"}}, "required": ["script_path"]}},
            {"name": "RUN_PYTEST", "description": "Run pytest inside the isolated sandbox. Empty paths means the full suite.", "parameters": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}}},
            {"name": "READ_DIFF", "description": "Show the diff of files modified during this Coder run.", "parameters": {"type": "object", "properties": {}}},
            {"name": "VERIFY_GOAL", "description": "Run deterministic goal gates.", "parameters": {"type": "object", "properties": {}}},
        ]

    def execute(self, action: str, arguments: dict[str, Any], state: CoderState) -> Any:
        action = str(action or "").upper()
        if action not in self.ALLOWED_ACTIONS:
            raise PermissionError(f"安全策略禁止动作：{action}")
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象。")
        self._validate_arguments(action, arguments)

        handlers = {
            "SEARCH": self._search,
            "LIST_FILES": self._list_files,
            "READ_FILE": self._read_file,
            "WRITE_FILE": self._write_file,
            "PATCH_FILE": self._patch_file,
            "CREATE_TEST": self._create_test,
            "RUN_PYTHON": self._run_python,
            "RUN_PYTEST": self._run_pytest,
            "READ_DIFF": self._read_diff,
            "VERIFY_GOAL": self._verify_goal,
        }
        return handlers[action](arguments, state)


    def _validate_arguments(self, action: str, arguments: dict[str, Any]) -> None:
        expected = self.ARGUMENT_KEYS[action]
        extra = set(arguments) - expected
        if extra:
            raise PermissionError(
                f"{action} 包含未授权参数：{sorted(map(str, extra))}"
            )

        string_fields = {
            "query", "path", "script_path", "content", "old_text", "new_text"
        }
        for key, value in arguments.items():
            if key in string_fields and not isinstance(value, str):
                raise ValueError(f"{action}.{key} 必须是字符串。")
        if "paths" in arguments:
            paths = arguments["paths"]
            if not isinstance(paths, list):
                raise ValueError("RUN_PYTEST.paths 必须是数组。")
            if len(paths) > 20 or any(not isinstance(path, str) for path in paths):
                raise ValueError("RUN_PYTEST.paths 必须是最多 20 个字符串。")
        for key in string_fields:
            value = arguments.get(key)
            if isinstance(value, str) and len(value.encode("utf-8")) > WorkspaceFS.MAX_FILE_BYTES:
                raise PermissionError(f"{action}.{key} 超过安全长度上限。")

    @classmethod
    def _validate_search_query(cls, query: str) -> str:
        normalized = unicodedata.normalize("NFKC", query).strip()
        if not normalized:
            raise ValueError("SEARCH 需要 query。")
        if len(normalized.encode("utf-8")) > cls.MAX_SEARCH_QUERY_BYTES:
            raise PermissionError("SEARCH query 超过安全长度上限。")
        if any(ord(ch) < 32 for ch in normalized):
            raise PermissionError("SEARCH query 包含控制字符。")
        if "\x60\x60\x60" in normalized:
            raise PermissionError("SEARCH query 不允许携带代码块。")
        if any(pattern.search(normalized) for pattern in cls._SECRET_PATTERNS):
            raise PermissionError("SEARCH query 疑似包含凭据或敏感数据，已拒绝发送。")
        if any(pattern.search(normalized) for pattern in cls._PATH_EXFIL_PATTERNS):
            raise PermissionError("SEARCH query 疑似包含本地路径，已拒绝发送。")
        quoted = re.findall(r"['\"]([^'\"]{32,})['\"]", normalized)
        if quoted:
            raise PermissionError("SEARCH query 疑似携带长文本片段，已拒绝发送。")
        for token in re.findall(r"[A-Za-z0-9_+/=-]{40,}", normalized):
            categories = sum([
                bool(re.search(r"[a-z]", token)),
                bool(re.search(r"[A-Z]", token)),
                bool(re.search(r"[0-9]", token)),
                bool(re.search(r"[^A-Za-z0-9]", token)),
            ])
            if len(token) >= 40 and categories >= 3:
                raise PermissionError("SEARCH query 疑似包含高熵凭据/内容片段，已拒绝发送。")
        return normalized

    def _search(self, args, _state):
        query = self._validate_search_query(args.get("query", ""))
        if self.search_router is None:
            raise RuntimeError("Coder SearchRouter 未配置。")
        from tools.search import SearchQuery
        results = self.search_router.search(SearchQuery(
            query=query,
            source="auto",
            source_preferences=[],
            categories=[],
            max_results=15,
            sort_by="relevance",
            sort_order="descending",
        ))
        return [
            {
                "source": item.source,
                "title": item.title,
                "url": item.url,
                "abstract": str(item.abstract)[:700],
                "identifier": item.identifier,
            }
            for item in results[:15]
        ]

    def _list_files(self, _args, _state):
        return self.fs.list_files()

    def _remember_baseline(self, path: str) -> None:
        if path in self._baseline:
            return
        try:
            self._baseline[path] = self.fs.read_text(path)
        except FileNotFoundError:
            self._baseline[path] = None

    def _read_file(self, args, _state):
        path = str(args.get("path", "")).strip()
        value = self.fs.read_text(path)
        self._remember_baseline(path)
        return {"path": path, "content": value}

    def _record_write(self, path: str, state: CoderState, *, created_test: bool = False) -> None:
        self._write_count += 1
        if self._write_count > self.MAX_WRITES:
            raise PermissionError("超过单次 Coder 文件修改上限。")
        self._remember_baseline(path)
        state.modified_files.add(path)
        state.modification_generation += 1
        if created_test:
            state.created_tests.add(path)

    def _write_file(self, args, state):
        path = str(args.get("path", "")).strip()
        self._remember_baseline(path)
        self.fs.write_text(path, str(args.get("content", "")))
        self._record_write(path, state)
        return {"status": "written", "path": path}

    def _patch_file(self, args, state):
        path = str(args.get("path", "")).strip()
        self._remember_baseline(path)
        self.fs.patch_text(path, str(args.get("old_text", "")), str(args.get("new_text", "")))
        self._record_write(path, state)
        return {"status": "patched", "path": path}

    def _create_test(self, args, state):
        path = str(args.get("path", "")).strip()
        self._remember_baseline(path)
        self.fs.write_text(path, str(args.get("content", "")), test=True)
        self._record_write(path, state, created_test=True)
        return {"status": "test_created", "path": path}

    def _run_python(self, args, state):
        path = str(args.get("script_path", "")).strip()
        rel, _ = self.fs._target(path)
        self.fs._policy(rel)
        if rel.suffix.casefold() != ".py":
            raise ValueError(f"Python 执行目标必须是 .py：{path}")
        self._test_count += 1
        if self._test_count > self.MAX_TEST_RUNS:
            raise PermissionError("超过单次 Coder 执行次数上限。")
        result = self.sandbox.run("python", [path])
        state.last_test_result = {
            "kind": "python", "paths": [path], "passed": result.passed,
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        }
        return state.last_test_result

    def _run_pytest(self, args, state):
        paths = args.get("paths", [])
        if not isinstance(paths, list) or len(paths) > 20:
            raise ValueError("RUN_PYTEST paths 必须是最多 20 个相对路径。")
        paths = [str(p).strip() for p in paths if str(p).strip()]
        self._test_count += 1
        if self._test_count > self.MAX_TEST_RUNS:
            raise PermissionError("超过单次 Coder 测试次数上限。")
        for path in paths:
            rel, _ = self.fs._target(path)
            self.fs._policy(rel)
            if rel.suffix.casefold() != ".py":
                raise ValueError(f"pytest 目标必须是 Python 文件：{path}")
        result = self.sandbox.run("pytest", paths)
        state.last_test_result = {
            "kind": "pytest", "paths": paths, "passed": result.passed,
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
            "timed_out": result.timed_out, "output_limited": result.output_limited,
            "backup_ok": False,
        }
        state.test_generation = state.modification_generation
        if result.passed:
            try:
                snapshot = self.backup.snapshot(state.modification_generation)
                state.backup_generation = snapshot.generation
                state.last_test_result["backup_ok"] = True
            except Exception as exc:
                state.last_test_result["backup_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        return state.last_test_result

    def _read_diff(self, _args, _state):
        chunks = []
        for path, baseline in sorted(self._baseline.items()):
            try:
                current = self.fs.read_text(path)
            except FileNotFoundError:
                current = ""
            before = (baseline or "").splitlines(keepends=True)
            after = current.splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(before, after, fromfile=path, tofile=path))
            if diff:
                chunks.append(diff)
        return "\n".join(chunks) if chunks else "No changes recorded."
    def _verify_goal(self, _args, state: CoderState):
        goal = state.goal or CoderGoal(state.request)
        checks = []
        required_files = goal.required_files
        for path in required_files:
            checks.append({"check": f"required_file:{path}", "ok": self.fs.exists(path)})
        for path in goal.required_tests:
            checks.append({"check": f"required_test:{path}", "ok": self.fs.exists(path)})
        checks.append({"check": "modified_files", "ok": bool(state.modified_files) if goal.must_modify else True})
        checks.append({"check": "pytest_created", "ok": bool(state.created_tests) if goal.must_create_tests else True})
        checks.append({
            "check": "pytest_passed_after_latest_change",
            "ok": bool(
                goal.must_pass_tests
                and state.last_test_result
                and state.last_test_result.get("kind") == "pytest"
                and state.last_test_result.get("passed")
                and state.test_generation == state.modification_generation
            ) if goal.must_pass_tests else True,
        })
        backup_present = self.backup.has_latest_snapshot()
        checks.append({
            "check": "last_known_good_backup",
            "ok": bool(
                goal.must_pass_tests
                and state.backup_generation == state.modification_generation
                and state.last_test_result
                and state.last_test_result.get("backup_ok") is True
                and backup_present
            ) if goal.must_pass_tests else True,
        })
        verified = all(item["ok"] for item in checks)
        state.goal_verified = verified
        return {"verified": verified, "checks": checks}

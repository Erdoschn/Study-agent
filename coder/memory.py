from __future__ import annotations

import json
import os
import stat
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.redaction import redact_sensitive_text
from .knowledge_graph import CoderKnowledgeGraph


class CoderMemoryStore:
    """Persistent, host-side memory for Coder runs.

    The memory file lives at the Coder workspace root rather than inside any
    project directory, so the coding sandbox cannot access it through its
    workspace boundary.
    """

    VERSION = 1
    MAX_RUNS = 200
    MAX_KNOWLEDGE = 100
    MAX_FILE_BYTES = 16 * 1024 * 1024
    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_LOCKS_GUARD = threading.Lock()

    def __init__(self, workspace_root: str | Path):
        self.root = Path(workspace_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / ".coder-memory.json"
        with self._ROOT_LOCKS_GUARD:
            self._lock = self._ROOT_LOCKS.setdefault(
                str(self.root).casefold(),
                threading.RLock(),
            )
        self._ensure_safe_memory_path()
        self.knowledge_graph = CoderKnowledgeGraph(self.root)

    def _ensure_safe_memory_path(self) -> None:
        """Reject links and special files instead of following them for read/write."""
        try:
            status = self.path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise PermissionError("无法检查 Coder memory 文件。") from exc
        if stat.S_ISLNK(status.st_mode):
            raise PermissionError("Coder memory 文件禁止使用符号链接。")
        if not stat.S_ISREG(status.st_mode):
            raise PermissionError("Coder memory 路径必须是普通文件。")
        if os.name == "nt" and getattr(status, "st_file_attributes", 0) & 0x400:
            raise PermissionError("Coder memory 文件禁止使用 Windows reparse point。")
        if status.st_size > self.MAX_FILE_BYTES:
            raise PermissionError("Coder memory 文件超过 16 MiB 安全上限。")

    @staticmethod
    def _sanitize_data(value: Any) -> Any:
        """Redact credentials and host paths recursively before data is exposed or stored."""
        if isinstance(value, str):
            return redact_sensitive_text(value)
        if isinstance(value, list):
            return [CoderMemoryStore._sanitize_data(item) for item in value]
        if isinstance(value, dict):
            return {
                key: CoderMemoryStore._sanitize_data(item)
                for key, item in value.items()
            }
        return value

    def load(self) -> dict[str, Any]:
        with self._lock:
            return self._load_unlocked()

    def record_run(self, *, project: str, state: Any) -> dict[str, Any]:
        entry = self._build_run(project=project, state=state)
        with self._lock:
            data = self._load_unlocked()
            data["runs"].append(entry)
            data["runs"] = data["runs"][-self.MAX_RUNS:]
            self._merge_knowledge(data, entry)
            self._write_unlocked(data)
        try:
            self.knowledge_graph.record_run(entry)
        except Exception:
            # Graph persistence is additive; a graph I/O failure must not erase a run history entry.
            pass
        return entry

    def recent_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), self.MAX_RUNS))
        return list(reversed(self.load()["runs"][-limit:]))

    def knowledge(self) -> dict[str, list[dict[str, Any]]]:
        data = self.load()
        return {
            "strategies": list(data.get("strategies", [])),
            "experiences": list(data.get("experiences", [])),
            "technologies": list(data.get("technologies", [])),
        }

    def _load_unlocked(self) -> dict[str, Any]:
        self._ensure_safe_memory_path()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            value = {
                "version": self.VERSION,
                "runs": [],
                "strategies": [],
                "experiences": [],
                "technologies": [],
            }
        except (OSError, ValueError) as exc:
            # Never silently replace a corrupt history with an empty database.
            raise RuntimeError(
                "Coder memory 无法读取或 JSON 格式无效；为避免数据丢失，已拒绝覆盖。"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError("Coder memory 顶层格式无效；已拒绝覆盖。")
        value.setdefault("version", self.VERSION)
        for key in ("runs", "strategies", "experiences", "technologies"):
            rows = value.get(key)
            if not isinstance(rows, list):
                raise RuntimeError(f"Coder memory 字段 {key} 格式无效；已拒绝覆盖。")
            value[key] = [item for item in rows if isinstance(item, dict)]
        sanitized = self._sanitize_data(value)
        if sanitized != value:
            self._write_unlocked(sanitized)
        return sanitized

    def record_feedback(self, *, run_id: str, feedback: str) -> dict[str, Any] | None:
        """Attach explicit user feedback to a completed run."""
        run_id = str(run_id or "").strip()
        feedback = redact_sensitive_text(" ".join(str(feedback or "").split()).strip(), max_chars=4000)
        if not run_id or not feedback:
            raise ValueError("run_id 和 feedback 不能为空。")
        if len(feedback.encode("utf-8")) > 16 * 1024:
            raise ValueError("用户评价超过 16 KiB。")

        with self._lock:
            data = self._load_unlocked()
            for run in reversed(data["runs"]):
                if str(run.get("id", "")) != run_id:
                    continue
                now = datetime.now(timezone.utc).isoformat()
                run["feedback"] = feedback[:4000]
                run["feedback_at"] = now
                self._write_unlocked(data)
                return {
                    "run_id": run_id,
                    "project": str(run.get("project", "")),
                    "feedback": run["feedback"],
                    "created_at": now,
                }
        return None

    def recent_feedback(self, limit: int = 8) -> list[dict[str, Any]]:
        """Return recent explicit user feedback for future Coder decisions."""
        limit = max(1, min(int(limit), self.MAX_RUNS))
        runs = self.load()["runs"]
        feedback: list[dict[str, Any]] = []
        for run in reversed(runs):
            text = str(run.get("feedback", "") or "").strip()
            if not text:
                continue
            feedback.append({
                "project": str(run.get("project", "")),
                "request": str(run.get("request", ""))[:300],
                "feedback": text[:800],
                "created_at": str(run.get("feedback_at", run.get("created_at", ""))),
            })
            if len(feedback) >= limit:
                break
        return feedback

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        self._ensure_safe_memory_path()
        safe_data = self._sanitize_data(data)
        payload = json.dumps(safe_data, ensure_ascii=False, indent=2) + "\n"
        encoded = payload.encode("utf-8")
        if len(encoded) > self.MAX_FILE_BYTES:
            raise ValueError("Coder memory 写入超过 16 MiB 安全上限。")

        # A unique sibling file plus os.replace avoids cross-instance temp-file
        # races and keeps readers from observing partial JSON.
        temp = self.root / f".coder-memory.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with temp.open("x", encoding="utf-8", newline="") as handle:
                if os.name != "nt":
                    os.chmod(temp, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _build_run(*, project: str, state: Any) -> dict[str, Any]:
        steps = list(getattr(state, "steps", []) or [])
        actions = [str(step.action).upper() for step in steps if getattr(step, "action", None)]
        failures = []
        for step in steps:
            error = str(getattr(step, "error", "") or "").strip()
            if error:
                failures.append({
                    "action": str(getattr(step, "action", "")),
                    "error": redact_sensitive_text(error, max_chars=500),
                })

        extensions = set()
        for step in steps:
            args = getattr(step, "arguments", {})
            if not isinstance(args, dict):
                continue
            for key in ("path", "script_path"):
                value = str(args.get(key, "") or "").strip()
                if "." in value:
                    extensions.add(Path(value).suffix.casefold())

        technologies = []
        if ".py" in extensions or any(
            action in {"RUN_PYTHON", "RUN_PYTEST", "WRITE_FILE", "WRITE_NOTEBOOK", "PATCH_FILE", "CREATE_TEST"}
            for action in actions
        ):
            technologies.append("Python")
        if ".ipynb" in extensions:
            technologies.append("Jupyter Notebook")
        if any(action in {"CREATE_TEST", "RUN_PYTEST"} for action in actions):
            technologies.append("pytest")
        if "SEARCH" in actions:
            technologies.append("web research")

        strategy_actions = [
            action for action in actions
            if action not in {"PLAN", "VERIFY_GOAL"}
        ]
        strategy = " → ".join(strategy_actions[:20])

        experience = []
        for failure in failures[:8]:
            experience.append(
                f"{failure['action']}: {failure['error']}"
            )
        if getattr(state, "goal_verified", False) and failures:
            experience.append("最终通过验证：先失败后修正并完成 Goal。")

        last_test = getattr(state, "last_test_result", None)
        if isinstance(last_test, dict):
            last_test = dict(last_test)
            for key in ("stdout", "stderr"):
                if key in last_test:
                    last_test[key] = redact_sensitive_text(last_test[key] or "")[-2000:]
        else:
            last_test = None

        return {
            "id": "run-" + uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project": str(project),
            "request": redact_sensitive_text(getattr(state, "request", ""), max_chars=2000),
            "finished": bool(getattr(state, "finished", False)),
            "verified": bool(getattr(state, "goal_verified", False)),
            "cancelled": bool(getattr(state, "cancelled", False)),
            "error": redact_sensitive_text(getattr(state, "error", "") or "", max_chars=1000),
            "summary": redact_sensitive_text(getattr(state, "summary", "") or "", max_chars=4000),
            "step_count": int(getattr(state, "step_count", 0)),
            "modified_files": sorted(getattr(state, "modified_files", set()) or set()),
            "created_tests": sorted(getattr(state, "created_tests", set()) or set()),
            "chat_resets": int(getattr(state, "chat_resets", 0)),
            "study_agent_calls": int(getattr(state, "metrics", {}).get("study_agent_calls", 0)),
            "last_test_result": last_test,
            "strategy": strategy,
            "experience": experience,
            "technologies": technologies,
            "feedback": "",
        }

    def _merge_knowledge(self, data: dict[str, Any], entry: dict[str, Any]) -> None:
        now = entry["created_at"]

        if entry["strategy"]:
            self._upsert(
                data["strategies"],
                key="text",
                value=entry["strategy"],
                now=now,
            )

        for experience in entry["experience"]:
            self._upsert(
                data["experiences"],
                key="text",
                value=experience,
                now=now,
            )

        for technology in entry["technologies"]:
            self._upsert(
                data["technologies"],
                key="name",
                value=technology,
                now=now,
            )

        data["strategies"] = data["strategies"][-self.MAX_KNOWLEDGE:]
        data["experiences"] = data["experiences"][-self.MAX_KNOWLEDGE:]
        data["technologies"] = data["technologies"][-self.MAX_KNOWLEDGE:]

    @staticmethod
    def _upsert(
        items: list[dict[str, Any]],
        *,
        key: str,
        value: str,
        now: str,
    ) -> None:
        for item in items:
            if item.get(key) == value:
                item["count"] = int(item.get("count", 0)) + 1
                item["last_seen"] = now
                return
        items.append({
            key: value,
            "count": 1,
            "last_seen": now,
        })

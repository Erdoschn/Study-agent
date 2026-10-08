from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    def __init__(self, workspace_root: str | Path):
        self.root = Path(workspace_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / ".coder-memory.json"
        self.knowledge_graph = CoderKnowledgeGraph(self.root)
        self._lock = threading.Lock()

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
        if not self.path.exists():
            return {
                "version": self.VERSION,
                "runs": [],
                "strategies": [],
                "experiences": [],
                "technologies": [],
            }
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        value.setdefault("version", self.VERSION)
        for key in ("runs", "strategies", "experiences", "technologies"):
            if not isinstance(value.get(key), list):
                value[key] = []
        return value

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)

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
                    "error": error[:500],
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

        return {
            "id": "run-" + uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project": str(project),
            "request": str(getattr(state, "request", ""))[:2000],
            "finished": bool(getattr(state, "finished", False)),
            "verified": bool(getattr(state, "goal_verified", False)),
            "error": str(getattr(state, "error", "") or "")[:1000],
            "summary": str(getattr(state, "summary", "") or "")[:4000],
            "step_count": int(getattr(state, "step_count", 0)),
            "modified_files": sorted(getattr(state, "modified_files", set()) or set()),
            "created_tests": sorted(getattr(state, "created_tests", set()) or set()),
            "chat_resets": int(getattr(state, "chat_resets", 0)),
            "study_agent_calls": int(getattr(state, "metrics", {}).get("study_agent_calls", 0)),
            "strategy": strategy,
            "experience": experience,
            "technologies": technologies,
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

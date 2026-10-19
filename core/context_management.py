"""Role-scoped context, explicit handoffs, and safe cache-key utilities.

This module intentionally keeps role histories separate from shared task facts.
It does not implement an LLM-result cache: cache keys are only safe identities,
and callers must still honor provider policy, expiry, and authorization.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator, Mapping

_SCOPE = contextvars.ContextVar("study_agent_context_scope", default=None)


@dataclass(frozen=True)
class ContextScope:
    user_id: str
    task_id: str
    role_id: str = "orchestrator"


@dataclass(frozen=True)
class Handoff:
    task_id: str
    source_role: str
    target_role: str
    artifact_version: str
    payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...] = ()


@dataclass
class _TaskContext:
    facts: dict[str, Any] = field(default_factory=dict)
    role_histories: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=24)))
    handoffs: deque = field(default_factory=lambda: deque(maxlen=128))


class RoleContextManager:
    """Thread-safe in-memory task ledger with strict user/task/role boundaries."""

    def __init__(self, *, max_tasks: int = 512, history_limit: int = 24):
        if max_tasks < 1 or history_limit < 1:
            raise ValueError("max_tasks and history_limit must be positive")
        self.max_tasks = max_tasks
        self.history_limit = history_limit
        self._lock = threading.RLock()
        self._tasks: dict[tuple[str, str], _TaskContext] = {}
        self._task_order: deque[tuple[str, str]] = deque()

    @staticmethod
    def _identity(value: object, field_name: str) -> str:
        result = str(value or "").strip()
        if not result or len(result) > 256 or any(ord(ch) < 32 for ch in result):
            raise ValueError(f"{field_name} must be a non-empty safe identifier")
        return result

    def _task(self, user_id: str, task_id: str) -> _TaskContext:
        key = (self._identity(user_id, "user_id"), self._identity(task_id, "task_id"))
        task = self._tasks.get(key)
        if task is None:
            task = _TaskContext()
            task.role_histories = defaultdict(lambda: deque(maxlen=self.history_limit))
            self._tasks[key] = task
            self._task_order.append(key)
            while len(self._tasks) > self.max_tasks:
                oldest = self._task_order.popleft()
                self._tasks.pop(oldest, None)
        return task

    @contextmanager
    def scope(self, user_id: str, task_id: str, role_id: str = "orchestrator") -> Iterator[ContextScope]:
        scope = ContextScope(
            self._identity(user_id, "user_id"),
            self._identity(task_id, "task_id"),
            self._identity(role_id, "role_id"),
        )
        token = _SCOPE.set(scope)
        try:
            yield scope
        finally:
            _SCOPE.reset(token)

    def add_turn(self, user_id: str, task_id: str, role_id: str, *, prompt: str, response: str) -> None:
        role = self._identity(role_id, "role_id")
        with self._lock:
            task = self._task(user_id, task_id)
            task.role_histories[role].append({"prompt": str(prompt), "response": str(response)})

    def role_history(self, user_id: str, task_id: str, role_id: str) -> list[dict[str, str]]:
        role = self._identity(role_id, "role_id")
        with self._lock:
            task = self._tasks.get((self._identity(user_id, "user_id"), self._identity(task_id, "task_id")))
            return [dict(turn) for turn in task.role_histories.get(role, ())] if task else []

    def put_fact(self, user_id: str, task_id: str, key: str, value: Any) -> None:
        fact_key = self._identity(key, "fact key")
        with self._lock:
            self._task(user_id, task_id).facts[fact_key] = value

    def task_facts(self, user_id: str, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get((self._identity(user_id, "user_id"), self._identity(task_id, "task_id")))
            return dict(task.facts) if task else {}

    def handoff(self, user_id: str, record: Handoff) -> None:
        with self._lock:
            task = self._task(user_id, record.task_id)
            task.handoffs.append(record)

    def handoffs(self, user_id: str, task_id: str, *, target_role: str | None = None) -> list[Handoff]:
        with self._lock:
            task = self._tasks.get((self._identity(user_id, "user_id"), self._identity(task_id, "task_id")))
            records = list(task.handoffs) if task else []
        if target_role is not None:
            records = [record for record in records if record.target_role == target_role]
        return records

    def clear_task(self, user_id: str, task_id: str) -> None:
        key = (self._identity(user_id, "user_id"), self._identity(task_id, "task_id"))
        with self._lock:
            self._tasks.pop(key, None)
            self._task_order = deque(item for item in self._task_order if item != key)


ROLE_CONTEXTS = RoleContextManager()


def current_context() -> ContextScope | None:
    return _SCOPE.get()


@contextmanager
def context_scope(user_id: str, task_id: str, role_id: str = "orchestrator"):
    with ROLE_CONTEXTS.scope(user_id, task_id, role_id) as scope:
        yield scope


def role_context(role_id: str) -> Callable:
    """Decorator that marks calls so usage events and subcalls can identify roles."""
    role = str(role_id).strip()
    if not role:
        raise ValueError("role_id must not be empty")

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args, **kwargs):
            current = current_context()
            if current is None:
                # Isolate standalone/test calls rather than accidentally inheriting
                # another invocation's context. No user data is guessed from globals.
                with context_scope("local", "standalone", role):
                    return function(*args, **kwargs)
            with context_scope(current.user_id, current.task_id, role):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def stable_prefix_fingerprint(*parts: str, version: str = "v1") -> str:
    """Hash stable prompt components; dynamic user/task data must not be passed here."""
    payload = json.dumps({"version": version, "parts": list(parts)}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def application_cache_key(
    *,
    user_id: str,
    task_id: str,
    role_id: str,
    provider_id: str,
    model_id: str,
    prompt_version: str,
    input_digest: str,
    evidence_version: str = "",
    tool_version: str = "",
    permission_scope: str = "",
) -> str:
    """Return a deterministic, privacy-scoped cache identity with explicit invalidators."""
    values = {
        "user_id": user_id, "task_id": task_id, "role_id": role_id,
        "provider_id": provider_id, "model_id": model_id,
        "prompt_version": prompt_version, "input_digest": input_digest,
        "evidence_version": evidence_version, "tool_version": tool_version,
        "permission_scope": permission_scope,
    }
    canonical = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

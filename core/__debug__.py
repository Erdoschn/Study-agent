from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from threading import local
from typing import Callable, Iterator

from .redaction import redact_sensitive_text


class DebugTracer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.depth = 0
        self._local = local()
        self._local.depth = 0

    @staticmethod
    def _redact_attachment_metadata(message: str) -> str:
        """Hide attachment transport metadata from every log sink.

        The raw file bytes are never part of the question, but the API appends
        filenames to a transport marker so independent audit calls can identify
        the files. Redact that suffix before it reaches terminal logs or UI sinks.
        """
        value = str(message)
        marker = "【原始文件附件】"
        if marker not in value:
            return value
        return value.split(marker, 1)[0] + marker + "[attachment metadata redacted]"

    @staticmethod
    def _redact_user_text(module: str, message: str) -> str:
        """Keep useful diagnostics without printing user prompts or query text."""
        value = str(message)

        if module == "StudyAgent" and "QUESTION → " in value:
            prefix, request = value.split("QUESTION → ", 1)
            value = f"{prefix}QUESTION → [user request redacted; chars={len(request)}]"

        if module == "GoalMatcher" and "MATCH → current=" in value:
            prefix, rest = value.split("MATCH → current=", 1)
            if ", candidates=" in rest:
                current, suffix = rest.split(", candidates=", 1)
                value = f"{prefix}MATCH → current=[redacted; chars={len(current)}], candidates={suffix}"

        if module == "KnowledgeGraph":
            for label in ("RESOLVE → query=", "CONTEXT → query="):
                if label in value:
                    prefix, rest = value.split(label, 1)
                    boundary = ", matched="
                    if boundary in rest:
                        query, suffix = rest.split(boundary, 1)
                        value = f"{prefix}{label}[redacted; chars={len(query)}]{boundary}{suffix}"
                    break

        sensitive_fields = ()
        if module == "ToolExecutor":
            sensitive_fields = (
                ("CALCULATE → expression=", ", result_type="),
                ("VERIFY → claim=", ", evidence_count="),
            )
        elif module == "AgentToolLoop":
            sensitive_fields = (("VERIFY RESULT → claim=", ", status="),)
        for label, boundary in sensitive_fields:
            if label in value and boundary in value.split(label, 1)[1]:
                prefix, rest = value.split(label, 1)
                sensitive, suffix = rest.split(boundary, 1)
                value = (
                    f"{prefix}{label}[redacted; chars={len(sensitive)}]"
                    f"{boundary}{suffix}"
                )
                break
        return value

    def log(self, module: str, message: str) -> None:
        message = self._redact_attachment_metadata(message)
        message = self._redact_user_text(module, message)
        message = redact_sensitive_text(message)
        sink = getattr(self._local, "sink", None)
        # API sinks translate selected internal events into user-facing status
        # updates. They must not replace terminal logging when debug is ON.
        if sink is not None:
            sink(module, message)

        # Global config controls terminal verbosity consistently in every
        # worker thread, including Playwright's dedicated browser thread.
        if not self.enabled:
            return

        timestamp = datetime.now().strftime(
            "%H:%M:%S.%f"
        )[:-3]

        indent = "  " * getattr(self._local, "depth", 0)

        print(
            f"{indent}[DEBUG {timestamp}] "
            f"[{module}] {message}",
            flush=True,
        )

    @contextmanager
    def scope(
        self,
        module: str,
        message: str,
    ) -> Iterator[None]:
        self.log(
            module,
            f"ENTER → {message}",
        )

        depth = getattr(self._local, "depth", 0)
        self._local.depth = depth + 1

        try:
            yield
        except Exception as exc:
            self.log(
                module,
                f"ERROR → "
                f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            self._local.depth = max(
                0,
                depth,
            )

            self.log(
                module,
                "EXIT",
            )

    def bind_thread(
        self,
        sink: Callable[[str, str], None],
        *,
        enabled: bool = True,
    ) -> None:
        self._local.sink = sink
        self._local.enabled = bool(enabled)

    def clear_thread_binding(self) -> None:
        for name in ("sink", "enabled"):
            try:
                delattr(self._local, name)
            except AttributeError:
                pass

    def set_enabled(
        self,
        enabled: bool,
    ) -> None:
        self.enabled = enabled


debug = DebugTracer()
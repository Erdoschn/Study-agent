from __future__ import annotations

from threading import Event


class RunCancelled(RuntimeError):
    """Raised when a Coder run is explicitly cancelled by the user."""


def raise_if_cancelled(event: Event | None) -> None:
    if event is not None and event.is_set():
        raise RunCancelled("Coder 任务已被用户中止。")

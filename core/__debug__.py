from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from threading import local
from typing import Callable, Iterator


class DebugTracer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.depth = 0
        self._local = local()

    def log(self, module: str, message: str) -> None:
        sink = getattr(self._local, "sink", None)
        enabled = getattr(self._local, "enabled", self.enabled)
        if sink is not None:
            sink(module, message)
            return
        if not enabled:
            return

        timestamp = datetime.now().strftime(
            "%H:%M:%S.%f"
        )[:-3]

        indent = "  " * self.depth

        print(
            f"{indent}[DEBUG {timestamp}] "
            f"[{module}] {message}"
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

        self.depth += 1

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
            self.depth = max(
                0,
                self.depth - 1,
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
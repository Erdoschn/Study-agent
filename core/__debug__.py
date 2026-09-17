from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator


class DebugTracer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.depth = 0

    def log(self, module: str, message: str) -> None:
        if not self.enabled:
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

    def set_enabled(
        self,
        enabled: bool,
    ) -> None:
        self.enabled = enabled


debug = DebugTracer()
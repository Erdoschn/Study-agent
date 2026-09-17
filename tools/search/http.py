from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from core.__debug__ import debug


@dataclass
class HttpResponse:
    status: int
    reason: str
    headers: dict[str, str]
    body: bytes
    attempts: int
    elapsed_seconds: float


class HttpRequestError(RuntimeError):
    """HTTP/network failure with enough context for the search layer."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        reason: str = "",
        response_body: str = "",
        retryable: bool = False,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.response_body = response_body
        self.retryable = retryable
        self.attempts = attempts


class HttpClient:
    """Small stdlib-only HTTP transport shared by search providers."""

    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 2,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> HttpResponse:
        total_attempts = self.retries + 1
        started = time.monotonic()

        for attempt in range(1, total_attempts + 1):
            debug.log("HttpClient", f"GET attempt={attempt}/{total_attempts} → {url}")
            request = urllib.request.Request(url=url, method="GET", headers=headers or {})

            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                    return HttpResponse(
                        status=response.status,
                        reason=response.reason or "",
                        headers=dict(response.headers.items()),
                        body=body,
                        attempts=attempt,
                        elapsed_seconds=time.monotonic() - started,
                    )
            except urllib.error.HTTPError as exc:
                body = self._read_error_body(exc)
                retryable = exc.code in self.RETRYABLE_STATUS_CODES
                debug.log(
                    "HttpClient",
                    f"HTTP ERROR attempt={attempt}: {exc.code} {exc.reason}; retryable={retryable}",
                )
                if not retryable or attempt >= total_attempts:
                    raise HttpRequestError(
                        f"HTTP 请求失败：{exc.code} {exc.reason}",
                        status_code=exc.code,
                        reason=str(exc.reason or ""),
                        response_body=body,
                        retryable=retryable,
                        attempts=attempt,
                    ) from exc
            except urllib.error.URLError as exc:
                reason = str(exc.reason)
                debug.log("HttpClient", f"NETWORK ERROR attempt={attempt}: {reason}")
                if attempt >= total_attempts:
                    raise HttpRequestError(
                        f"网络请求失败：{reason}",
                        reason=reason,
                        retryable=True,
                        attempts=attempt,
                    ) from exc
            except TimeoutError as exc:
                debug.log("HttpClient", f"TIMEOUT attempt={attempt}")
                if attempt >= total_attempts:
                    raise HttpRequestError(
                        "网络请求超时",
                        reason="timeout",
                        retryable=True,
                        attempts=attempt,
                    ) from exc

            delay = self.backoff_seconds * (2 ** (attempt - 1))
            if delay:
                time.sleep(delay)

        raise AssertionError("unreachable")

    @staticmethod
    def _read_error_body(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read(4096).decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

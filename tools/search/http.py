from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import requests

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
    """Shared HTTP transport based on requests."""

    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 2,
        backoff_seconds: float = 2.0,
        session: requests.Session | None = None,
        session_factory: Callable[[], requests.Session] | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        if session is not None and session_factory is not None:
            raise ValueError("session 和 session_factory 只能设置一个。")
        self.session = (
            session
            or session_factory()
            if session_factory is not None
            else session
        )
        if self.session is None:
            self.session = requests.Session()

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        total_attempts = self.retries + 1
        started = time.monotonic()

        for attempt in range(1, total_attempts + 1):
            debug.log(
                "HttpClient",
                f"GET attempt={attempt}/{total_attempts} → {url}",
            )

            try:
                response = self.session.get(
                    url,
                    headers=headers or {},
                    timeout=self.timeout,
                )

                status = response.status_code
                reason = response.reason or ""
                body = response.content or b""
                response_headers = dict(response.headers)

                if 200 <= status < 300:
                    return HttpResponse(
                        status=status,
                        reason=reason,
                        headers=response_headers,
                        body=body,
                        attempts=attempt,
                        elapsed_seconds=time.monotonic() - started,
                    )

                retryable = status in self.RETRYABLE_STATUS_CODES
                debug.log(
                    "HttpClient",
                    f"HTTP ERROR attempt={attempt}: {status} {reason}; retryable={retryable}",
                )

                if not retryable or attempt >= total_attempts:
                    error_body = body.decode("utf-8", errors="replace").strip()
                    raise HttpRequestError(
                        f"HTTPError: HTTP Error {status}: {reason}",
                        status_code=status,
                        reason=reason,
                        response_body=error_body,
                        retryable=retryable,
                        attempts=attempt,
                    )

            except HttpRequestError:
                raise
            except requests.Timeout as exc:
                debug.log(
                    "HttpClient",
                    f"TIMEOUT attempt={attempt}: {exc}",
                )
                if attempt >= total_attempts:
                    raise HttpRequestError(
                        f"网络请求超时：{exc}",
                        reason="timeout",
                        retryable=True,
                        attempts=attempt,
                    ) from exc
            except requests.RequestException as exc:
                debug.log(
                    "HttpClient",
                    f"NETWORK ERROR attempt={attempt}: {type(exc).__name__}: {exc}",
                )
                if attempt >= total_attempts:
                    raise HttpRequestError(
                        f"网络请求失败：{exc}",
                        reason=type(exc).__name__,
                        retryable=True,
                        attempts=attempt,
                    ) from exc

            delay = self.backoff_seconds * (2 ** (attempt - 1))
            if delay:
                time.sleep(delay)

        raise AssertionError("unreachable")

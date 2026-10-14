from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from core.http_security import is_loopback_host
from core.redaction import redact_sensitive_text


DEFAULT_STUDY_AGENT_URL = "http://127.0.0.1:8000/internal/study/ask"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward a bridge request or its Authorization header via redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _validate_bridge_url(url: str) -> str:
    value = str(url or "").strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        # Force validation of an explicit port now rather than during the request.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Study Agent Bridge URL 格式无效。") from exc

    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Study Agent Bridge URL 必须是无凭据、无查询参数的 HTTP(S) endpoint。"
        )
    if not is_loopback_host(parsed.hostname) and parsed.scheme.casefold() != "https":
        raise ValueError("非本机 Study Agent Bridge 必须使用 HTTPS。")
    return value


class StudyAgentBridge:
    """Small HTTP boundary between Coder and the independently running Study Agent."""

    def __init__(
        self,
        url: str | None = None,
        *,
        timeout: float | None = None,
        api_key: str | None = None,
    ):
        self.url = _validate_bridge_url(
            url or os.getenv("CODER_STUDY_AGENT_URL", DEFAULT_STUDY_AGENT_URL)
        )
        self.timeout = max(
            1.0,
            float(
                timeout
                if timeout is not None
                else os.getenv("CODER_STUDY_AGENT_TIMEOUT", "180")
            ),
        )
        self.api_key = str(
            api_key
            if api_key is not None
            else os.getenv("CODER_STUDY_AGENT_API_KEY", "")
        ).strip()

    def ask(
        self,
        question: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        question = redact_sensitive_text(
            str(question or "").strip(),
            max_chars=8000,
        )
        if not question:
            raise ValueError("ASK_STUDY_AGENT 需要 question。")
        payload = {
            "question": question,
            "context": self._limit_context(context),
            "requester": "coder",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")

        # Do not follow redirects: the bridge may carry an Authorization header
        # and the endpoint is expected to respond directly with its JSON payload.
        opener = urllib.request.build_opener(_NoRedirectHandler())
        try:
            with opener.open(request, timeout=self.timeout) as response:
                raw = response.read(1_048_577)
        except urllib.error.HTTPError as exc:
            # A response one byte over the limit is rejected; never parse a clipped JSON body.
            detail = redact_sensitive_text(
                exc.read(4096).decode("utf-8", errors="replace"),
                max_chars=500,
            )
            raise RuntimeError(
                f"Study Agent Bridge HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = redact_sensitive_text(exc.reason, max_chars=300)
            raise RuntimeError(
                f"Study Agent Bridge 无法连接配置的 endpoint：{reason}"
            ) from exc
        except TimeoutError as exc:
            raise TimeoutError(
                f"Study Agent Bridge 请求超时（{self.timeout:.0f}s）。"
            ) from exc

        if len(raw) > 1_048_576:
            raise RuntimeError("Study Agent Bridge 响应超过 1 MiB 安全上限。")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Study Agent Bridge 返回了无效 JSON。") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Study Agent Bridge 返回值必须是 JSON 对象。")
        if "error" in value:
            error = value.get("error")
            raise RuntimeError(f"Study Agent Bridge 错误：{error}")
        answer = str(value.get("answer", "") or "").strip()
        if not answer:
            raise RuntimeError("Study Agent 没有返回可用回答。")
        return value

    @staticmethod
    def _limit_context(context: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(context, dict):
            return {}
        value = dict(context)
        value["request"] = redact_sensitive_text(value.get("request", ""), max_chars=2000)
        value["goal"] = redact_sensitive_text(value.get("goal", ""), max_chars=2000)
        value["modified_files"] = [
            redact_sensitive_text(item, max_chars=200)
            for item in list(value.get("modified_files", []))[:20]
        ]
        value["created_tests"] = [
            redact_sensitive_text(item, max_chars=200)
            for item in list(value.get("created_tests", []))[:20]
        ]
        value["last_test_result"] = redact_sensitive_text(
            value.get("last_test_result", ""),
            max_chars=6000,
        )
        value["last_observation"] = redact_sensitive_text(
            value.get("last_observation", ""),
            max_chars=6000,
        )
        value["recent_actions"] = [
            redact_sensitive_text(item, max_chars=100)
            for item in list(value.get("recent_actions", []))[-12:]
        ]
        return value

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_STUDY_AGENT_URL = "http://127.0.0.1:8000/internal/study/ask"


class StudyAgentBridge:
    """Small HTTP boundary between Coder and the independently running Study Agent."""

    def __init__(
        self,
        url: str | None = None,
        *,
        timeout: float | None = None,
        api_key: str | None = None,
    ):
        self.url = str(
            url
            or os.getenv("CODER_STUDY_AGENT_URL", DEFAULT_STUDY_AGENT_URL)
        ).strip().rstrip("/")
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
        question = str(question or "").strip()
        if not question:
            raise ValueError("ASK_STUDY_AGENT 需要 question。")
        payload = {
            "question": question[:8000],
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

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(1_048_576)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Study Agent Bridge HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Study Agent Bridge 无法连接 {self.url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise TimeoutError(
                f"Study Agent Bridge 请求超时（{self.timeout:.0f}s）。"
            ) from exc

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
        value["request"] = str(value.get("request", ""))[:2000]
        value["goal"] = str(value.get("goal", ""))[:2000]
        value["modified_files"] = [
            str(item)[:200] for item in list(value.get("modified_files", []))[:20]
        ]
        value["created_tests"] = [
            str(item)[:200] for item in list(value.get("created_tests", []))[:20]
        ]
        value["last_test_result"] = str(value.get("last_test_result", ""))[:6000]
        value["last_observation"] = str(value.get("last_observation", ""))[:6000]
        value["recent_actions"] = [
            str(item)[:100] for item in list(value.get("recent_actions", []))[-12:]
        ]
        return value

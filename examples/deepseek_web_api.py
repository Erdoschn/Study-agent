"""OpenAI-compatible API that uses only the local DeepSeek Web Model.

This adapter intentionally bypasses StudyAgent, ModelRouter, Teacher, search, and
all API providers. Requests go directly to BrowserModel, which drives the normal
visible DeepSeek Web UI through a local persistent browser profile.

Run:
    python examples/deepseek_web_api.py

Environment:
    STUDY_AGENT_WEB_HOST=127.0.0.1
    STUDY_AGENT_WEB_PORT=8001
    STUDY_AGENT_WEB_API_KEY=optional-secret
"""

from __future__ import annotations

import json
import os
import time
import uuid
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coder.browser_session import CoderBrowserSession
from core.http_security import (
    HTTPRequestError,
    RequestBodyTooLarge,
    is_allowed_local_browser_origin,
    read_limited_request_body,
    require_auth_for_non_loopback_bind,
)


MODEL_ID = "deepseek-web"
HOST = os.getenv("STUDY_AGENT_WEB_HOST", "127.0.0.1")
PORT = int(os.getenv("STUDY_AGENT_WEB_PORT", "8001"))
API_KEY = os.getenv("STUDY_AGENT_WEB_API_KEY", "").strip()
MAX_REQUEST_BODY_BYTES = 4 * 1024 * 1024
RUN_LOCK = threading.Lock()


def _require_auth_for_non_loopback_bind() -> None:
    require_auth_for_non_loopback_bind(
        HOST,
        API_KEY,
        api_name="DeepSeek Web Model API",
        api_key_env="STUDY_AGENT_WEB_API_KEY",
        host_env="STUDY_AGENT_WEB_HOST",
    )


def _allowed_browser_origin(origin: str) -> bool:
    return is_allowed_local_browser_origin(origin, PORT)


def completion(answer: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
    }


def sse_chunk(
    *,
    completion_id: str,
    content: str | None = None,
    finish: str | None = None,
) -> bytes:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
        }],
    }
    return (
        "data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


class WebModelServer(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "StudyAgentWebModelAPI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def _origin_allowed(self) -> bool:
        return _allowed_browser_origin(self.headers.get("Origin", ""))

    def _reject_disallowed_origin(self) -> bool:
        if self._origin_allowed():
            return False
        self._send_json(
            {"error": {"message": "不允许此浏览器来源访问本地模型 API."}},
            403,
        )
        return True

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _read_json(self) -> dict[str, Any]:
        raw = read_limited_request_body(
            self,
            MAX_REQUEST_BODY_BYTES,
            empty_body=b"{}",
        )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求 JSON 格式无效。") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    @staticmethod
    def _extract_prompt(messages: list[Any]) -> str:
        for item in reversed(messages):
            if (
                isinstance(item, dict)
                and item.get("role") == "user"
                and str(item.get("content", "")).strip()
            ):
                return str(item["content"]).strip()
        raise ValueError("messages 中没有可用的 user 内容")

    def do_GET(self) -> None:
        if self._reject_disallowed_origin():
            return
        if not self._authorized():
            self._send_json(
                {"error": {"message": "Unauthorized", "type": "authentication_error"}},
                401,
            )
            return

        if self.path == "/health":
            self._send_json({"status": "ok", "model": MODEL_ID})
            return

        if self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "Erdoschn/Study-agent",
                }],
            })
            return

        self._send_json(
            {"error": {"message": "Not found", "type": "invalid_request_error"}},
            404,
        )

    def do_POST(self) -> None:
        if self._reject_disallowed_origin():
            return
        if not self._authorized():
            self._send_json(
                {"error": {"message": "Unauthorized", "type": "authentication_error"}},
                401,
            )
            return

        if self.path != "/v1/chat/completions":
            self._send_json(
                {"error": {"message": "Not found", "type": "invalid_request_error"}},
                404,
            )
            return

        if not RUN_LOCK.acquire(blocking=False):
            self._send_json(
                {"error": {"message": "模型正在处理另一个请求，请稍后重试。"}},
                409,
            )
            return

        try:
            self.server.web_model.begin_run()
            request = self._read_json()
            messages = request.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError("messages 必须是数组")
            if len(messages) > 256:
                raise ValueError("messages 数量超过 256 条。")

            prompt = self._extract_prompt(messages)
            system = next(
                (
                    str(item.get("content", "")).strip()
                    for item in messages
                    if isinstance(item, dict)
                    and item.get("role") == "system"
                    and str(item.get("content", "")).strip()
                ),
                "",
            )

            client = self.server.web_model
            answer = client.generate(system, prompt)

            if request.get("stream", False):
                self._stream_answer(answer)
            else:
                self._send_json(completion(answer))

        except (HTTPRequestError, ValueError) as exc:
            status = 413 if isinstance(exc, RequestBodyTooLarge) else 400
            self._send_json({
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            }, status)
        except Exception as exc:
            reference = uuid.uuid4().hex[:8]
            print(
                f"[DeepSeekWebAPI] request failed ref={reference} "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
            self._send_json({
                "error": {
                    "message": f"服务器内部错误（参考编号 {reference}）。",
                    "type": "server_error",
                }
            }, 500)
        finally:
            RUN_LOCK.release()

    def _stream_answer(self, answer: str) -> None:
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("Content-Encoding", "identity")
        self.end_headers()

        self.wfile.write(
            sse_chunk(completion_id=completion_id, content=answer)
        )
        self.wfile.write(
            sse_chunk(completion_id=completion_id, finish="stop")
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    # Check exposure policy before opening a browser or allocating resources.
    _require_auth_for_non_loopback_bind()
    web_model = CoderBrowserSession(
        model=MODEL_ID,
        user_data_dir=".study-agent-browser",
        timeout=180,
        cleanup_after_generate=False,
        reuse_chat=True,
        min_send_interval_seconds=5.0,
    )
    server = WebModelServer((HOST, PORT), Handler)
    server.web_model = web_model

    print(f"DeepSeek Web Model API listening on http://{HOST}:{PORT}")
    print(f"OpenAI endpoint: http://{HOST}:{PORT}/v1")
    print(f"Model: {MODEL_ID}")
    print("Backend: visible local browser -> DeepSeek Web")
    print("StudyAgent/ModelRouter/Teacher/Search: disabled")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDeepSeek Web Model API stopped.")
    finally:
        web_model.close()
        server.server_close()


if __name__ == "__main__":
    main()

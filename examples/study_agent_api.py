"""OpenAI-compatible HTTP adapter for Study Agent.

This module is deliberately an outer shell: the existing StudyAgent core is not
modified. It exposes /v1/models and /v1/chat/completions for Open WebUI.

Run:
    python examples/study_agent_api.py

Environment:
    STUDY_AGENT_HOST=127.0.0.1
    STUDY_AGENT_PORT=8000
    STUDY_AGENT_API_KEY=optional-secret
    STUDY_AGENT_ALLOW_PAID=0
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_config, setup_debug
from core import AgentReasoner, ModelClientFactory, ModelRegistry, ModelRouter, StudyAgent, Teacher, ToolExecutor
from core.__debug__ import debug
from tools.search import ArxivSearchProvider, SearchRouter, WikipediaSearchProvider


MODEL_ID = "study-agent"
HOST = os.getenv("STUDY_AGENT_HOST", "127.0.0.1")
PORT = int(os.getenv("STUDY_AGENT_PORT", "8000"))
API_KEY = os.getenv("STUDY_AGENT_API_KEY", "").strip()
ALLOW_PAID = os.getenv("STUDY_AGENT_ALLOW_PAID", "0").strip().lower() in {"1", "true", "yes", "on"}

_RUN_LOCK = threading.Lock()


def build_search_router() -> SearchRouter:
    router = SearchRouter()
    router.register(ArxivSearchProvider())
    router.register(WikipediaSearchProvider())
    return router


def build_agent(config: dict[str, Any]) -> StudyAgent:
    registry = ModelRegistry(config)
    model_router = ModelRouter(registry)
    model_factory = ModelClientFactory(config)
    reasoner = AgentReasoner(
        model_router=model_router,
        model_factory=model_factory,
        allow_paid=ALLOW_PAID,
    )
    teacher = Teacher(
        model_router=model_router,
        model_factory=model_factory,
        allow_paid=ALLOW_PAID,
    )
    return StudyAgent(
        reasoner=reasoner,
        teacher=teacher,
        tool_executor=ToolExecutor(search_router=build_search_router()),
        max_steps=None,
    )


def sse_event(data: dict[str, Any]) -> bytes:
    return (
        "data: "
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


def chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    finish: str | None = None,
    completion_id: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {
        "id": completion_id or "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class AgentHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "StudyAgentAPI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {API_KEY}"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:
        if not self._authorized():
            self._send_json({"error": {"message": "Unauthorized", "type": "authentication_error"}}, 401)
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
        self._send_json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)

    def do_POST(self) -> None:
        if not self._authorized():
            self._send_json({"error": {"message": "Unauthorized", "type": "authentication_error"}}, 401)
            return
        if self.path != "/v1/chat/completions":
            self._send_json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)

            return

        try:
            request = self._read_json()
            messages = request.get("messages", [])
            question = next(
                (
                    str(item.get("content", "")).strip()
                    for item in reversed(messages)
                    if isinstance(item, dict)
                    and item.get("role") == "user"
                    and str(item.get("content", "")).strip()
                ),
                "",
            )
            if not question:
                raise ValueError("messages 中没有可用的 user 内容")
            stream = bool(request.get("stream", True))
            if stream:
                self._stream_run(question)
            else:
                result = self._run_agent(question, lambda _: None)
                answer = str(result.final_answer or "").strip()
                if not answer:
                    raise RuntimeError("StudyAgent 完成但 final_answer 为空")
                self._send_json(self._completion(answer))
        except Exception as exc:
            # Streaming responses already sent their headers. _stream_run handles
            # its own SSE errors; only ordinary JSON requests reach this branch.
            self._send_json({
                "error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "server_error",
                }
            }, 500)

    def _completion(self, answer: str) -> dict[str, Any]:
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

    def _run_agent(self, question: str, emit) -> Any:
        events: list[str] = []

        def log_hook(module: str, message: str) -> None:
            text = f"[{module}] {message}"
            events.append(text)
            print(text, flush=True)
            emit(text)

        with _RUN_LOCK:
            original_log = debug.log
            debug.log = log_hook
            try:
                status = "🤔 Study Agent 正在分析任务..."
                print(status, flush=True)
                emit(status)
                result = self.server.agent.run(question)
                status = (
                    f"✓ Agent 完成：steps={result.step_count}, "
                    f"evidence={len(result.evidence)}, claims={len(result.claims)}"
                )
                print(status, flush=True)
                emit(status)
                return result
            finally:
                debug.log = original_log

    def _stream_run(self, question: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        sentinel = object()
        completion_id = "chatcmpl-" + uuid.uuid4().hex

        def emit(status: str) -> None:
            events.put(("status", status))

        def worker() -> None:
            try:
                result = self._run_agent(question, emit)
                events.put(("result", result))
            except Exception as exc:
                print(f"❌ Agent 执行失败：{type(exc).__name__}: {exc}", flush=True)
                events.put(("error", exc))
            finally:
                events.put(("done", sentinel))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            kind, value = events.get()
            if kind == "status":
                self.wfile.write(sse_event(chunk(reasoning=value, completion_id=completion_id)))
                self.wfile.flush()
            elif kind == "result":
                answer = str(value.final_answer or "").strip()
                print(f"✓ FINAL ANSWER → {len(answer)} chars", flush=True)
                if answer:
                    self.wfile.write(sse_event(
                        chunk(content=answer, completion_id=completion_id, role="assistant")
                    ))
                    self.wfile.write(sse_event(chunk(finish="stop", completion_id=completion_id)))
                else:
                    self.wfile.write(sse_event(chunk(
                        reasoning="❌ StudyAgent 完成但 final_answer 为空",
                        completion_id=completion_id,
                    )))
                    self.wfile.write(sse_event(chunk(
                        finish="stop",
                        completion_id=completion_id,
                    )))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            elif kind == "error":
                self.wfile.write(sse_event(chunk(
                    reasoning=f"❌ Agent 执行失败：{value}",
                    completion_id=completion_id,
                )))
                self.wfile.write(sse_event(chunk(
                    finish="stop",
                    completion_id=completion_id,
                )))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            elif kind == "done" and value is sentinel:
                break


def main() -> None:
    config = load_config()
    setup_debug(config)
    agent = build_agent(config)

    server = AgentHTTPServer((HOST, PORT), Handler)
    server.agent = agent
    print(f"Study Agent API listening on http://{HOST}:{PORT}")
    print(f"OpenAI endpoint: http://{HOST}:{PORT}/v1")
    print(f"Model: {MODEL_ID}")
    print(f"Paid models: {'allowed' if ALLOW_PAID else 'disabled'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStudy Agent API stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

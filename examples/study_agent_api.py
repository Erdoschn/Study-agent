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
        # Support clients that use either common reasoning field name.
        delta["reasoning_content"] = reasoning
        delta["reasoning"] = reasoning
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
    protocol_version = "HTTP/1.1"
    SSE_HEARTBEAT_SECONDS = 5.0
    SSE_STATUS_SECONDS = 15.0

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

            # Web UI helper prompts are routed here and never enter StudyAgent.
            # The core agent therefore remains independent of Open WebUI.
            ui_result = self._handle_ui_auxiliary_request(question, messages)
            if ui_result is not None:
                self._send_json(self._completion(ui_result))
                return

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

    @classmethod
    def _handle_ui_auxiliary_request(cls, question: str, messages: list[Any]) -> str | None:
        if cls._is_title_request(question):
            return json.dumps({"title": cls._build_title(question, messages)}, ensure_ascii=False)
        if cls._is_follow_up_request(question):
            return json.dumps({"follow_ups": cls._build_follow_ups(messages)}, ensure_ascii=False)
        return None

    @staticmethod
    def _is_title_request(question: str) -> bool:
        text = str(question or "").lower()
        markers = (
            "generate a concise title",
            "summarizing the chat history",
            '"title"',
            "2-4 words",
            "raw json object",
        )
        return sum(marker in text for marker in markers) >= 2

    @staticmethod
    def _is_follow_up_request(question: str) -> bool:
        text = str(question or "").lower()
        markers = (
            "suggest 3-5 relevant follow-up questions",
            '"follow_ups"',
            "based on the chat history",
            "response must be a json object with a",
        )
        return sum(marker in text for marker in markers) >= 2

    @staticmethod
    def _extract_title_topic(question: str, messages: list[Any]) -> str:
        import re
        match = re.search(r"<chat_history>\s*(.*?)(?:\s*</chat_history>|$)",
                          str(question or ""), flags=re.I | re.S)
        history = match.group(1) if match else ""
        users = re.findall(r"USER:\s*(.*?)(?=\s*ASSISTANT:|$)", history, flags=re.I | re.S)
        if users:
            return users[-1].strip()
        for item in reversed(messages):
            if (isinstance(item, dict) and item.get("role") == "user"
                    and str(item.get("content", "")).strip()
                    and not Handler._is_title_request(str(item.get("content", "")))):
                return str(item["content"]).strip()
        return ""

    @classmethod
    def _build_title(cls, question: str, messages: list[Any]) -> str:
        import re
        topic = cls._extract_title_topic(question, messages)
        topic = re.sub(r"^#+\s*", "", topic).strip()
        topic = re.sub(r"(是什么|是什么意思|怎么理解|如何理解|请问|帮我|能否|可以吗)\s*[？?。.!！]*$",
                       "", topic, flags=re.I).strip(" ：:，,。.!！？?")
        if not topic:
            return "Study Topic"
        if re.search(r"[\u3400-\u9fff]", topic):
            return topic[:12]
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", topic)
        return " ".join(words[:4]) if words else topic[:40].strip()

    @staticmethod
    def _build_follow_ups(messages: list[Any]) -> list[str]:
        """Return lightweight follow-ups without re-entering StudyAgent.

        Keep this deterministic so the UI helper never consumes another Agent
        run or another model call.
        """
        user_messages = [
            str(item.get("content", "")).strip()
            for item in messages
            if isinstance(item, dict)
            and item.get("role") == "user"
            and str(item.get("content", "")).strip()
        ]
        topic = user_messages[-2] if len(user_messages) >= 2 else (
            user_messages[-1] if user_messages else "这个问题"
        )
        if len(topic) > 80:
            topic = topic[:80].rstrip() + "..."

        return [
            f"如果我继续研究“{topic}”，下一步最值得深入哪个具体部分？",
            "如果我要自己实现一个最小版本，哪些组件是必须的，哪些可以先不做？",
            "怎么设计一个实验来比较不同模型、工具和 Harness 结构的效果？",
        ]

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
            # Full DebugTracer output is terminal-only. The frontend receives
            # only a small set of user-relevant progress events.
            text = f"[{module}] {message}"
            events.append(text)
            print(text, flush=True)

            if module == "AgentReasoner" and message.startswith("ACTION →"):
                action = message.split("→", 1)[1].strip()
                emit(f"🧠 下一步：{action}")
                return

            if module == "ToolExecutor" and message.startswith("ARGS →"):
                raw = message.split("→", 1)[1].strip()
                if raw.startswith("{") and "'query':" in raw:
                    import ast
                    try:
                        args = ast.literal_eval(raw)
                        query = str(args.get("query", "")).strip()
                        source = str(args.get("source", "auto")).strip() or "auto"
                        emit(f"🔎 搜索：{query}（{source}）")
                    except Exception:
                        emit("🔧 正在调用搜索工具...")
                elif raw.startswith("{"):
                    emit("🔧 正在调用工具...")
                return

            if module == "Teacher" and message.startswith("SUCCESS →"):
                emit("✍️ 正在整理最终教学回答...")
                return

            if module == "TaskAnalyzer" and message.startswith("RESULT →"):
                emit("🧩 任务分析完成，开始规划执行...")
                return

        # Emit before taking the process-wide debug lock so the frontend never
        # looks frozen while another request is finishing its run.
        status = "🤔 Study Agent 正在分析任务..."
        print(status, flush=True)
        emit(status)

        with _RUN_LOCK:
            original_log = debug.log
            debug.log = log_hook
            try:
                emit("🔄 Agent 已开始执行，正在分析 / 搜索 / 推理...")
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

    def _write_sse(self, payload: bytes) -> None:
        """Write one SSE frame as an HTTP/1.1 chunk and flush immediately."""
        header = f"{len(payload):X}\r\n".encode("ascii")
        self.wfile.write(header)
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _finish_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _stream_run(self, question: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        # [DONE] terminates the SSE stream; close the HTTP connection
        # after the terminal frame so simple clients can detect completion.
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Content-Encoding", "identity")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        sentinel = object()
        completion_id = "chatcmpl-" + uuid.uuid4().hex

        # _run_agent emits the initial status exactly once, before acquiring
        # the global run lock. Do not emit it here as well, or the UI will show
        # duplicate "正在分析任务" messages.
        self._write_sse(sse_event(chunk(role="assistant", completion_id=completion_id)))

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

        last_status_at = time.monotonic()
        try:
            while True:
                try:
                    kind, value = events.get(timeout=self.SSE_HEARTBEAT_SECONDS)
                except queue.Empty:
                    # Keep the SSE connection active during long model/search calls.
                    self._write_sse(b": keep-alive\n\n")
                    continue

                if kind == "status":
                    self._write_sse(sse_event(chunk(
                        reasoning=str(value).rstrip() + "\n\n",
                        completion_id=completion_id,
                    )))
                elif kind == "result":
                    answer = str(value.final_answer or "").strip()
                    print(f"✓ FINAL ANSWER → {len(answer)} chars", flush=True)
                    if answer:
                        self._write_sse(sse_event(
                            chunk(content=answer, role="assistant", completion_id=completion_id)
                        ))
                        self._write_sse(sse_event(chunk(finish="stop", completion_id=completion_id)))
                    else:
                        self._write_sse(sse_event(chunk(
                            reasoning="❌ StudyAgent 完成但 final_answer 为空\n\n",
                            completion_id=completion_id,
                        )))
                        self._write_sse(sse_event(chunk(
                            finish="stop",
                            completion_id=completion_id,
                        )))
                    self._write_sse(b"data: [DONE]\n\n")
                elif kind == "error":
                    self._write_sse(sse_event(chunk(
                        reasoning=f"❌ Agent 执行失败：{value}\n\n",
                        completion_id=completion_id,
                    )))
                    self._write_sse(sse_event(chunk(
                        finish="stop",
                        completion_id=completion_id,
                    )))
                    self._write_sse(b"data: [DONE]\n\n")
                elif kind == "done" and value is sentinel:
                    break
        except (BrokenPipeError, ConnectionResetError):
            print("⚠️ SSE 客户端已断开连接", flush=True)
        finally:
            try:
                self._finish_chunked()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True


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

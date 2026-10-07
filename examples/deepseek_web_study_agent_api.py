"""OpenAI-compatible API: StudyAgent with DeepSeek Web browser model only."""
from __future__ import annotations
import json, os, queue, threading, time, uuid
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

MODEL_ID = "deepseek-web"
HOST = os.getenv("STUDY_AGENT_WEB_HOST", "127.0.0.1")
PORT = int(os.getenv("STUDY_AGENT_WEB_PORT", "8001"))
API_KEY = os.getenv("STUDY_AGENT_WEB_API_KEY", "").strip()
_RUN_LOCK = threading.Lock()


def build_search_router() -> SearchRouter:
    router = SearchRouter()
    router.register(ArxivSearchProvider())
    router.register(WikipediaSearchProvider())
    return router


def build_web_only_config(config: dict[str, Any]) -> dict[str, Any]:
    providers = config.get("providers", {})
    models = config.get("models", {})
    provider = providers.get("deepseek_web")
    model = models.get(MODEL_ID)

    # providers.json may predate the browser-model addition. Keep this API
    # self-contained: use the BrowserModel defaults when those entries are absent.
    if not isinstance(provider, dict):
        provider = {
            "type": "browser",
            "url": "https://chat.deepseek.com/",
            "browser_channel": "msedge",
            "user_data_dir": ".study-agent-browser",
            "timeout": 180,
            "enabled": True,
        }
    if not isinstance(model, dict):
        model = {
            "provider": "deepseek_web",
            "model": MODEL_ID,
            "enabled": True,
            "paid": False,
        }

    return {
        "providers": {"deepseek_web": dict(provider, enabled=True)},
        "models": {MODEL_ID: dict(model, provider="deepseek_web", model=MODEL_ID, enabled=True, paid=False)},
    }


def build_agent(config: dict[str, Any]) -> StudyAgent:
    isolated = build_web_only_config(config)
    registry = ModelRegistry(isolated)
    router = ModelRouter(registry)
    factory = ModelClientFactory(isolated)
    reasoner = AgentReasoner(router, factory, allow_paid=False)
    teacher = Teacher(router, factory, allow_paid=False)
    return StudyAgent(
        reasoner=reasoner,
        teacher=teacher,
        tool_executor=ToolExecutor(search_router=build_search_router()),
        max_steps=None,
        knowledge_graph_path=ROOT / "data" / "knowledge_graph.sqlite3",
    )


def sse(data: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode()


def chunk(cid: str, content=None, reasoning=None, finish=None, role=None) -> dict[str, Any]:
    delta = {}
    if role is not None: delta["role"] = role
    if content is not None: delta["content"] = content
    if reasoning is not None: delta["reasoning_content"] = reasoning
    return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": MODEL_ID, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class Server(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "StudyAgentWebAPI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        return not API_KEY or self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def _json(self, payload: dict[str, Any], status=200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(n).decode() if n else "{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    @staticmethod
    def _prompt(messages: list[Any]) -> str:
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "user" and str(item.get("content", "")).strip():
                return str(item["content"]).strip()
        raise ValueError("messages 中没有可用的 user 内容")

    def do_GET(self) -> None:
        if not self._authorized():
            self._json({"error": {"message": "Unauthorized", "type": "authentication_error"}}, 401)
            return
        if self.path == "/health":
            self._json({"status": "ok", "model": MODEL_ID, "backend": "browser", "agent": "StudyAgent"})
            return
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                "created": int(time.time()), "owned_by": "Erdoschn/Study-agent"}]})
            return
        self._json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)

    def do_POST(self) -> None:
        if not self._authorized():
            self._json({"error": {"message": "Unauthorized", "type": "authentication_error"}}, 401)
            return
        if self.path != "/v1/chat/completions":
            self._json({"error": {"message": "Not found", "type": "invalid_request_error"}}, 404)
            return
        try:
            request = self._body()
            messages = request.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError("messages 必须是数组")
            question = self._prompt(messages)
            if request.get("stream", True):
                self._stream(question)
            else:
                result = self._run(question, lambda _: None)
                answer = str(result.final_answer or "").strip()
                if not answer:
                    raise RuntimeError("StudyAgent 完成但 final_answer 为空")
                self._json({"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
                    "created": int(time.time()), "model": MODEL_ID,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]})
        except Exception as exc:
            self._json({"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}}, 500)

    def _run(self, question: str, emit) -> Any:
        def log_hook(module: str, message: str) -> None:
            print(f"[{module}] {message}", flush=True)
            if module == "AgentReasoner" and message.startswith("ACTION →"):
                emit("🧠 下一步：" + message.split("→", 1)[1].strip())
            elif module == "ToolExecutor" and message.startswith("ARGS →"):
                emit("🔧 正在调用工具...")
            elif module == "Teacher" and message.startswith("SUCCESS →"):
                emit("✍️ 正在整理最终教学回答...")
            elif module == "TaskAnalyzer" and message.startswith("RESULT →"):
                emit("🧩 任务分析完成，开始规划执行...")

        emit("🤔 Study Agent 正在分析任务...")
        with _RUN_LOCK:
            old = debug.log
            debug.log = log_hook
            try:
                emit("🔄 Agent 已开始执行，正在分析 / 搜索 / 推理...")
                result = self.server.agent.run(question)
                emit(f"✓ Agent 完成：steps={result.step_count}, evidence={len(result.evidence)}, claims={len(result.claims)}")
                return result
            finally:
                debug.log = old

    def _write(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()

    def _stream(self, question: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        events = queue.Queue()
        sentinel = object()
        cid = "chatcmpl-" + uuid.uuid4().hex
        self._write(sse(chunk(cid, role="assistant")))

        def emit(status: str) -> None:
            events.put(("status", status))

        def worker() -> None:
            try:
                events.put(("result", self._run(question, emit)))
            except Exception as exc:
                events.put(("error", exc))
            finally:
                events.put(("done", sentinel))

        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                kind, value = events.get()
                if kind == "status":
                    self._write(sse(chunk(cid, reasoning=str(value) + "\n")))
                elif kind == "result":
                    answer = str(value.final_answer or "").strip()
                    if answer:
                        self._write(sse(chunk(cid, content=answer, role="assistant")))
                    self._write(sse(chunk(cid, finish="stop")))
                    self._write(b"data: [DONE]\n\n")
                elif kind == "error":
                    self._write(sse(chunk(cid, reasoning=f"❌ Agent 执行失败：{value}\n")))
                    self._write(sse(chunk(cid, finish="stop")))
                    self._write(b"data: [DONE]\n\n")
                elif kind == "done" and value is sentinel:
                    break
        except (BrokenPipeError, ConnectionResetError):
            print("⚠️ SSE 客户端已断开连接", flush=True)


def main() -> None:
    config = load_config()
    setup_debug(config)
    agent = build_agent(config)
    server = Server((HOST, PORT), Handler)
    server.agent = agent
    print(f"Study Agent Web API listening on http://{HOST}:{PORT}")
    print(f"OpenAI endpoint: http://{HOST}:{PORT}/v1")
    print("Model: deepseek-web")
    print("Agent: StudyAgent")
    print("LLM candidates: deepseek-web ONLY")
    print("Backend: visible local browser -> DeepSeek Web")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStudy Agent Web API stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

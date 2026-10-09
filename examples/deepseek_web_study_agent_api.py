"""OpenAI-compatible API and lightweight browser UI for StudyAgent + DeepSeek Web."""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_config, setup_debug
from core import AgentReasoner, ModelClientFactory, ModelRegistry, ModelRouter, StudyAgent, Teacher, ToolExecutor
from core.__debug__ import debug
from core.cancellation import RunCancelled
from tools.search import ArxivSearchProvider, SearchRouter, WikipediaSearchProvider

MODEL_ID = "deepseek-web"
HOST = os.getenv("STUDY_AGENT_WEB_HOST", "127.0.0.1")
PORT = int(os.getenv("STUDY_AGENT_WEB_PORT", "8001"))
API_KEY = os.getenv("STUDY_AGENT_WEB_API_KEY", "").strip()
_RUN_LOCK = threading.Lock()


def _run_agent_with_cancellation(agent, question: str, cancellation_event: threading.Event):
    """Pass cancellation to current StudyAgent implementations without breaking test doubles."""
    import inspect

    run = agent.run
    try:
        parameters = inspect.signature(run).parameters.values()
        supports_cancellation = any(
            parameter.name == "cancellation_event"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_cancellation = False
    if supports_cancellation:
        return run(question, cancellation_event=cancellation_event)
    return run(question)


def build_search_router() -> SearchRouter:
    router = SearchRouter()
    router.register(ArxivSearchProvider())
    router.register(WikipediaSearchProvider())
    return router


def build_web_only_config(config: dict[str, Any]) -> dict[str, Any]:
    providers = config.get("providers", {})
    models = config.get("models", {})
    provider = providers.get("deepseek_web") if isinstance(providers, dict) else None
    model = models.get(MODEL_ID) if isinstance(models, dict) else None
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
        model = {"provider": "deepseek_web", "model": MODEL_ID, "enabled": True, "paid": False}
    return {
        "providers": {"deepseek_web": dict(provider, enabled=True)},
        "models": {MODEL_ID: dict(model, provider="deepseek_web", model=MODEL_ID, enabled=True, paid=False)},
    }


def build_agent(config: dict[str, Any]) -> StudyAgent:
    isolated = build_web_only_config(config)
    registry = ModelRegistry(isolated)
    # This endpoint intentionally has exactly one LLM candidate. Core routing
    # cooldowns would turn one transient browser error into "no available model"
    # for the rest of the same request, so the isolated registry retries the
    # browser backend instead of hiding the only candidate.
    registry.BASE_COOLDOWN_SECONDS = 0.0
    registry.MAX_COOLDOWN_SECONDS = 0.0
    registry.PROVIDER_COOLDOWN_SECONDS = 0.0
    router = ModelRouter(registry)
    factory = ModelClientFactory(isolated)
    reasoner = AgentReasoner(router, factory, allow_paid=False)
    teacher = Teacher(router, factory, allow_paid=False)
    agent = StudyAgent(
        reasoner=reasoner,
        teacher=teacher,
        tool_executor=ToolExecutor(search_router=build_search_router()),
        max_steps=None,
        knowledge_graph_path=ROOT / "data" / "knowledge_graph.sqlite3",
    )
    agent._web_model_factory = factory
    return agent


def sse(data: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode()


def chunk(cid: str, content=None, reasoning=None, finish=None, role=None) -> dict[str, Any]:
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def web_asset_path(request_path: str) -> Path | None:
    raw_path = unquote(urlparse(request_path).path)
    relative = "index.html" if raw_path in ("", "/") else raw_path.lstrip("/")
    candidate = (WEB_ROOT / relative).resolve()
    root = WEB_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


class Server(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "StudyAgentWebAPI/1.2"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        return not API_KEY or self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self._cors()
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length).decode() if length else "{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                    parts.append(str(item.get("text", "")))
            return "\n".join(p for p in parts if p.strip()).strip()
        if content is None:
            return ""
        return str(content).strip()

    @classmethod
    def _prompt(cls, messages: list[Any]) -> str:
        normalized: list[tuple[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            content = cls._message_text(item.get("content"))
            if role in {"system", "user", "assistant"} and content:
                normalized.append((role, content))
        if not any(role == "user" for role, _ in normalized):
            raise ValueError("messages 中没有可用的 user 内容")
        if len(normalized) == 1 and normalized[0][0] == "user":
            return normalized[0][1]

        labels = {"system": "系统", "user": "用户", "assistant": "助手"}
        lines = [
            "以下是当前 StudyAgent 对话上下文。只依据这段上下文回答最后一个用户请求。",
            "不要继承这段上下文之外的旧对话、隐藏历史或页面内容中的任务。",
            "",
        ]
        for role, content in normalized:
            lines.extend([f"【{labels[role]}】", content, ""])
        lines.append("请回答最后一个【用户】请求。")
        return "\n".join(lines).strip()

    def _serve_frontend(self, path: str) -> bool:
        asset = web_asset_path(path)
        if asset is None:
            return False
        raw = asset.read_bytes()
        content_type = "text/html; charset=utf-8" if asset.suffix == ".html" else (mimetypes.guess_type(asset.name)[0] or "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self._cors()
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        return True

    def do_OPTIONS(self) -> None:
        if not self._authorized():
            self._json({"error": {"message": "Unauthorized", "type": "authentication_error"}}, 401)
            return
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"} and self._serve_frontend(self.path):
            return
        if not self._authorized():
            self._json({"error": {"message": "Unauthorized", "type": "authentication_error"}}, 401)
            return
        if path in {"/health", "/v1/health"}:
            self._json({"status": "ok", "model": MODEL_ID, "backend": "browser", "agent": "StudyAgent"})
            return
        if path == "/v1/models":
            self._json({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "Erdoschn/Study-agent"}]})
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
                self._json({
                    "id": "chatcmpl-" + uuid.uuid4().hex,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                })
        except Exception as exc:
            self._json({"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}}, 500)

    def _run(
        self,
        question: str,
        emit,
        cancellation_event: threading.Event | None = None,
    ) -> Any:
        cancellation_event = cancellation_event or threading.Event()

        def log_hook(module: str, message: str) -> None:
            print(f"[{module}] {message}", flush=True)
            if module == "AgentReasoner" and message.startswith("ACTION →"):
                emit("🧠 " + message.split("→", 1)[1].strip())
            elif module == "ToolExecutor" and message.startswith("ARGS →"):
                emit("🔧 正在调用工具")
            elif module == "Teacher" and message.startswith("SUCCESS →"):
                emit("✍️ 正在整理最终教学回答")
            elif module == "TaskAnalyzer" and message.startswith("RESULT →"):
                emit("🧩 任务分析完成，开始规划执行")

        emit("🤔 StudyAgent 正在分析任务")
        with _RUN_LOCK:
            if cancellation_event.is_set():
                raise RunCancelled("客户端已断开，Study Agent 请求已取消。")
            old = debug.log
            factory = getattr(self.server.agent, "_web_model_factory", None)
            begin_run = getattr(factory, "begin_run", None)
            if callable(begin_run):
                begin_run(cancellation_event)
            debug.log = log_hook
            try:
                emit("🔄 已进入 Agent Loop：分析 / 搜索 / 推理 / 教学")
                result = _run_agent_with_cancellation(self.server.agent, question, cancellation_event)
                emit(f"✓ Agent 完成：steps={result.step_count}, evidence={len(result.evidence)}, claims={len(result.claims)}")
                return result
            finally:
                debug.log = old
                end_run = getattr(factory, "end_run", None)
                if callable(end_run):
                    end_run(cancellation_event)
                close_factory = getattr(factory, "close", None)
                if callable(close_factory):
                    close_factory()

    def _write(self, data: bytes) -> None:
        self.wfile.write(data)
        self.wfile.flush()

    def _stream(self, question: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

        events = queue.Queue()
        sentinel = object()
        cancellation_event = threading.Event()
        cid = "chatcmpl-" + uuid.uuid4().hex
        self._write(sse(chunk(cid, role="assistant")))

        def emit(status: str) -> None:
            events.put(("status", status))

        def worker() -> None:
            try:
                events.put(("result", self._run(question, emit, cancellation_event)))
            except Exception as exc:
                events.put(("error", exc))
            finally:
                events.put(("done", sentinel))

        threading.Thread(target=worker, daemon=True).start()
        finished = False
        try:
            while True:
                try:
                    kind, value = events.get(timeout=15)
                except queue.Empty:
                    self._write(b": ping\n\n")
                    continue
                if kind == "status":
                    self._write(sse(chunk(cid, reasoning=str(value) + "\n")))
                elif kind == "result":
                    answer = str(value.final_answer or "").strip()
                    if answer:
                        self._write(sse(chunk(cid, content=answer, role="assistant")))
                    self._write(sse(chunk(cid, finish="stop")))
                    self._write(b"data: [DONE]\n\n")
                    finished = True
                elif kind == "error":
                    self._write(sse(chunk(cid, reasoning=f"❌ Agent 执行失败：{value}\n")))
                    self._write(sse(chunk(cid, finish="stop")))
                    self._write(b"data: [DONE]\n\n")
                    finished = True
                elif kind == "done" and value is sentinel:
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            cancellation_event.set()
            print("⚠️ SSE 客户端已断开连接，已请求取消当前浏览器模型调用", flush=True)
        finally:
            if not finished:
                cancellation_event.set()
                print("[StudyAgentWebAPI] 客户端提前结束 SSE 等待", flush=True)


def main() -> None:
    config = load_config()
    setup_debug(config)
    agent = build_agent(config)
    server = Server((HOST, PORT), Handler)
    server.agent = agent
    print(f"Study Agent Web API listening on http://{HOST}:{PORT}")
    print(f"OpenAI endpoint: http://{HOST}:{PORT}/v1")
    print(f"Web UI: http://{HOST}:{PORT}/")
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

r"""Local web UI and API for the browser-backed Coder Agent.

Run:
    python examples/coder_web_api.py

Environment:
    CODER_WEB_HOST=127.0.0.1
    CODER_WEB_PORT=8002
    CODER_WORKSPACE=D:\Coder_workspace
    CODER_WEB_API_KEY=optional-secret
    CODER_MAX_RUNTIME_SECONDS=1800
    CODER_UPLOAD_MAX_BYTES=1048576
    CODER_STUDY_AGENT_URL=http://127.0.0.1:8000/internal/study/ask
    CODER_STUDY_AGENT_API_KEY=optional-secret
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import threading
from email import policy
from email.parser import BytesParser
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coder import CoderAgent
from coder.filesystem import WorkspaceFS, WorkspaceSecurityError
from coder.reasoner import CoderReasoner
from coder.memory import CoderMemoryStore
from coder.browser_session import CoderBrowserSession
from core.__debug__ import debug
from core.cancellation import RunCancelled, raise_if_cancelled
from core.web_model import BrowserModel


HOST = os.getenv("CODER_WEB_HOST", "127.0.0.1")
PORT = int(os.getenv("CODER_WEB_PORT", "8002"))
API_KEY = os.getenv("CODER_WEB_API_KEY", "").strip()
WORKSPACE = os.getenv("CODER_WORKSPACE", r"D:\Coder_workspace")
MAX_RUNTIME_SECONDS = max(
    30.0, float(os.getenv("CODER_MAX_RUNTIME_SECONDS", "1800"))
)
UPLOAD_MAX_BYTES = min(
    WorkspaceFS.MAX_FILE_BYTES,
    max(1, int(os.getenv("CODER_UPLOAD_MAX_BYTES", str(WorkspaceFS.MAX_FILE_BYTES)))),
)
FEEDBACK_MAX_BYTES = 16 * 1024
RUN_LOCK = threading.Lock()
RUNS_LOCK = threading.Lock()
ACTIVE_RUNS: dict[str, "CoderRunControl"] = {}
PROJECT_NAME_MAX = 64
PROJECT_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
PROJECT_SLUG_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+){0,7}\b", re.IGNORECASE)
PROJECT_NAME_FILLER = {
    "project", "project-name", "name", "coder", "code", "here", "the", "new",
}
def _coder_debug_hook(module: str, message: str) -> None:
    print(f"[{module}] {message}", flush=True)


PROJECT_NAME_PROMPT = (
    "You are naming a coding project. Based only on the user's coding task, "
    "choose one short, specific project slug. Return ONLY the slug, using "
    "lowercase ASCII letters, digits, and hyphens. Prefer 1-4 meaningful words, "
    "maximum 40 characters. No explanation, no Markdown, no quotes. "
    "Examples: simple-calculator, score-analyzer, transformer-demo."
)

def _project_name(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    text = PROJECT_NAME_RE.sub("-", text).strip("-.")[:PROJECT_NAME_MAX]
    return text or "coder-project"


def _project_slug_from_llm(value: str) -> str:
    text = str(value or "").replace(chr(96), " ").strip()
    matches = PROJECT_SLUG_RE.findall(text)
    for candidate in reversed(matches):
        candidate = re.sub(r"[-_]+", "-", candidate.casefold()).strip("-")
        if candidate and candidate not in PROJECT_NAME_FILLER and len(candidate) <= 40:
            return _project_name(candidate)
    return ""


def _new_coder_browser_model() -> BrowserModel:
    return BrowserModel(
        model="deepseek-web",
        user_data_dir=".coder-browser",
        timeout=max(1, int(os.getenv("CODER_WEB_BROWSER_TIMEOUT", "180"))),
        cleanup_after_generate=False,
        reuse_chat=True,
        min_send_interval_seconds=5.0,
        debug_mode=False,
    )


def _prewarm_coder_browser() -> CoderBrowserSession | None:
    """Open the Coder browser profile on its dedicated Playwright thread."""
    try:
        session = CoderBrowserSession(
            user_data_dir=".coder-browser",
            reuse_chat=True,
            min_send_interval_seconds=5.0,
            debug_sink=_coder_debug_hook,
        )
        debug.log(
            "CoderWebAPI",
            f"BROWSER PREWARM → model={session.model}, profile={session.user_data_dir}",
        )
        return session
    except Exception as exc:
        debug.log(
            "CoderWebAPI",
            f"BROWSER PREWARM FAILED → {type(exc).__name__}: {exc}",
        )
        return None


def _llm_project_name(task: str, model: BrowserModel | None = None) -> str:
    owns_model = model is None
    if model is None:
        model = BrowserModel(
            model="deepseek-web",
            user_data_dir=".coder-browser",
            timeout=60,
            cleanup_after_generate=True,
            reuse_chat=False,
            min_send_interval_seconds=5.0,
            debug_mode=False,
        )
    try:
        raw = model.generate(
            PROJECT_NAME_PROMPT,
            f"User coding task:\n{str(task or '').strip()}",
            json_mode=False,
        )
        name = _project_slug_from_llm(raw)
        if name:
            debug.log("CoderWebAPI", f"PROJECT NAME → {name}")
            return name
        debug.log("CoderWebAPI", "PROJECT NAME → model output unusable; using coder-project")
    except RunCancelled:
        raise
    except Exception as exc:
        debug.log(
            "CoderWebAPI",
            f"PROJECT NAME FALLBACK → {type(exc).__name__}: {exc}",
        )
    finally:
        if owns_model:
            try:
                model.close()
            except Exception:
                pass
    return "coder-project"

def _project_root(name: str) -> Path:
    candidate = Path(name)
    if candidate.name != name or name in {".", ".."} or "/" in name or "\\" in name:
        raise WorkspaceSecurityError("项目名无效。")
    root = WorkspaceFS(WORKSPACE).root
    target = (root / name).resolve()
    target.relative_to(root)
    return target

def _list_projects() -> list[dict]:
    root = WorkspaceFS(WORKSPACE).root
    projects = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        if path.is_symlink():
            continue
        try:
            fs = WorkspaceFS(path)
            files = fs.list_files()
        except WorkspaceSecurityError:
            continue
        projects.append({"name": path.name, "files": files, "file_count": len(files)})
    return sorted(projects, key=lambda x: x["name"].casefold())

def _resolve_existing_project(name: str) -> str:
    project = str(name or "").strip()
    if not project:
        raise ValueError("必须指定已有项目。")
    target = _project_root(project)
    if not target.is_dir():
        raise ValueError(f"指定项目不存在：{project}")
    return project


def _ensure_project(
    name: str | None = None,
    task: str = "",
    *,
    unique_if_requested: bool = False,
) -> str:
    requested = str(name).strip() if name is not None else ""
    base = _project_name(requested or task)
    root = WorkspaceFS(WORKSPACE).root
    project = base
    if not requested or unique_if_requested:
        index = 2
        while (root / project).exists():
            project = f"{base}-{index}"
            index += 1
    target = _project_root(project)
    if target == root:
        raise WorkspaceSecurityError("项目目录不能是 workspace 根目录。")
    target.mkdir(parents=True, exist_ok=True)
    return project


def _cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")


def _sse(payload: dict) -> bytes:
    return (
        "data: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


def _step_summary(step) -> str:
    action = str(step.action or "").upper()
    args = step.arguments if isinstance(step.arguments, dict) else {}
    path = str(args.get("path", "")).strip()
    labels = {
        "PLAN": "制定开发计划",
        "SEARCH": "搜索资料",
        "LIST_FILES": "查看工作区文件",
        "READ_FILE": f"读取 {path}" if path else "读取文件",
        "WRITE_FILE": f"整体写入 {path}" if path else "整体写入文件",
        "WRITE_NOTEBOOK": f"更新 Notebook {path}" if path else "更新 Notebook",
        "PATCH_FILE": f"局部修改 {path}" if path else "局部修改文件",
        "CREATE_TEST": f"新增测试 {path}" if path else "新增测试",
        "RUN_PYTHON": f"运行 {str(args.get('script_path', '')).strip()}" if args.get("script_path") else "运行 Python",
        "RUN_PYTEST": "运行 pytest",
        "ASK_STUDY_AGENT": "咨询 Study Agent",
        "READ_DIFF": "检查代码修改",
        "VERIFY_GOAL": "验证任务完成条件",
        "NEW_CHAT": "切换新的模型会话",
        "FINISH": "检查并尝试完成任务",
    }
    summary = labels.get(action, action or "执行操作")
    if action == "SEARCH" and args.get("query"):
        summary += "：" + str(args["query"])[:100]
    if action == "RUN_PYTEST":
        paths = args.get("paths") or []
        if paths:
            summary += "：" + ", ".join(str(x) for x in paths[:5])
    if action == "ASK_STUDY_AGENT" and args.get("question"):
        summary += "：" + str(args["question"]).replace("\n", " ")[:100]
    if not step.success:
        error = str(step.error or "").strip()
        summary += "；失败" + (f"：{error[:180]}" if error else "")
    return summary


def _step_payload(step) -> dict:
    observation = step.observation
    if isinstance(observation, dict):
        safe_observation = {}
        for key, value in observation.items():
            if key in {"content", "stdout", "stderr"}:
                text = str(value or "")
                safe_observation[key] = (
                    text[:2000] + "…"
                    if len(text) > 2000
                    else text
                )
            else:
                safe_observation[key] = value
        observation = safe_observation
    elif isinstance(observation, str):
        observation = observation[:2000] + ("…" if len(observation) > 2000 else "")
    else:
        observation = str(observation)[:2000]
    return {
        "step_id": step.step_id,
        "action": step.action,
        "summary": _step_summary(step),
        "arguments": step.arguments,
        "observation": observation,
        "success": step.success,
        "error": step.error,
    }


def _state_payload(state) -> dict:
    return {
        "request": state.request,
        "finished": state.finished,
        "goal_verified": state.goal_verified,
        "cancelled": bool(getattr(state, "cancelled", False)),
        "error": state.error,
        "summary": state.summary,
        "step_count": state.step_count,
        "chat_resets": state.chat_resets,
        "modified_files": sorted(state.modified_files),
        "created_tests": sorted(state.created_tests),
        "metrics": dict(state.metrics),
        "last_test_result": state.last_test_result,
        "steps": [_step_payload(step) for step in state.steps[-50:]],
    }


def _frontend_path(request_path: str) -> Path | None:
    raw = urlparse(request_path).path
    relative = "coder.html" if raw in {"", "/", "/coder", "/coder/"} else raw.lstrip("/")
    candidate = (WEB_ROOT / relative).resolve()
    root = WEB_ROOT.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def parse_multipart_upload(content_type: str, body: bytes) -> tuple[str, bytes]:
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("上传必须使用 multipart/form-data。")
    raw = (
        "Content-Type: " + content_type + "\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=policy.default).parsebytes(raw)
    if not message.is_multipart():
        raise ValueError("multipart/form-data 解析失败。")
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="content-disposition") != "file":
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if not filename:
            raise ValueError("上传文件缺少文件名。")
        return str(filename), bytes(payload)
    raise ValueError("请求中没有 file 字段。")


class CoderRunControl:
    def __init__(self, run_id: str, cancellation_event: threading.Event | None = None):
        self.run_id = run_id
        self.cancel_event = cancellation_event or threading.Event()
        self.finished = False

    def cancel(self) -> None:
        self.cancel_event.set()


class CoderServer(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "StudyAgentCoderWebAPI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _authorized(self) -> bool:
        return not API_KEY or self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def _json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        _cors_headers(self)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def _body_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ValueError("JSON 请求超过 64 KiB。")
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _upload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("上传请求为空。")
        if length > UPLOAD_MAX_BYTES + 64 * 1024:
            raise WorkspaceSecurityError("上传请求超过安全大小上限。")
        body = self.rfile.read(length)
        filename, data = parse_multipart_upload(
            self.headers.get("Content-Type", ""),
            body,
        )
        if len(data) > UPLOAD_MAX_BYTES:
            raise WorkspaceSecurityError("上传文件超过安全大小上限。")
        filename = filename.replace("\\", "/")
        if not filename:
            raise ValueError("上传文件缺少文件名。")

        project = str(parse_qs(urlparse(self.path).query).get("project", [""])[0]).strip()
        if not project:
            raise ValueError("上传文件前必须选择项目。")
        workspace = WorkspaceFS(_project_root(project))
        workspace.write_uploaded_text(filename, data)
        return {"status": "uploaded", "project": project, "path": filename, "bytes": len(data)}

    def _serve_frontend(self) -> bool:
        asset = _frontend_path(self.path)
        if asset is None:
            return False
        raw = asset.read_bytes()
        content_type = (
            "text/html; charset=utf-8"
            if asset.suffix.casefold() == ".html"
            else mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        )
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        _cors_headers(self)
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        return True

    def do_OPTIONS(self) -> None:
        if not self._authorized():
            self._json({"error": {"message": "Unauthorized"}}, 401)
            return
        self.send_response(204)
        _cors_headers(self)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/coder", "/coder/"} and self._serve_frontend():
            return
        if not self._authorized():
            self._json({"error": {"message": "Unauthorized"}}, 401)
            return

        if path == "/health":
            self._json({
                "status": "ok",
                "service": "coder",
                "workspace": str(self.server.workspace.root),
                "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                "upload_max_bytes": UPLOAD_MAX_BYTES,
            })
            return

        if path == "/v1/coder/projects":
            self._json({
                "workspace": str(self.server.workspace.root),
                "projects": _list_projects(),
            })
            return

        if path == "/v1/coder/history":
            params = parse_qs(urlparse(self.path).query)
            try:
                limit = int(params.get("limit", ["30"])[0])
            except ValueError:
                limit = 30
            memory = CoderMemoryStore(self.server.workspace.root)
            data = memory.load()
            limit = max(1, min(limit, memory.MAX_RUNS))
            self._json({
                "workspace": str(self.server.workspace.root),
                "runs": list(reversed(data["runs"][-limit:])),
                "knowledge": {
                    "strategies": list(reversed(data["strategies"][-20:])),
                    "experiences": list(reversed(data["experiences"][-20:])),
                    "technologies": list(reversed(data["technologies"][-20:])),
                },
                "knowledge_graph": memory.knowledge_graph.snapshot(60),
            })
            return

        if path == "/v1/coder/files":
            params = parse_qs(urlparse(self.path).query)
            project = str(params.get("project", [""])[0]).strip()
            if not project:
                raise ValueError("必须指定 project。")
            workspace = WorkspaceFS(_project_root(project))
            self._json({
                "workspace": str(workspace.root),
                "project": project,
                "files": workspace.list_files(),
            })
            return

        if path == "/v1/coder/file":
            params = parse_qs(urlparse(self.path).query)
            project = str(params.get("project", [""])[0]).strip()
            requested = str(params.get("path", [""])[0])
            workspace = WorkspaceFS(_project_root(project))
            content = workspace.read_text(requested)
            self._json({"project": project, "path": requested, "content": content})
            return

        self._json(
            {"error": {"message": "Not found", "type": "invalid_request_error"}},
            404,
        )

    def do_POST(self) -> None:
        if not self._authorized():
            self._json({"error": {"message": "Unauthorized"}}, 401)
            return

        path = urlparse(self.path).path
        try:
            if path == "/v1/coder/projects":
                if RUN_LOCK.locked():
                    self._json({"error": {"message": "Coder 正在运行，暂时不能创建项目。"}}, 409)
                    return
                request = self._body_json()
                unique = bool(request.get("unique", False))
                name = _ensure_project(
                    request.get("name"),
                    unique_if_requested=unique,
                )
                self._json({"status": "created", "project": name}, 201)
                return

            if path == "/v1/coder/files":
                if RUN_LOCK.locked():
                    self._json(
                        {"error": {"message": "Coder 正在运行，暂时不能修改 workspace。"}},
                        409,
                    )
                    return
                self._json(self._upload(), 201)
                return

            if path == "/v1/coder/feedback":
                request = self._body_json()
                run_id = str(request.get("run_id", "")).strip()
                feedback = str(request.get("feedback", "")).strip()
                if not run_id:
                    raise ValueError("缺少 run_id。")
                if not feedback:
                    raise ValueError("用户评价不能为空。")
                if len(feedback.encode("utf-8")) > FEEDBACK_MAX_BYTES:
                    raise WorkspaceSecurityError("用户评价超过安全长度上限。")
                memory = CoderMemoryStore(self.server.workspace.root)
                saved = memory.record_feedback(run_id=run_id, feedback=feedback)
                if saved is None:
                    self._json({"error": {"message": "找不到对应的 Coder 运行记录。"}}, 404)
                    return
                self._json({"status": "saved", "feedback": saved}, 201)
                return

            if path == "/v1/coder/cancel":
                request = self._body_json()
                run_id = str(request.get("run_id", "")).strip()
                if not run_id:
                    raise ValueError("缺少 run_id。")
                with RUNS_LOCK:
                    control = ACTIVE_RUNS.get(run_id)
                if control is None:
                    self._json({"status": "not_found", "run_id": run_id}, 404)
                    return
                control.cancel()
                self._json({"status": "cancelling", "run_id": run_id}, 202)
                return

            if path != "/v1/coder/run":
                self._json(
                    {"error": {"message": "Not found", "type": "invalid_request_error"}},
                    404,
                )
                return

            request = self._body_json()
            task = str(request.get("request", "")).strip()
            if not task:
                raise ValueError("Coder request 不能为空。")
            if len(task.encode("utf-8")) > WorkspaceFS.MAX_FILE_BYTES:
                raise WorkspaceSecurityError("Coder request 超过安全长度上限。")
            requested_project = request.get("project")
            if requested_project is not None and str(requested_project).strip():
                requested_project = _resolve_existing_project(str(requested_project))
            if not RUN_LOCK.acquire(blocking=False):
                self._json(
                    {"error": {"message": "已有 Coder 任务正在运行。"}},
                    409,
                )
                return

            control = None
            try:
                browser_model = getattr(self.server, "coder_browser_model", None)
                cancellation_event = (
                    browser_model.cancellation_event
                    if browser_model is not None
                    else threading.Event()
                )
                if browser_model is not None:
                    browser_model.begin_run()
                control = CoderRunControl(
                    "coder-" + uuid.uuid4().hex,
                    cancellation_event=cancellation_event,
                )
                with RUNS_LOCK:
                    ACTIVE_RUNS[control.run_id] = control
                self._stream_run(task, requested_project, control)
            except Exception:
                if control is not None:
                    control.cancel()
                    with RUNS_LOCK:
                        ACTIVE_RUNS.pop(control.run_id, None)
                    control.finished = True
                RUN_LOCK.release()
                raise
        except Exception as exc:
            self._json({
                "error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "server_error",
                }
            }, 400 if isinstance(exc, (ValueError, WorkspaceSecurityError)) else 500)

    def _stream_run(
        self,
        task: str,
        project: str | None,
        control: CoderRunControl,
    ) -> None:
        worker_started = False
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            _cors_headers(self)
            self.end_headers()
        except Exception:
            control.cancel()
            with RUNS_LOCK:
                ACTIVE_RUNS.pop(control.run_id, None)
                control.finished = True
            RUN_LOCK.release()
            raise

        events: queue.Queue = queue.Queue()
        done = object()
        cid = control.run_id

        def emit(event: dict) -> None:
            events.put(event)

        def worker() -> None:
            actual_project = str(project).strip() if project is not None else ""
            try:
                browser_model = getattr(self.server, "coder_browser_model", None)
                if actual_project:
                    actual_project = _resolve_existing_project(actual_project)
                else:
                    raise_if_cancelled(control.cancel_event)
                if not actual_project:
                    generated_name = _llm_project_name(task, browser_model)
                    actual_project = _ensure_project(
                        generated_name,
                        task,
                        unique_if_requested=True,
                    )
                emit({"type": "named", "project": actual_project})
                if browser_model is not None:
                    # Naming may have used a chat; every coding task starts in
                    # its own Coder conversation while reusing the same browser.
                    browser_model.new_chat()
                memory = CoderMemoryStore(self.server.workspace.root)
                recent_feedback = memory.recent_feedback()
                emit({
                    "type": "started",
                    "project": actual_project,
                    "runtime_timeout_seconds": MAX_RUNTIME_SECONDS,
                })
                agent = self.server.build_agent(
                    actual_project,
                    browser_model,
                    recent_user_feedback=recent_feedback,
                    cancellation_event=control.cancel_event,
                )
                debug.bind_thread(_coder_debug_hook)
                try:
                    def agent_event_hook(event: dict) -> None:
                        # The Web API already emits its own STARTED event carrying
                        # project/runtime metadata. CoderAgent's lifecycle STARTED
                        # event has a different schema, so do not forward it.
                        if event.get("type") != "started":
                            emit(event)

                    result = agent.run(task, event_hook=agent_event_hook)
                    entry = memory.record_run(project=actual_project, state=result)
                    events.put({"type": "result", "state": result, "memory": entry})
                finally:
                    debug.clear_thread_binding()
            except RunCancelled:
                events.put({
                    "type": "cancelled",
                    "project": actual_project,
                    "state": None,
                })
            except Exception as exc:
                events.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                with RUNS_LOCK:
                    ACTIVE_RUNS.pop(control.run_id, None)
                    control.finished = True
                RUN_LOCK.release()

        try:
            # Send the first SSE frame before starting browser/model work. This
            # makes the UI visibly enter RUNNING even if Playwright startup or
            # DeepSeek login takes time.
            self.wfile.write(_sse({
                "id": cid,
                "type": "preparing",
                "message": (
                    "正在准备已有项目…"
                    if project
                    else "正在根据任务让 LLM 自动命名新项目…"
                ),
            }))
            self.wfile.flush()
            threading.Thread(target=worker, daemon=True).start()
            worker_started = True
            while True:
                try:
                    event = events.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                kind = event.get("type")
                if kind == "preparing":
                    self.wfile.write(_sse({
                        "id": cid,
                        "type": "preparing",
                        "message": event.get("message", "正在准备…"),
                    }))
                    self.wfile.flush()
                elif kind == "named":
                    self.wfile.write(_sse({
                        "id": cid,
                        "type": "named",
                        "project": event["project"],
                    }))
                    self.wfile.flush()
                elif kind == "started":
                    self.wfile.write(_sse({
                        "id": cid,
                        "type": "started",
                        "runtime_timeout_seconds": event["runtime_timeout_seconds"],
                        "project": event["project"],
                    }))
                    self.wfile.flush()
                elif kind == "step":
                    self.wfile.write(_sse({
                        "id": cid,
                        "type": "step",
                        "step": _step_payload(event["step"]),
                    }))
                    self.wfile.flush()
                elif kind == "cancelled":
                    self.wfile.write(_sse({
                        "id": cid,
                        "type": "cancelled",
                        "project": event.get("project", ""),
                        "state": _state_payload(event["state"]) if event.get("state") is not None else None,
                    }))
                    self.wfile.flush()
                elif kind == "finished":
                    self.wfile.write(_sse({
                        "id": cid,
                        "type": "finished",
                        "state": _state_payload(event["state"]),
                    }))
                    self.wfile.flush()
                elif kind == "result":
                    self.wfile.write(_sse({
                        "id": cid,
                        "type": "result",
                        "state": _state_payload(event["state"]),
                        "memory": event.get("memory", {}),
                    }))
                    self.wfile.flush()
                elif kind == "error":
                    self.wfile.write(_sse({
                        "id": cid,
                        "type": "error",
                        "error": event["error"],
                    }))
                    self.wfile.flush()
                elif kind == "done":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
        except (BrokenPipeError, ConnectionResetError):
            control.cancel()
            print("[CoderWebAPI] SSE client disconnected; cancelling backend task.", flush=True)
        finally:
            # A disconnected browser must cancel the backend task. The worker
            # remains the owner of RUN_LOCK once it has started.
            if not control.finished:
                control.cancel()
            if not worker_started and not control.finished:
                with RUNS_LOCK:
                    ACTIVE_RUNS.pop(control.run_id, None)
                control.finished = True
                RUN_LOCK.release()


def build_agent(
    project: str,
    browser_model: CoderBrowserSession | None = None,
    *,
    recent_user_feedback: list[dict] | None = None,
    cancellation_event: threading.Event | None = None,
) -> CoderAgent:
    reasoner = (
        CoderReasoner(
            model=browser_model,
            reuse_chat=True,
            min_send_interval_seconds=5.0,
            debug_mode=False,
            recent_user_feedback=recent_user_feedback,
            cancellation_event=cancellation_event,
        )
        if browser_model is not None
        else None
    )
    return CoderAgent(
        workspace=str(_project_root(project)),
        reasoner=reasoner,
        max_runtime_seconds=MAX_RUNTIME_SECONDS,
        debug_mode=False,
        reuse_chat=True,
        min_send_interval_seconds=5.0,
        close_model_on_run=browser_model is None,
        cancellation_event=cancellation_event,
    )


def main() -> None:
    workspace = WorkspaceFS(WORKSPACE)
    server = CoderServer((HOST, PORT), Handler)
    server.workspace = workspace
    server.coder_browser_model = _prewarm_coder_browser()
    server.build_agent = build_agent
    print(f"Coder Web API listening on http://{HOST}:{PORT}")
    if server.coder_browser_model is not None:
        print("Coder browser: .coder-browser 已启动，可在 Edge 中登录 DeepSeek。")
    else:
        print("Coder browser: 启动预热失败，将在首次任务时重试。")
    print(f"Coder UI: http://{HOST}:{PORT}/coder")
    print(f"Workspace root: {workspace.root}")
    print(f"Runtime timeout: {MAX_RUNTIME_SECONDS}s")
    print(f"Upload limit: {UPLOAD_MAX_BYTES} bytes")
    print("Projects: each Coder task is isolated in its own workspace subdirectory.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCoder Web API stopped.")
    finally:
        with RUNS_LOCK:
            active_controls = list(ACTIVE_RUNS.values())
        for control in active_controls:
            control.cancel()
        try:
            browser_model = getattr(server, "coder_browser_model", None)
            if browser_model is not None:
                browser_model.close()
        except Exception as exc:
            debug.log("CoderWebAPI", f"BROWSER CLOSE SKIP → {type(exc).__name__}: {exc}")
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())

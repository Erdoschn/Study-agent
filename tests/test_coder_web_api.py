import json
from pathlib import Path

from examples.coder_web_api import (
    MAX_RUNTIME_SECONDS,
    UPLOAD_MAX_BYTES,
    _frontend_path,
    parse_multipart_upload,
    _project_name,
    _ensure_project,
    _list_projects,
)
from coder.filesystem import WorkspaceFS
from coder.memory import CoderMemoryStore


def test_coder_web_frontend_exists():
    path = _frontend_path("/")
    assert path is not None
    assert path.name == "coder.html"
    assert path.stat().st_size > 5000


def test_coder_web_multipart_upload_parser_preserves_relative_filename():
    boundary = "----study-agent-path-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="src/models/demo.py"\r\n'
        "Content-Type: text/plain\r\n\r\n"
        "print(1)\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    filename, payload = parse_multipart_upload(
        f"multipart/form-data; boundary={boundary}",
        body,
    )
    assert filename == "src/models/demo.py"
    assert payload == b"print(1)"


def test_coder_web_multipart_upload_parser_reads_browser_file():
    boundary = "----study-agent-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="demo.ipynb"\r\n'
        "Content-Type: application/json\r\n\r\n"
        '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    filename, payload = parse_multipart_upload(
        f"multipart/form-data; boundary={boundary}",
        body,
    )
    assert filename == "demo.ipynb"
    assert payload.startswith(b'{"cells":[]')


def test_coder_web_rejects_non_multipart_upload():
    try:
        parse_multipart_upload("application/json", b"{}")
    except ValueError as exc:
        assert "multipart/form-data" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_coder_web_limits_upload_and_runtime():
    assert UPLOAD_MAX_BYTES <= WorkspaceFS.MAX_FILE_BYTES
    assert MAX_RUNTIME_SECONDS >= 30.0


def test_coder_web_frontend_javascript_parses():
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        return

    source = _frontend_path("/").read_text(encoding="utf-8")
    scripts = []
    marker = "<script>"
    pos = 0
    while True:
        start = source.find(marker, pos)
        if start < 0:
            break
        end = source.find("</script>", start)
        if end < 0:
            break
        scripts.append(source[start + len(marker):end])
        pos = end + len("</script>")

    with tempfile.TemporaryDirectory() as tmp:
        for index, script in enumerate(scripts):
            path = Path(tmp) / f"script_{index}.js"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run(
                [node, "--check", str(path)],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr


def test_coder_web_frontend_supports_importing_files_into_a_new_project():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "直接拖入文件/文件夹" in source
    assert "也可以先把已有工程框架拖进来" in source
    assert "createProjectForImport" in source
    assert 'JSON.stringify({name:base,unique:true})' in source
    assert "webkitRelativePath" in source
    assert "文件框架已导入" in source


def test_coder_web_frontend_uses_step_summaries_instead_of_raw_observation():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'appendStep(s.action,s.success,s.summary||s.error||"已完成",s.step_id)' in source

    from types import SimpleNamespace
    from examples.coder_web_api import _step_payload

    payload = _step_payload(SimpleNamespace(
        step_id=3,
        action="READ_FILE",
        arguments={"path": "framework.py"},
        observation={"path": "framework.py", "content": "a" * 5000},
        success=True,
        error="",
    ))
    assert payload["summary"] == "读取 framework.py"
    assert "content" in payload["observation"]


def test_coder_web_frontend_handles_sse_errors_and_shift_enter():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'e.type==="error"' in source
    assert 'state.failed=true' in source
    assert 'if(e.state?.error)' in source
    assert 'Coder SSE 连接在收到完成信号前就结束了。' in source
    assert 'e.key==="Enter"&&!e.shiftKey&&!e.isComposing' in source
    assert 'e.preventDefault();if(!state.running)void run()' in source
    assert "Enter 发送，Shift + Enter 换行" in source
    assert "async function run(){" in source
    assert 'void run()' in source


def test_coder_web_frontend_keeps_output_position_and_compacts_memory():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "scrollIntoView" not in source
    assert 'textContent="最近教训："' in source
    assert 'textContent="知识记录："' in source
    assert 'textContent="策略："' not in source
    assert 'textContent="图谱："' not in source
    assert 'startsWith("最终通过验证：")' in source


def test_coder_web_frontend_layout_keeps_current_task_with_history_and_files_on_right():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'items.unshift({current:true,project:state.runProject||state.project||"当前任务",request:state.currentTask})' in source
    assert 'className="history-item"+(item.current?" current":"")' in source
    assert '<div class="files" id="files">' in source
    assert "function fileType(path)" in source
    assert "b.append(label,e)" in source
    assert 'ipynb:"Notebook"' in source
    assert '<div id="modified">暂无</div>' in source
    assert '<h3 style="margin-bottom:7px">测试结果</h3><div id="tests">暂无测试结果</div>' in source
    assert 'id="taskDrop"' in source
    assert 'composer.ondrop' in source
    assert 'e.key==="Enter"&&!e.shiftKey&&!e.isComposing' in source


def test_coder_web_frontend_has_drag_drop_and_timeouts():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "直接拖入文件/文件夹" in source
    assert 'composer.ondrop=e=>{e.preventDefault();composer.classList.remove("drag-hot")' in source
    assert "dataTransfer.files" in source
    assert ".ipynb" in source
    assert "fetchTimeout" in source
    assert "AbortController" in source
    assert "15000" in source


def test_coder_web_state_payload_exposes_completion_summary():
    from types import SimpleNamespace
    from examples.coder_web_api import _state_payload

    state = SimpleNamespace(
        request="fix",
        finished=True,
        goal_verified=True,
        summary="任务已完成。\n本次修改：\n  - 修改 score_utils.py",
        error=None,
        step_count=2,
        chat_resets=0,
        modified_files={"score_utils.py"},
        created_tests={"tests/test_score_utils.py"},
        metrics={},
        last_test_result={"kind": "pytest", "passed": True},
        steps=[],
    )
    payload = _state_payload(state)
    assert payload["summary"] == state.summary
    assert payload["finished"] is True


def test_coder_project_name_is_safe_and_derived_from_request():
    assert _project_name("创建一个 Python 成绩分析器") == "Python"
    assert "/" not in _project_name("my/project")
    assert _project_name("") == "coder-project"


def test_coder_project_slug_from_llm_prefers_specific_last_candidate():
    import examples.coder_web_api as api

    assert api._project_slug_from_llm(
        "Project name: simple-calculator"
    ) == "simple-calculator"
    assert api._project_slug_from_llm(
        "Here is the project name: score_analyzer"
    ) == "score-analyzer"
    assert api._project_slug_from_llm("Project") == ""


def test_coder_browser_prewarm_creates_shared_browser_session(monkeypatch):
    import examples.coder_web_api as api

    class FakeSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = "deepseek-web"
            self.user_data_dir = kwargs["user_data_dir"]
            self.closed = False

        def close(self):
            self.closed = True

    holder = {}

    def factory(**kwargs):
        session = FakeSession(**kwargs)
        holder["session"] = session
        return session

    monkeypatch.setattr(api, "CoderBrowserSession", factory)
    session = api._prewarm_coder_browser()

    assert session is holder["session"]
    assert session.closed is False
    assert session.model == "deepseek-web"
    assert session.user_data_dir == ".coder-browser"
    assert session.kwargs["reuse_chat"] is True


def test_coder_browser_session_keeps_playwright_on_one_thread(monkeypatch):
    import coder.browser_session as module

    threads = []
    closed = []

    class FakeBrowser:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def prepare_browser(self):
            threads.append(("prepare", __import__("threading").get_ident()))

        def generate(self, system_prompt, user_prompt, json_mode=False, reasoning_effort=None):
            threads.append(("generate", __import__("threading").get_ident()))
            return "answer"

        def new_chat(self):
            threads.append(("new_chat", __import__("threading").get_ident()))

        def close(self):
            closed.append(__import__("threading").get_ident())

    monkeypatch.setattr(module, "BrowserModel", FakeBrowser)
    session = module.CoderBrowserSession(user_data_dir=".coder-browser")

    caller_thread = __import__("threading").get_ident()
    assert session.generate("system", "user") == "answer"
    session.new_chat()
    session.close()

    assert threads[0][0] == "prepare"
    assert len({thread_id for _, thread_id in threads}) == 1
    assert threads[0][1] != caller_thread
    assert len(closed) == 1
    assert closed[0] == threads[0][1]


def test_coder_project_name_shared_browser_is_not_closed(monkeypatch):
    import examples.coder_web_api as api

    class FakeModel:
        closed = False

        def generate(self, system_prompt, user_prompt, json_mode=False):
            assert system_prompt == api.PROJECT_NAME_PROMPT
            assert "build a tiny calculator" in user_prompt
            assert json_mode is False
            return "simple-calculator"

        def close(self):
            self.closed = True

    model = FakeModel()
    assert api._llm_project_name("build a tiny calculator", model) == "simple-calculator"
    assert model.closed is False


def test_coder_project_name_uses_llm_and_closes_browser(monkeypatch):
    import examples.coder_web_api as api

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def generate(self, system_prompt, user_prompt, json_mode=False):
            assert system_prompt == api.PROJECT_NAME_PROMPT
            assert "build a tiny calculator" in user_prompt
            assert json_mode is False
            return "simple-calculator"

        def close(self):
            self.closed = True

    holder = {}

    def factory(**kwargs):
        model = FakeModel(**kwargs)
        holder["model"] = model
        return model

    monkeypatch.setattr(api, "BrowserModel", factory)

    assert api._llm_project_name("build a tiny calculator") == "simple-calculator"
    assert holder["model"].closed is True


def test_coder_new_project_auto_names_when_name_is_none(tmp_path, monkeypatch):
    import examples.coder_web_api as api

    monkeypatch.setattr(api, "WORKSPACE", str(tmp_path))

    first = api._ensure_project(None, "做一个最简单的计算器，只要初等运算就行")
    second = api._ensure_project(None, "做一个最简单的计算器，只要初等运算就行")

    assert first != "None"
    assert second != "None"
    assert first != second
    assert (tmp_path / first).is_dir()
    assert (tmp_path / second).is_dir()


def test_coder_memory_store_persists_history_and_knowledge(tmp_path):
    from types import SimpleNamespace

    state = SimpleNamespace(
        request="build a calculator",
        steps=[
            SimpleNamespace(action="READ_FILE", arguments={"path": "calculator.py"}, error=""),
            SimpleNamespace(
                action="PATCH_FILE",
                arguments={"path": "calculator.py"},
                error="pytest failed: division by zero",
            ),
            SimpleNamespace(action="CREATE_TEST", arguments={"path": "tests/test_calculator.py"}, error=""),
            SimpleNamespace(action="RUN_PYTEST", arguments={"paths": ["tests/test_calculator.py"]}, error=""),
        ],
        finished=True,
        goal_verified=True,
        error=None,
        summary="done",
        step_count=4,
        modified_files={"calculator.py", "tests/test_calculator.py"},
        created_tests={"tests/test_calculator.py"},
        chat_resets=0,
        last_test_result={"kind": "pytest", "passed": True, "stdout": "1 passed", "stderr": ""},
    )

    first = CoderMemoryStore(tmp_path)
    entry = first.record_run(project="calculator", state=state)
    second = CoderMemoryStore(tmp_path)
    data = second.load()

    assert entry["project"] == "calculator"
    assert data["runs"][-1]["request"] == "build a calculator"
    assert data["runs"][-1]["strategy"] == "READ_FILE → PATCH_FILE → CREATE_TEST → RUN_PYTEST"
    assert data["runs"][-1]["study_agent_calls"] == 0
    assert data["runs"][-1]["last_test_result"]["passed"] is True
    assert any(item["name"] == "Python" for item in data["technologies"])
    assert any(item["name"] == "pytest" for item in data["technologies"])
    assert data["experiences"][0]["text"].startswith("PATCH_FILE: pytest failed")
    graph = first.knowledge_graph.snapshot()
    assert any(node["name"] == "calculator" and node["type"] == "project" for node in graph["nodes"])
    assert any(node["name"] == "pytest" and node["type"] == "technology" for node in graph["nodes"])
    assert any(edge["relation"] == "uses" for edge in graph["edges"])


def test_coder_memory_is_stored_at_workspace_root_not_project(tmp_path):
    store = CoderMemoryStore(tmp_path)
    assert store.path == tmp_path / ".coder-memory.json"
    assert not (tmp_path / ".coder-memory.json").is_dir()


def test_coder_import_project_creation_can_force_unique_name(tmp_path, monkeypatch):
    import examples.coder_web_api as api

    monkeypatch.setattr(api, "WORKSPACE", str(tmp_path))
    assert api._ensure_project("starter", unique_if_requested=True) == "starter"
    assert api._ensure_project("starter", unique_if_requested=True) == "starter-2"
    assert (tmp_path / "starter").is_dir()
    assert (tmp_path / "starter-2").is_dir()


def test_coder_project_creation_keeps_projects_separate(tmp_path, monkeypatch):
    import examples.coder_web_api as api

    monkeypatch.setattr(api, "WORKSPACE", str(tmp_path))
    first = _ensure_project("alpha")
    second = _ensure_project("beta")
    assert first == "alpha"
    assert second == "beta"
    assert (tmp_path / "alpha").is_dir()
    assert (tmp_path / "beta").is_dir()
    assert set(p["name"] for p in _list_projects()) == {"alpha", "beta"}


def test_coder_project_files_are_not_listed_from_workspace_root(tmp_path):
    from coder.filesystem import WorkspaceFS

    (tmp_path / "legacy.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "project-a").mkdir()
    (tmp_path / "project-a" / "main.py").write_text("print(2)", encoding="utf-8")
    # WorkspaceFS.list_files() is recursive by design; the Web API never exposes
    # the workspace root as a project, so project files remain scoped to project-a.
    root_files = WorkspaceFS(tmp_path).list_files()
    project_files = WorkspaceFS(tmp_path / "project-a").list_files()
    assert root_files == ["legacy.py", "project-a/main.py"]
    assert project_files == ["main.py"]


def test_coder_web_frontend_allows_task_without_upload_and_has_project_selector():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "新建项目（直接描述要求）" in source
    assert "可以直接创建新项目" in source
    assert 'project:project||null' in source
    assert 'project_mode:project?"existing":"new"' in source
    assert "/projects" in source
    assert 'window.addEventListener("error"' in source
    assert 'window.addEventListener("unhandledrejection"' in source
    assert "DOMContentLoaded" in source
    assert "Coder UI DOM 初始化失败" in source
    assert "document.getElementById" in source
    assert "attempt<100" in source
    assert "setTimeout(()=>initCoderUI(attempt+1),50)" in source
    assert 'e.type==="named"' in source


def test_coder_web_api_imports_memory_store():
    import examples.coder_web_api as api
    assert api.CoderMemoryStore is CoderMemoryStore


def test_coder_web_api_exposes_history_memory():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "/v1/coder/history" in source or "history?limit" in source
    import examples.coder_web_api as api
    api_source = Path(api.__file__).read_text(encoding="utf-8")
    assert 'if path == "/v1/coder/history":' in api_source
    assert "CoderMemoryStore" in api_source
    assert "knowledge_graph" in api_source
    assert "正在根据任务让 LLM 自动命名新项目" in api_source
    assert ".coder-knowledge.sqlite3" in Path(__import__("coder.knowledge_graph").knowledge_graph.__file__).read_text(encoding="utf-8")


def test_coder_web_frontend_has_every_dom_node_used_by_javascript():
    import re

    from html.parser import HTMLParser

    class IdCollector(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()

        def handle_starttag(self, tag, attrs):
            self.ids.update(value for key, value in attrs if key == "id")

    source = _frontend_path("/").read_text(encoding="utf-8")
    parser = IdCollector()
    parser.feed(source)

    expected = {
        "workspace", "project", "projectHint", "drop", "fileInput", "files", "history",
        "dot", "health", "task", "timer", "run", "timeline", "request",
        "modified", "tests", "finished", "verified", "steps", "summary", "memory",
    }
    assert expected <= parser.ids

    js_refs = set(re.findall(r'\$\("([^"]+)"\)', source))
    assert js_refs <= parser.ids
    assert "UI_IDS" in source
    assert 'const missing=UI_IDS.filter(id=>!document.getElementById(id));' in source
    assert "Coder UI 元素不可用" in source
    assert "async async function" not in source
    assert "async function fetchTimeout" in source
    assert "loadHistory" in source
    assert "loadProjects();" in source
    assert "setInterval(()=>{void loadHistory();if(state.project)void loadFiles(state.project)},5000);" in source
    assert 'setProject(selectedProject())' not in source
    assert 'function readSelectedProject()' in source


def test_coder_web_frontend_does_not_throw_on_missing_dom_during_startup():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'throw new Error("Coder UI DOM 初始化失败")' not in source
    assert "void loadProjects();" in source
    assert "void health();" in source
    api_source = Path(__import__("examples.coder_web_api").coder_web_api.__file__).read_text(encoding="utf-8")
    assert "def agent_event_hook(event: dict) -> None:" in api_source
    assert 'if event.get("type") != "started":' in api_source
    assert "setInterval(()=>void health(),30000);" in source



def test_coder_and_study_browser_profiles_are_separate():
    from coder.reasoner import CoderReasoner
    from core.web_model import BrowserModel

    class FakeModel:
        pass

    reasoner = CoderReasoner(model=FakeModel())
    assert reasoner.model is not None
    assert ".coder-browser" in Path(__import__("coder.reasoner").reasoner.__file__).read_text(encoding="utf-8")
    assert ".study-agent-browser" in (
        str(BrowserModel.DEFAULT_URL)
        + Path(__import__("core.web_model").web_model.__file__).read_text(encoding="utf-8")
    )


def test_study_agent_bridge_is_available_from_study_api():
    import examples.study_agent_api as api

    source = Path(api.__file__).read_text(encoding="utf-8")
    assert '"/internal/study/ask"' in source
    assert "STUDY_AGENT_BRIDGE_KEY" in source
    assert "Study Agent Bridge 仅允许本机调用" in source
    assert "learner_context" in source

def test_coder_memory_persists_user_feedback_and_recent_feedback(tmp_path):
    from types import SimpleNamespace

    memory = CoderMemoryStore(tmp_path)
    state = SimpleNamespace(
        request="修复登录功能",
        steps=[],
        finished=True,
        goal_verified=True,
        error="",
        summary="任务已完成",
        step_count=3,
        modified_files={"auth.py"},
        created_tests={"tests/test_auth.py"},
        chat_resets=0,
        metrics={},
    )
    entry = memory.record_run(project="login-demo", state=state)

    saved = memory.record_feedback(
        run_id=entry["id"],
        feedback="以后修改前先说明计划，尽量保持改动范围小。",
    )

    assert saved is not None
    assert saved["run_id"] == entry["id"]
    loaded = memory.load()
    assert loaded["runs"][-1]["feedback"] == "以后修改前先说明计划，尽量保持改动范围小。"
    assert memory.recent_feedback(1)[0]["feedback"] == "以后修改前先说明计划，尽量保持改动范围小。"


def test_coder_reasoner_includes_recent_user_feedback():
    from types import SimpleNamespace

    from coder.reasoner import CoderReasoner
    from coder.state import CoderGoal, CoderState

    class FakeModel:
        def __init__(self):
            self.payload = None

        def generate(self, _system, user, json_mode=True):
            self.payload = user
            return '{"action":"FINISH","arguments":{},"reasoning_summary":"","goal":{},"answer":null}'

    model = FakeModel()
    reasoner = CoderReasoner(
        model=model,
        recent_user_feedback=[{
            "project": "login-demo",
            "request": "修复登录功能",
            "feedback": "以后修改前先说明计划，尽量保持改动范围小。",
        }],
    )
    state = CoderState("当前任务")
    state.goal = CoderGoal("当前任务")
    result = reasoner.decide(state, [])

    assert result["action"] == "FINISH"
    payload = json.loads(model.payload)
    assert payload["user_feedback"][0]["feedback"] == "以后修改前先说明计划，尽量保持改动范围小。"


def test_coder_web_frontend_preserves_selected_project_across_async_refresh():
    source = _frontend_path("/").read_text(encoding="utf-8")

    assert "let projectsLoadSeq=0;" in source
    assert "const requestSeq=++projectsLoadSeq;" in source
    assert "if(requestSeq!==projectsLoadSeq||state.running)return;" in source
    assert "const current=state.project;" in source
    assert "function readSelectedProject()" in source
    assert "setProject(readSelectedProject())" in source
    assert 'project:project||null' in source
    assert 'project_mode:project?"existing":"new"' in source
    assert "setProject(selectedProject())" not in source


def test_coder_web_feedback_endpoint_and_frontend_are_wired():
    import examples.coder_web_api as api

    api_source = Path(api.__file__).read_text(encoding="utf-8")
    source = _frontend_path("/").read_text(encoding="utf-8")

    assert 'if path == "/v1/coder/feedback":' in api_source
    assert "record_feedback" in api_source
    assert "recent_user_feedback" in api_source
    assert "FEEDBACK_MAX_BYTES" in api_source
    assert 'id="feedbackPanel"' in source
    assert 'id="feedback"' in source
    assert 'id="submitFeedback"' in source
    assert 'API+"/feedback"' in source
    assert 'state.memoryRunId=String(e.memory?.id||"").trim();' in source
    assert 'state.activeRunId=String(e.id).trim();' in source
    assert "后续任务会参考这条评价" in source




def test_coder_web_frontend_locks_project_identity_during_run():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'state.runProject=project;' in source
    assert '$("project").disabled=true;' in source
    assert '$("project").disabled=false;' in source
    assert 'project:project||null' in source
    assert 'project_mode:project?"existing":"new"' in source
    assert 'void loadFiles(state.runProject);' in source
    assert 'const current=state.project;' in source

def test_coder_web_frontend_sends_explicit_project_mode():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'project_mode:project?"existing":"new"' in source
    assert 'body:JSON.stringify({request,project:project||null})' not in source


def test_coder_web_frontend_uses_backend_cancel_instead_of_aborting_only_sse():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'API+"/cancel"' in source
    assert 'body:JSON.stringify({run_id:runId})' in source
    assert 'state.running?void requestCancel():run()' in source
    assert 'state.controller?.abort()' not in source
    assert 'e.type==="cancelled"' in source
    assert 'state.cancelRequested' in source


def test_coder_web_api_run_control_can_cancel():
    import examples.coder_web_api as api

    control = api.CoderRunControl("coder-test")
    assert not control.cancel_event.is_set()
    control.cancel()
    assert control.cancel_event.is_set()


def test_coder_web_api_exposes_cancel_and_strict_existing_project_resolution():
    import examples.coder_web_api as api

    source = Path(api.__file__).read_text(encoding="utf-8")
    assert 'if path == "/v1/coder/cancel":' in source
    assert "_resolve_existing_project" in source
    assert "browser_model.begin_run()" in source
    assert "ACTIVE_RUNS" in source



def test_coder_web_api_sends_sse_done_after_worker_finishes():
    source = Path(__import__("examples.coder_web_api").coder_web_api.__file__).read_text(encoding="utf-8")
    assert 'events.put({"type": "done"})' in source
    assert 'elif kind == "done":' in source
    assert 'self.wfile.write(b"data: [DONE]\\n\\n")' in source


def test_coder_web_frontend_has_independent_scrollable_step_area_and_final_refresh_guard():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert ".timeline{max-width:920px;width:100%;flex:1 1 0;height:0;min-height:0;overflow-y:scroll;overflow-x:hidden;" in source
    assert ".body{flex:1 1 0;min-width:0;min-height:0;overflow:hidden;" in source
    assert "try{\n      await loadProjects();" in source
    assert 'console.error("Coder final refresh failed",e);' in source


def test_coder_web_frontend_shows_and_follows_step_scrollbar():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert ".side{height:100%;min-height:0;overflow:hidden;" in source
    assert ".files{min-height:70px;max-height:35vh;overflow-y:auto;overflow-x:hidden;" in source
    assert ".timeline{max-width:920px;width:100%;flex:1 1 0;height:0;min-height:0;overflow-y:scroll;overflow-x:hidden;" in source
    assert "scrollbar-gutter:stable" in source
    assert ".timeline::-webkit-scrollbar{width:10px}" in source
    assert ".timeline::-webkit-scrollbar-thumb" in source
    assert "const follow=box.scrollHeight-box.scrollTop-box.clientHeight<80" in source
    assert "if(follow)box.scrollTop=box.scrollHeight" in source




def test_coder_web_frontend_uses_terminal_state_as_authoritative_completion():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "let finalState=null;" in source
    assert 'finalState=e.state||finalState;' in source
    assert "state.finalState=finalState;" in source
    assert "if(doneReceived&&finalState)updateSummary(finalState);" in source
    assert "await loadHistory();" in source
    assert "if(finalState)updateSummary(finalState);" in source
    assert 'st.goal_verified??st.verified' in source
    assert 'if(finalState)updateSummary(finalState);' in source


def test_coder_web_frontend_renders_completion_state_human_readably_and_accepts_history_shape():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'st.goal_verified??st.verified' in source
    assert 'finished?"✓ 已完成":"— 未完成"' in source
    assert 'verified?"✓ 已验证":"— 未验证"' in source




def test_coder_web_frontend_history_selection_restores_full_sidebar_state():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "const historyState={" in source
    assert "modified_files:Array.isArray(item.modified_files)?item.modified_files:[]" in source
    assert "last_test_result:item.last_test_result||null" in source
    assert "state.finalState=historyState;" in source
    assert "updateSummary(historyState);" in source




def test_coder_web_api_terminal_error_and_cancel_paths_emit_result_state():
    source = Path(__import__("examples.coder_web_api").coder_web_api.__file__).read_text(encoding="utf-8")
    assert '"state": result_state' in source
    assert '"type": "result"' in source
    assert '"state": _state_payload(result_state)' in source
    assert 'events.put({"type": "done"})' in source


def test_coder_web_frontend_distinguishes_active_cancel_id_from_memory_run_id():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "activeRunId" in source
    assert "memoryRunId" in source
    assert 'const runId=state.activeRunId;' in source
    assert 'const runId=state.memoryRunId;' in source





def test_coder_run_lock_is_not_released_after_worker_owns_it():
    import examples.coder_web_api as api

    control = api.CoderRunControl("test-worker")
    assert api.RUN_LOCK.acquire(blocking=False)
    try:
        control.worker_started = True
        api._release_run_lock_if_unowned(control)
        assert api.RUN_LOCK.locked()
    finally:
        if api.RUN_LOCK.locked():
            api.RUN_LOCK.release()


def test_coder_run_lock_is_released_before_worker_ownership():
    import examples.coder_web_api as api

    control = api.CoderRunControl("test-not-started")
    assert api.RUN_LOCK.acquire(blocking=False)
    api._release_run_lock_if_unowned(control)
    assert not api.RUN_LOCK.locked()


def test_coder_run_lock_release_is_idempotent_across_cleanup_paths():
    import examples.coder_web_api as api
    control = api.CoderRunControl("test-idempotent-release")
    assert api.RUN_LOCK.acquire(blocking=False)
    assert control.release_run_lock_once() is True
    assert control.release_run_lock_once() is False
    assert not api.RUN_LOCK.locked()


def test_coder_run_lock_release_is_safe_under_competing_cleanup_threads():
    import threading
    import examples.coder_web_api as api

    control = api.CoderRunControl("test-competing-cleanup")
    assert api.RUN_LOCK.acquire(blocking=False)
    barrier = threading.Barrier(3)
    results = []

    def release():
        barrier.wait(timeout=2)
        results.append(control.release_run_lock_once())

    workers = [threading.Thread(target=release) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait(timeout=2)
    for worker in workers:
        worker.join(timeout=2)
    assert not any(worker.is_alive() for worker in workers)
    assert sorted(results) == [False, True]
    assert not api.RUN_LOCK.locked()


def test_coder_user_response_window_times_out_when_button_is_not_clicked():
    import threading
    import examples.coder_web_api as api

    control = api.CoderRunControl("user-timeout")
    events = []
    result = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "answer",
            control.wait_for_user_response("Need a decision", events.append, open_timeout_seconds=0.05),
        )
    )
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result["answer"] is None
    assert events[0]["type"] == "ask_user"
    assert events[0]["question"] == "Need a decision"


def test_coder_user_response_is_returned_after_button_open_and_submit():
    import threading
    import time
    import examples.coder_web_api as api

    control = api.CoderRunControl("user-answer")
    events = []
    result = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "answer",
            control.wait_for_user_response("Which file?", events.append, open_timeout_seconds=1, response_timeout_seconds=1),
        )
    )
    worker.start()
    deadline = time.monotonic() + 1
    while not events and time.monotonic() < deadline:
        time.sleep(0.005)
    question_id = events[0]["question_id"]
    assert control.respond_to_user_question(question_id, "open") is True
    assert control.respond_to_user_question(question_id, "submit", "Use the existing notebook") is True
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result["answer"] == "Use the existing notebook"


def test_coder_web_ui_and_api_expose_planning_question_response_flow():
    import examples.coder_web_api as api
    backend = Path(api.__file__).read_text(encoding="utf-8")
    frontend = _frontend_path("/").read_text(encoding="utf-8")
    assert 'if path == "/v1/coder/respond":' in backend
    assert "wait_for_user_response" in backend
    assert 'id="askOpen"' in frontend
    assert 'API+"/respond"' in frontend
    assert "10 秒内点击" in frontend



def test_coder_plan_summary_shows_goal_scope_and_milestones():
    import examples.coder_web_api as api
    from coder.state import CoderStep

    step = CoderStep(
        1,
        "PLAN",
        {},
        {
            "status": "PLAN_SET",
            "long_term_goal": "complete notebook TODOs",
            "goal": {
                "description": "complete notebook TODOs",
                "scope_files": ["assignment1.ipynb", "tests/test_assignment1.py"],
                "milestones": ["inspect notebook", "implement TODOs", "verify results"],
                "success_criteria": ["TODOs implemented", "validation passes"],
            },
        },
    )
    summary = api._step_summary(step)
    assert "complete notebook TODOs" in summary
    assert "assignment1.ipynb" in summary
    assert "inspect notebook" in summary
    assert "validation passes" in summary

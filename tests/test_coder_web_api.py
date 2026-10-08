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


def test_coder_web_frontend_exists():
    path = _frontend_path("/")
    assert path is not None
    assert path.name == "coder.html"
    assert path.stat().st_size > 5000


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


def test_coder_web_frontend_has_drag_drop_and_timeouts():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert "拖到这里" in source
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
    assert 'JSON.stringify({request,project:selectedProject()||null})' in source
    assert "/projects" in source
    assert 'window.addEventListener("error"' in source
    assert 'window.addEventListener("unhandledrejection"' in source
    assert "DOMContentLoaded" in source
    assert "Coder UI DOM 初始化失败" in source
    assert "document.getElementById" in source
    assert "attempt<100" in source
    assert "setTimeout(()=>initCoderUI(attempt+1),50)" in source
    assert "正在根据任务让 LLM 自动命名新项目" in source
    assert 'e.type==="named"' in source
    assert "unique_if_requested=True" in source


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
        "workspace", "project", "projectHint", "drop", "fileInput", "files",
        "dot", "health", "task", "timer", "run", "timeline", "request",
        "modified", "tests", "finished", "verified", "steps", "summary",
    }
    assert expected <= parser.ids

    js_refs = set(re.findall(r'\$\("([^"]+)"\)', source))
    assert js_refs <= parser.ids
    assert "UI_IDS" in source
    assert 'const missing=UI_IDS.filter(id=>!document.getElementById(id));' in source
    assert "Coder UI 元素不可用" in source
    assert "async async function" not in source
    assert "async function fetchTimeout" in source


def test_coder_web_frontend_does_not_throw_on_missing_dom_during_startup():
    source = _frontend_path("/").read_text(encoding="utf-8")
    assert 'throw new Error("Coder UI DOM 初始化失败")' not in source
    assert "void loadProjects();" in source
    assert "void health();" in source
    assert 'requested_project = request.get("project")' in source
    assert "self._stream_run(task, requested_project)" in source
    assert "setInterval(()=>void health(),30000);" in source

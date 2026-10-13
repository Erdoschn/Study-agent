from types import SimpleNamespace
from examples.deepseek_web_study_agent_api import (
    Handler,
    WEB_ROOT,
    add_attachment_context,
    build_web_only_config,
    parse_chat_request_body,
    web_asset_path,
)


def test_web_api_extracts_latest_user_message_from_history():
    prompt = Handler._prompt([
        {"role": "system", "content": "teach clearly"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "new"},
    ])
    assert "【用户】" in prompt
    assert "new" in prompt
    assert "teach clearly" in prompt


def test_web_api_rejects_missing_user_message():
    try:
        Handler._prompt([{"role": "system", "content": "only system"}])
    except ValueError as exc:
        assert "user" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_web_api_accepts_multimodal_text_content():
    assert Handler._message_text([
        {"type": "input_text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]) == "hello\nworld"


def test_web_api_isolates_browser_model():
    config = {
        "providers": {
            "deepseek_web": {"type": "browser", "enabled": True},
            "other": {"type": "openai_compatible", "enabled": True},
        },
        "models": {
            "deepseek-web": {"provider": "deepseek_web", "enabled": True, "paid": False},
            "other-model": {"provider": "other", "enabled": True, "paid": False},
        },
    }
    isolated = build_web_only_config(config)
    assert list(isolated["providers"]) == ["deepseek_web"]
    assert list(isolated["models"]) == ["deepseek-web"]


def test_web_api_keeps_browser_factory_alive_after_run():
    from examples.deepseek_web_study_agent_api import Handler

    class Factory:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    factory = Factory()

    class Agent:
        def __init__(self):
            self._web_model_factory = factory

        def run(self, question):
            assert question == "hello"
            return SimpleNamespace(
                step_count=1,
                evidence=[],
                claims=[],
            )

    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(agent=Agent())
    events = []

    result = handler._run("hello", events.append)

    assert result.step_count == 1
    assert factory.closed is False
    assert events[0] == "🤔 StudyAgent 正在分析任务"


def test_web_api_does_not_cooldown_its_only_browser_model():
    config = {"providers": {}, "models": {}}
    isolated = build_web_only_config(config)

    from examples.deepseek_web_study_agent_api import build_agent

    agent = build_agent(config)
    registry = agent.reasoner.model_router.registry

    assert list(isolated["models"]) == ["deepseek-web"]
    assert registry.BASE_COOLDOWN_SECONDS == 0.0
    assert registry.MAX_COOLDOWN_SECONDS == 0.0
    assert registry.PROVIDER_COOLDOWN_SECONDS == 0.0


def test_web_api_supplies_browser_defaults_when_local_config_lacks_them():
    isolated = build_web_only_config({"providers": {}, "models": {}})
    provider = isolated["providers"]["deepseek_web"]
    assert provider["type"] == "browser"
    assert provider["url"] == "https://chat.deepseek.com/"
    assert provider["browser_channel"] == "msedge"
    assert isolated["models"]["deepseek-web"]["paid"] is False


def test_web_api_resolves_only_files_inside_web_root():
    assert web_asset_path("/") == WEB_ROOT / "index.html"
    assert web_asset_path("/index.html") == WEB_ROOT / "index.html"
    assert web_asset_path("/../examples/deepseek_web_study_agent_api.py") is None


def test_web_frontend_does_not_force_markdown_line_breaks():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert "breaks:false" in source
    assert "breaks:true" not in source

def test_web_frontend_uses_markdown_safe_math_placeholders():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert 'const token=i=>"STUDYAGENTMATHTOKEN"+i+"X";' in source
    assert "__STUDY_AGENT_MATH_" not in source


def test_web_frontend_protects_latex_before_markdown():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert "function markdown(x)" in source
    assert r".replace(/\\\[[\s\S]*?\\\]/g,keep)" in source
    assert r".replace(/\$\$[\s\S]*?\$\$/g,keep)" in source
    assert r".replace(/\\\([\s\S]*?\\\)/g,keep)" in source
    assert r'return keep((/\\begin\{/.test(inner)||inner.includes("\n"))' in source
    assert "outsideCode" in source
    assert "MathJax.typesetPromise([root])" in source


def test_web_frontend_shows_agent_status_before_final_answer():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    status_marker = 'b.append(renderStatus(m.statuses,!m.done));'
    body_marker = 'else body.innerHTML=markdown(m.content||"");'
    assert source.index(status_marker) < source.index(body_marker)


def test_web_frontend_exists():
    path = WEB_ROOT / "index.html"
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_web_api_exposes_health_under_v1_prefix():
    source = (WEB_ROOT.parent / "examples" / "deepseek_web_study_agent_api.py").read_text(
        encoding="utf-8"
    )
    assert 'if path in {"/health", "/v1/health"}:' in source



def test_web_api_multipart_keeps_original_file_bytes_without_extracting_text():
    boundary = "study-agent-upload-test"
    request = b'{"model":"deepseek-web","messages":[{"role":"user","content":"Summarize the attachment"}],"stream":true}'
    raw_file = b"\x00\xffbinary-payload\nnot-extracted"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="request"\r\n'
        "Content-Type: application/json\r\n\r\n"
    ).encode() + request + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="report.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + raw_file + f"\r\n--{boundary}--\r\n".encode()

    parsed, attachments = parse_chat_request_body(
        body,
        f"multipart/form-data; boundary={boundary}",
    )

    assert parsed["messages"][0]["content"] == "Summarize the attachment"
    assert len(attachments) == 1
    assert attachments[0]["name"] == "report.pdf"
    assert attachments[0]["mimeType"] == "application/pdf"
    assert attachments[0]["buffer"] == raw_file


def test_attachment_context_adds_only_names_and_not_file_contents():
    marker = b"SHOULD_NEVER_ENTER_PROMPT_98765"
    question = add_attachment_context(
        "Summarize the attached file",
        [{"name": "paper.pdf", "mimeType": "application/pdf", "buffer": marker}],
    )

    assert "paper.pdf" in question
    assert "原始附件" in question
    assert marker.decode() not in question
    assert "Study Agent 没有预先解析或提取文件正文" in question


def test_web_api_rejects_oversized_attachment(monkeypatch):
    import examples.deepseek_web_study_agent_api as api

    monkeypatch.setattr(api, "MAX_UPLOAD_FILE_BYTES", 3)
    boundary = "too-big"
    request = b'{"messages":[{"role":"user","content":"open"}]}'
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="request"\r\n\r\n'
    ).encode() + request + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="x.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode() + b"1234" + f"\r\n--{boundary}--\r\n".encode()

    try:
        parse_chat_request_body(body, f"multipart/form-data; boundary={boundary}")
    except ValueError as exc:
        assert "单文件上限" in str(exc)
    else:
        raise AssertionError("expected oversized upload rejection")


def test_web_frontend_uses_multipart_and_direct_attachment_upload():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert 'id="fileInput" type="file" multiple' in source
    assert 'id="dropOverlay"' in source
    assert 'dropTarget.addEventListener("drop"' in source
    assert 'formData.append("file",file,file.name)' in source
    assert 'headers(true),body:formData' in source
    assert 'formData.append("request",JSON.stringify(payload))' in source
    assert "filesByConversation" not in source


def test_web_frontend_attachments_are_consumed_once_per_submission():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    send_start = source.index("async function sendMessage(){")
    clear_composer = source.index("clearPendingFiles();persist();render();running(true);scroll();", send_start)
    attach_files = source.index('for(const file of filesForRequest)formData.append("file",file,file.name);', send_start)

    assert "const filesForRequest=state.pendingFiles.slice()" in source
    assert 'c.messages.push({role:"user",content:text,attachments:filesForRequest.map(file=>file.name)})' in source
    assert clear_composer < attach_files
    assert "function clearPendingFiles(){state.pendingFiles=[];if(fileInput)fileInput.value=\"\";renderAttachments()}" in source
    assert "const retained=state.filesByConversation" not in source
    assert "for(const file of state.pendingFiles)" in source



def test_web_api_prewarm_creates_the_only_browser_candidate():
    from examples.deepseek_web_study_agent_api import MODEL_ID, _prewarm_browser_model

    model = SimpleNamespace(name=MODEL_ID)
    client = SimpleNamespace(user_data_dir=".study-agent-browser")

    class Registry:
        def available(self, *, allow_paid):
            assert allow_paid is False
            return [model]

    class Factory:
        def __init__(self):
            self.created = []

        def create(self, candidate):
            self.created.append(candidate)
            return client

    factory = Factory()
    agent = SimpleNamespace(_web_model_factory=factory, _web_model_registry=Registry())

    assert _prewarm_browser_model(agent) is True
    assert factory.created == [model]


def test_web_api_prewarm_failure_does_not_abort_startup():
    from examples.deepseek_web_study_agent_api import _prewarm_browser_model

    class Registry:
        def available(self, *, allow_paid):
            raise RuntimeError("browser unavailable")

    class Factory:
        pass

    agent = SimpleNamespace(_web_model_factory=Factory(), _web_model_registry=Registry())
    assert _prewarm_browser_model(agent) is False


def test_browser_origin_policy_rejects_external_and_wrong_port_origins():
    from examples.deepseek_web_study_agent_api import PORT, _allowed_browser_origin

    assert _allowed_browser_origin("")
    assert _allowed_browser_origin(f"http://127.0.0.1:{PORT}")
    assert _allowed_browser_origin(f"http://localhost:{PORT}")
    assert not _allowed_browser_origin("https://evil.example")
    assert not _allowed_browser_origin(f"http://127.0.0.1:{PORT + 1}")
    assert not _allowed_browser_origin("null")


def test_non_loopback_bind_requires_an_api_key():
    from examples.deepseek_web_study_agent_api import _require_auth_for_non_loopback_bind

    _require_auth_for_non_loopback_bind("127.0.0.1", "")
    _require_auth_for_non_loopback_bind("localhost", "")
    _require_auth_for_non_loopback_bind("0.0.0.0", "secret")
    try:
        _require_auth_for_non_loopback_bind("0.0.0.0", "")
    except RuntimeError as exc:
        assert "STUDY_AGENT_WEB_API_KEY" in str(exc)
    else:
        raise AssertionError("expected unauthenticated non-loopback bind to be rejected")


def test_multipart_upload_rejects_control_characters_in_filenames():
    boundary = "filename-control-test"
    request = b'{"messages":[{"role":"user","content":"read"}]}'
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="request"\r\n\r\n'
    ).encode() + request + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="bad\x01name.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + b"%PDF-test" + f"\r\n--{boundary}--\r\n".encode()
    try:
        parse_chat_request_body(body, f"multipart/form-data; boundary={boundary}")
    except ValueError as exc:
        assert "控制字符" in str(exc)
    else:
        raise AssertionError("expected unsafe filename to be rejected")


def test_web_frontend_does_not_persist_api_key_and_warns_on_remote_api():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert 'localStorage.setItem("studyagent.apiKey"' not in source
    assert 'localStorage.removeItem("studyagent.apiKey")' in source
    assert "该地址会收到你的提示词、对话上下文、上传附件和当前页面内存中的 API Key" in source
    assert "dompurify@3.4.17" in source
    assert "marked@18.1.1" in source
    assert "对话文本和附件文件名会保存在当前浏览器本地存储" in source



def test_web_frontend_fails_closed_if_markdown_sanitizer_is_unavailable():
    source = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    assert "const safeHtml=window.DOMPurify" in source
    assert ':escapeHtml(source).replace(/\\n/g,"<br>")' in source
    assert "replaceAll(token(i),escapeHtml(value))" in source
    assert "replaceAll(token(i),value)" not in source



def test_web_api_always_cleans_factory_when_begin_run_fails():
    events = []

    class Factory:
        def begin_run(self, cancellation_event):
            events.append("begin")
            raise RuntimeError("simulated attachment setup failure")

        def end_run(self, cancellation_event):
            events.append("end")

    factory = Factory()

    class Agent:
        _web_model_factory = factory

        def run(self, question):
            raise AssertionError("Agent.run must not run when setup fails")

    handler = object.__new__(Handler)
    handler.server = SimpleNamespace(agent=Agent())

    try:
        handler._run("hello", lambda _event: None)
    except RuntimeError as exc:
        assert "simulated attachment setup failure" in str(exc)
    else:
        raise AssertionError("expected setup failure")

    assert events == ["begin", "end"]

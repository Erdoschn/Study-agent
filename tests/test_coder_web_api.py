from pathlib import Path

from examples.coder_web_api import (
    MAX_RUNTIME_SECONDS,
    UPLOAD_MAX_BYTES,
    _frontend_path,
    parse_multipart_upload,
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

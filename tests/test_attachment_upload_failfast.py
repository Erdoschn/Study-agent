"""Regression tests for attachment delivery failures that must stop Agent execution."""

from types import SimpleNamespace

import pytest

import core.agent as agent_module
import core.task_analyzer as task_analyzer_module
from core.agent import StudyAgent
from core.errors import AttachmentUploadError
from core.task_analyzer import TaskAnalyzer
from core.web_model import BrowserModel


def test_browser_model_marks_input_file_timeout_as_attachment_error():
    class Locator:
        def count(self):
            return 1

        def nth(self, _index):
            return self

        def set_input_files(self, _files, *, timeout=None):
            assert timeout == BrowserModel.ATTACHMENT_SELECT_TIMEOUT_MS
            raise TimeoutError("slow browser input")

    class Page:
        def locator(self, _selector):
            return Locator()

    model = BrowserModel(poll_interval=0.01)
    with pytest.raises(AttachmentUploadError, match="文件未发送"):
        model._upload_attachments(
            Page(),
            [{"name": "large.pdf", "mimeType": "application/pdf", "buffer": b"raw"}],
        )


def test_task_analyzer_does_not_swallow_attachment_upload_failure(monkeypatch):
    model = SimpleNamespace(name="deepseek-web", reasoning_effort_param=None)
    choice = SimpleNamespace(model=model, effort=None)
    monkeypatch.setattr(
        task_analyzer_module,
        "get_model_choices",
        lambda *_args, **_kwargs: [choice],
    )
    failure = AttachmentUploadError("DeepSeek Web 附件上传超时。")
    monkeypatch.setattr(
        task_analyzer_module,
        "call_model_with_effort",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    recorded = []
    registry = SimpleNamespace(
        record_failure=lambda *args, **kwargs: recorded.append((args, kwargs)),
        is_provider_level_failure=lambda _exc: True,
    )
    reasoner = SimpleNamespace(
        model_router=SimpleNamespace(registry=registry),
        model_factory=SimpleNamespace(create=lambda _model: object()),
        allow_paid=False,
    )

    with pytest.raises(AttachmentUploadError, match="附件上传超时"):
        TaskAnalyzer(reasoner).analyze("请阅读我上传的文件")

    assert len(recorded) == 1


def test_study_agent_stops_instead_of_retrying_failed_attachment_upload(monkeypatch):
    failure = AttachmentUploadError("DeepSeek Web 附件上传未完成。")

    class BrokenTaskAnalyzer:
        def __init__(self, _reasoner):
            pass

        def analyze(self, *_args, **_kwargs):
            raise failure

    monkeypatch.setattr(agent_module, "TaskAnalyzer", BrokenTaskAnalyzer)
    loop_calls = []
    monkeypatch.setattr(
        agent_module.AgentToolLoop,
        "run",
        lambda _self, state: loop_calls.append(state) or state,
    )

    agent = StudyAgent(reasoner=SimpleNamespace())
    state = agent.run("请根据我上传的附件整理所有知识点")

    assert loop_calls == []
    assert state.error and state.error.startswith("附件上传失败")
    assert "Agent 未能完成任务" in state.final_answer
    assert "DeepSeek Web 附件上传未完成" in state.final_answer



def test_send_button_timeout_is_attachment_failure_and_stops_further_model_roles(monkeypatch):
    model = BrowserModel(poll_interval=0.001)
    model._attachment_session_active = True
    model.SEND_BUTTON_READY_TIMEOUT_SECONDS = 0.01
    monkeypatch.setattr(model, "_submission_is_confirmed", lambda _page: False)
    monkeypatch.setattr(model, "_composer_contains_prompt", lambda _textbox, _prompt: True)
    monkeypatch.setattr(model, "_click_send_button", lambda _page: False)
    monkeypatch.setattr(model, "_sleep", lambda _seconds: None)

    with pytest.raises(AttachmentUploadError, match="发送按钮在"):
        model._wait_for_send_button_and_confirmation(object(), object(), "study from files")

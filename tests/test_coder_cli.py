import sys

from examples import coder_agent as cli


def test_coder_cli_defaults_to_normal_mode(monkeypatch):
    calls = []

    class FakeResult:
        finished = True
        goal_verified = True
        step_count = 2
        modified_files = set()
        metrics = {"test_runs": 1}
        error = None
        summary = "任务已完成。\n本次修改：\n  - 修改 score_utils.py"

    class FakeAgent:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self, request, *, stop_confirmation=None):
            calls.append(request)
            return FakeResult()

    monkeypatch.setattr(cli, "CoderAgent", FakeAgent)
    monkeypatch.setattr(cli.debug, "set_enabled", lambda enabled: calls.append(("debug", enabled)))
    monkeypatch.setattr(sys, "argv", ["coder_agent.py", "inspect", "tests"])

    assert cli.main() == 0
    assert calls == [
        ("debug", False),
        {"debug_mode": False},
        "inspect tests",
    ]


def test_coder_cli_explicit_debug_keeps_debug_mode(monkeypatch):
    calls = []

    class FakeResult:
        finished = True
        goal_verified = True
        step_count = 1
        modified_files = {"a.py"}
        metrics = {"test_runs": 0}
        error = None

    class FakeAgent:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self, request, *, stop_confirmation=None):
            calls.append(request)
            return FakeResult()

    monkeypatch.setattr(cli, "CoderAgent", FakeAgent)
    monkeypatch.setattr(cli.debug, "set_enabled", lambda enabled: calls.append(("debug", enabled)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["coder_agent.py", "--debug", "inspect", "tests"],
    )

    assert cli.main() == 0
    assert calls == [
        ("debug", True),
        {"debug_mode": True},
        "inspect tests",
    ]


def test_coder_cli_prints_completion_summary(monkeypatch, capsys):
    class FakeResult:
        finished = True
        goal_verified = True
        step_count = 2
        modified_files = {"score_utils.py"}
        metrics = {"test_runs": 1}
        error = None
        summary = "任务已完成。\n本次修改：\n  - 修改 score_utils.py"

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, request, *, stop_confirmation=None):
            return FakeResult()

    monkeypatch.setattr(cli, "CoderAgent", FakeAgent)
    monkeypatch.setattr(cli.debug, "set_enabled", lambda enabled: None)
    monkeypatch.setattr(sys, "argv", ["coder_agent.py", "fix"])

    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "Coder summary:" in output
    assert "修改 score_utils.py" in output

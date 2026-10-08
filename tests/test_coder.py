from pathlib import Path
from types import SimpleNamespace

import pytest

from coder.agent import CoderAgent
from coder.filesystem import WorkspaceFS, WorkspaceSecurityError
from coder.harness import CoderHarness
from coder.sandbox import DockerPythonSandbox, SandboxResult


def test_workspace_rejects_escape_and_absolute_paths(tmp_path):
    fs = WorkspaceFS(tmp_path)
    with pytest.raises(WorkspaceSecurityError):
        fs.read_text("../secret.py")
    with pytest.raises(WorkspaceSecurityError):
        fs.read_text(r"C:\Users\secret.py")


def test_workspace_rejects_sensitive_files(tmp_path):
    fs = WorkspaceFS(tmp_path)
    (tmp_path / ".env").write_text("SECRET=x", encoding="utf-8")
    with pytest.raises(WorkspaceSecurityError):
        fs.read_text(".env")


def test_workspace_allows_notebook_read_write(tmp_path):
    fs = WorkspaceFS(tmp_path)
    notebook = '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}'
    fs.write_notebook("notes/demo.ipynb", notebook)
    assert '"nbformat": 4' in fs.read_text("notes/demo.ipynb")


def test_workspace_rejects_invalid_notebook(tmp_path):
    fs = WorkspaceFS(tmp_path)
    with pytest.raises(WorkspaceSecurityError, match="Notebook JSON 无效"):
        fs.write_notebook("demo.ipynb", "{not-json}")


def test_workspace_rejects_invalid_notebook_cell(tmp_path):
    fs = WorkspaceFS(tmp_path)
    bad = '{"cells":[{"cell_type":"unknown","source":[]}],"metadata":{},"nbformat":4}'
    with pytest.raises(WorkspaceSecurityError, match="cell\[0\]"):
        fs.write_notebook("demo.ipynb", bad)


def test_workspace_allows_only_python_writes(tmp_path):
    fs = WorkspaceFS(tmp_path)
    with pytest.raises(WorkspaceSecurityError):
        fs.write_text("notes.txt", "no")
    fs.write_text("src/a.py", "print(1)")
    assert fs.read_text("src/a.py") == "print(1)"


def test_workspace_patch_requires_exactly_one_match(tmp_path):
    fs = WorkspaceFS(tmp_path)
    fs.write_text("a.py", "x=1\nx=1\n")
    with pytest.raises(WorkspaceSecurityError):
        fs.patch_text("a.py", "x=1", "x=2")


def test_workspace_test_creation_is_scoped_to_tests(tmp_path):
    fs = WorkspaceFS(tmp_path)
    fs.write_text("tests/test_a.py", "def test_a(): assert True", test=True)
    with pytest.raises(WorkspaceSecurityError):
        fs.write_text("src/test_a.py", "def test_a(): assert True", test=True)


def test_harness_rejects_unapproved_actions(tmp_path):
    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    with pytest.raises(PermissionError):
        harness.execute("EXEC", {}, SimpleNamespace())


def test_harness_never_uses_shell_for_sandbox(tmp_path, monkeypatch):
    calls = []

    class FakeStream:
        def read(self, _size):
            return b""

    class FakeProc:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            self.stdout = FakeStream()
            self.stderr = FakeStream()
        def poll(self): return 0
        def wait(self): return 0

    monkeypatch.setattr("coder.sandbox.subprocess.Popen", FakeProc)
    monkeypatch.setattr("coder.sandbox.subprocess.run", lambda *a, **k: None)
    sandbox = DockerPythonSandbox(tmp_path, timeout_seconds=1)
    result = sandbox.run("python", ["a.py"])
    assert result.returncode == 0
    assert calls
    command = calls[0][0][0]
    kwargs = calls[0][1]
    assert kwargs["shell"] is False
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--cap-drop" in command
    assert "--pull=never" in command
    tmpfs_args = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--tmpfs"]
    assert any("/sandbox:" in value and "uid=65534" in value and "gid=65534" in value for value in tmpfs_args)
    assert any("/tmp:" in value and "uid=65534" in value and "gid=65534" in value for value in tmpfs_args)


def test_goal_verifier_requires_test_after_latest_modification(tmp_path):
    class FakeSandbox:
        def run(self, kind, paths):
            return SandboxResult(0, "1 passed", "")

    harness = CoderHarness(str(tmp_path), sandbox=FakeSandbox())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("fix")
    harness.backup.ensure_initial_snapshot(0)
    harness.execute("WRITE_FILE", {"path": "a.py", "content": "print(1)"}, state)
    first = harness.execute("VERIFY_GOAL", {}, state)
    assert first["verified"] is False
    harness.execute("CREATE_TEST", {"path": "tests/test_a.py", "content": "def test_a(): assert True"}, state)
    harness.execute("RUN_PYTEST", {"paths": []}, state)
    second = harness.execute("VERIFY_GOAL", {}, state)
    assert second["verified"] is True


def test_agent_builds_deterministic_completion_summary(tmp_path):
    class FakeHarness(CoderHarness):
        def _verify_goal(self, args, state):
            return {"verified": True, "checks": []}

    class FakeReasoner:
        def __init__(self):
            self.model = SimpleNamespace(close=lambda: None)
            self.calls = 0

        def decide(self, state, tools):
            self.calls += 1
            if self.calls == 1:
                return {
                    "action": "WRITE_FILE",
                    "arguments": {
                        "path": "score_utils.py",
                        "content": "print(1)\n",
                    },
                    "reasoning_summary": "",
                    "goal": {},
                }
            if self.calls == 2:
                return {
                    "action": "CREATE_TEST",
                    "arguments": {
                        "path": "tests/test_score_utils.py",
                        "content": "def test_ok(): assert True\n",
                    },
                    "reasoning_summary": "",
                    "goal": {},
                }
            if self.calls == 3:
                return {
                    "action": "RUN_PYTEST",
                    "arguments": {"paths": []},
                    "reasoning_summary": "",
                    "goal": {},
                }
            return {"action": "FINISH", "arguments": {}, "reasoning_summary": "", "goal": {}}

    class PassSandbox:
        def run(self, kind, paths):
            return SandboxResult(0, "1 passed", "")

    harness = FakeHarness(str(tmp_path), sandbox=PassSandbox())
    result = CoderAgent(tmp_path, reasoner=FakeReasoner(), harness=harness).run("fix")

    assert result.finished is True
    assert result.goal_verified is True
    assert "score_utils.py" in result.summary
    assert "tests/test_score_utils.py" in result.summary
    assert "pytest 测试通过" in result.summary


def test_agent_emits_progress_events(tmp_path):
    class FakeHarness(CoderHarness):
        def _verify_goal(self, args, state):
            state.goal_verified = True
            return {"verified": True, "checks": []}

    class FakeReasoner:
        def __init__(self):
            self.model = SimpleNamespace(close=lambda: None)

        def decide(self, state, tools):
            return {"action": "FINISH", "arguments": {}, "reasoning_summary": "", "goal": {}}

    harness = FakeHarness(str(tmp_path), sandbox=SimpleNamespace())
    events = []
    agent = CoderAgent(
        tmp_path,
        reasoner=FakeReasoner(),
        harness=harness,
        max_runtime_seconds=30,
    )
    result = agent.run("finish", event_hook=events.append)

    assert result.finished is True
    assert [event["type"] for event in events] == ["started", "step", "finished"]
    assert events[1]["step"].action == "FINISH"


def test_agent_captures_initial_backup_before_reasoning(tmp_path):
    (tmp_path / "main.py").write_bytes(b"VERSION = 0\n")

    class FakeHarness(CoderHarness):
        _verified = True

        def _verify_goal(self, args, state):
            state.goal_verified = True
            return {"verified": True, "checks": []}

    class FakeReasoner:
        def __init__(self):
            self.model = SimpleNamespace(close=lambda: None)

        def decide(self, state, tools):
            return {"action": "FINISH", "arguments": {}, "reasoning_summary": "", "goal": {}}

    harness = FakeHarness(str(tmp_path), sandbox=SimpleNamespace())
    agent = CoderAgent(
        tmp_path,
        reasoner=FakeReasoner(),
        harness=harness,
        max_runtime_seconds=30,
    )
    result = agent.run("fix")
    assert result.finished is True
    assert result.goal_verified is True
    assert result.initial_backup_generation == 0
    assert harness.backup.has_initial_snapshot()
    assert harness.backup.contains_text("main.py", "VERSION = 0\n", initial=True)


def test_agent_does_not_finish_before_goal_is_verified(tmp_path):
    class FakeHarness(CoderHarness):
        def _verify_goal(self, args, state):
            state.goal_verified = self._verified
            return {"verified": self._verified, "checks": []}
        _verified = False

    class FakeReasoner:
        def __init__(self):
            self.calls = 0
            self.model = SimpleNamespace(close=lambda: None)
        def decide(self, state, tools):
            self.calls += 1
            if self.calls == 2:
                harness._verified = True
            return {"action": "FINISH", "arguments": {}, "reasoning_summary": "", "goal": {}}

    harness = FakeHarness(str(tmp_path), sandbox=SimpleNamespace())
    agent = CoderAgent(
        tmp_path,
        reasoner=FakeReasoner(),
        harness=harness,
        max_runtime_seconds=30,
    )
    result = agent.run("fix")
    assert result.goal_verified is True
    assert result.finished is True
    assert result.error is None
    assert result.step_count == 2


def test_sandbox_staging_excludes_sensitive_files(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SECRET=x", encoding="utf-8")
    (tmp_path / "main.py").write_text("print(1)", encoding="utf-8")
    observed = {}

    def fake_run(command, name):
        mount_arg = command[command.index("--mount") + 1]
        stage = Path(mount_arg.split("src=", 1)[1].split(",dst=", 1)[0])
        observed["files"] = {p.name for p in stage.rglob("*") if p.is_file()}
        return SandboxResult(0, "", "")

    sandbox = DockerPythonSandbox(tmp_path)
    monkeypatch.setattr(sandbox, "_run_limited", fake_run)
    result = sandbox.run("python", ["main.py"])

    assert result.passed is True
    assert observed["files"] == {"main.py"}


def test_harness_rejects_python_execution_outside_workspace(tmp_path):
    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("fix")
    with pytest.raises(WorkspaceSecurityError):
        harness.execute(
            "RUN_PYTHON",
            {"script_path": "../outside.py"},
            state,
        )

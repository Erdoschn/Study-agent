from pathlib import Path
from types import SimpleNamespace

import pytest

from coder.agent import CoderAgent
from coder.filesystem import WorkspaceFS, WorkspaceSecurityError
from coder.harness import CoderHarness
from coder.sandbox import DockerPythonSandbox, SandboxResult



def test_agent_rejects_its_own_runtime_source_tree():
    from coder.agent import RUNTIME_ROOT

    with pytest.raises(WorkspaceSecurityError, match="自身源码目录"):
        CoderAgent(RUNTIME_ROOT, reasoner=SimpleNamespace(), harness=SimpleNamespace())


def test_agent_rejects_workspace_nested_under_runtime_source_tree():
    from coder.agent import RUNTIME_ROOT

    with pytest.raises(WorkspaceSecurityError, match="自身源码目录"):
        CoderAgent(
            RUNTIME_ROOT / ".coder-test-workspace",
            reasoner=SimpleNamespace(),
            harness=SimpleNamespace(),
        )

def test_agent_rejects_workspace_that_contains_runtime_source_tree():
    from coder.agent import RUNTIME_ROOT

    with pytest.raises(WorkspaceSecurityError, match="覆盖 Study-agent"):
        CoderAgent(
            RUNTIME_ROOT.parent,
            reasoner=SimpleNamespace(),
            harness=SimpleNamespace(),
        )


def test_workspace_rejects_escape_and_absolute_paths(tmp_path):
    fs = WorkspaceFS(tmp_path)
    with pytest.raises(WorkspaceSecurityError):
        fs.read_text("../secret.py")
    with pytest.raises(WorkspaceSecurityError):
        fs.read_text(r"C:\Users\secret.py")


def test_workspace_rejects_internal_coder_backup_directory(tmp_path):
    backup = tmp_path / ".coder-backup"
    backup.mkdir()
    (backup / "main.py").write_text("SECRET = True", encoding="utf-8")
    fs = WorkspaceFS(tmp_path)

    with pytest.raises(WorkspaceSecurityError):
        fs.read_text(".coder-backup/main.py")
    assert all(".coder-backup" not in path for path in fs.list_files())


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


def test_workspace_patches_one_notebook_cell_without_rewriting_other_cells(tmp_path):
    import json

    fs = WorkspaceFS(tmp_path)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {"keep": True},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": ["keep\n"]}],
                "source": ["x = 1\n", "print(x)\n"],
                "id": "cell-a",
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["说明\n"],
                "id": "cell-b",
            },
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    fs.write_notebook("demo.ipynb", json.dumps(notebook, ensure_ascii=False))

    fs.patch_notebook(
        "demo.ipynb",
        0,
        "x = 1\nprint(x)\n",
        "x = 2\nprint(x)\n",
    )

    value = json.loads(fs.read_text("demo.ipynb"))
    assert value["cells"][0]["source"] == "x = 2\nprint(x)\n"
    assert value["cells"][0]["metadata"] == {"keep": True}
    assert value["cells"][0]["outputs"] == notebook["cells"][0]["outputs"]
    assert value["cells"][0]["execution_count"] == 3
    assert value["cells"][1] == notebook["cells"][1]


def test_workspace_patch_notebook_requires_exact_source(tmp_path):
    import json

    fs = WorkspaceFS(tmp_path)
    fs.write_notebook(
        "demo.ipynb",
        json.dumps({
            "cells": [{
                "cell_type": "code",
                "metadata": {},
                "source": ["x = 1\n"],
            }],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
    )

    with pytest.raises(WorkspaceSecurityError, match="old_source"):
        fs.patch_notebook("demo.ipynb", 0, "x = 2\n", "x = 3\n")


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


def test_harness_rejects_destructive_existing_notebook_rewrite(tmp_path):
    import json
    from types import SimpleNamespace

    fs = WorkspaceFS(tmp_path)
    original = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": ["a = 1\n"], "id": "a"},
            {"cell_type": "code", "metadata": {}, "source": ["b = 2\n"], "id": "b"},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    fs.write_notebook("demo.ipynb", json.dumps(original))
    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("update notebook")
    harness.execute("READ_FILE", {"path": "demo.ipynb"}, state)

    shortened = {
        "cells": [original["cells"][0]],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with pytest.raises(WorkspaceSecurityError, match="cell 数量减少"):
        harness.execute(
            "WRITE_NOTEBOOK",
            {"path": "demo.ipynb", "content": json.dumps(shortened)},
            state,
        )


def test_harness_supports_patch_notebook_action(tmp_path):
    import json
    from types import SimpleNamespace

    fs = WorkspaceFS(tmp_path)
    original = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": ["a = 1\n"], "id": "a"},
            {"cell_type": "markdown", "metadata": {}, "source": ["note\n"], "id": "b"},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    fs.write_notebook("demo.ipynb", json.dumps(original))
    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("update notebook")

    result = harness.execute(
        "PATCH_NOTEBOOK",
        {
            "path": "demo.ipynb",
            "cell_index": 0,
            "old_source": "a = 1\n",
            "new_source": "a = 2\n",
        },
        state,
    )

    assert result["status"] == "notebook_cell_patched"
    assert json.loads(fs.read_text("demo.ipynb"))["cells"][1] == original["cells"][1]
    assert "demo.ipynb" in state.modified_files
    assert "PATCH_NOTEBOOK" in [h["name"] for h in harness.tool_specs()]


def test_harness_rejects_destructive_existing_file_rewrite(tmp_path):
    from types import SimpleNamespace

    fs = WorkspaceFS(tmp_path)
    original = "".join(f"def function_{i}():\n    return {i}\n" for i in range(30))
    fs.write_text("framework.py", original)

    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("complete framework")
    harness.execute(
        "READ_FILE",
        {"path": "framework.py"},
        state,
    )

    shortened = "".join(f"def function_{i}():\n    return {i}\n" for i in range(12))
    with pytest.raises(WorkspaceSecurityError, match="明显缩水"):
        harness.execute(
            "WRITE_FILE",
            {"path": "framework.py", "content": shortened},
            state,
        )

    assert fs.read_text("framework.py") == original


def test_harness_allows_normal_existing_file_rewrite(tmp_path):
    from types import SimpleNamespace

    fs = WorkspaceFS(tmp_path)
    original = "".join(f"line_{i}\n" for i in range(30))
    fs.write_text("framework.py", original)

    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("update framework")
    harness.execute("READ_FILE", {"path": "framework.py"}, state)

    revised = "".join(f"line_{i}\n" for i in range(24))
    harness.execute("WRITE_FILE", {"path": "framework.py", "content": revised}, state)

    assert fs.read_text("framework.py") == revised


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


def test_agent_can_keep_shared_model_open(tmp_path):
    closed = []

    class FakeHarness(CoderHarness):
        def _verify_goal(self, args, state):
            return {"verified": True, "checks": []}

    class FakeReasoner:
        def __init__(self):
            self.model = SimpleNamespace(
                close=lambda: closed.append(True),
            )

        def decide(self, state, tools):
            return {"action": "FINISH", "arguments": {}, "reasoning_summary": "", "goal": {}}

    harness = FakeHarness(str(tmp_path), sandbox=SimpleNamespace())
    result = CoderAgent(
        tmp_path,
        reasoner=FakeReasoner(),
        harness=harness,
        max_runtime_seconds=30,
        close_model_on_run=False,
    ).run("finish")

    assert result.finished is True
    assert result.goal_verified is True
    assert closed == []


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


def test_sandbox_kills_process_when_coder_run_is_cancelled(tmp_path, monkeypatch):
    import threading

    from core.cancellation import RunCancelled

    class Stream:
        def read(self, _size):
            return b""

    class Proc:
        def __init__(self):
            self.killed = False

        def poll(self):
            return 0 if self.killed else None

        def kill(self):
            self.killed = True

        def wait(self):
            return -9

        stdout = Stream()
        stderr = Stream()

    proc = Proc()
    calls = []

    def fake_popen(*args, **kwargs):
        calls.append(("popen", args, kwargs))
        return proc

    def fake_run(*args, **kwargs):
        calls.append(("run", args, kwargs))
        return None

    monkeypatch.setattr("coder.sandbox.subprocess.Popen", fake_popen)
    monkeypatch.setattr("coder.sandbox.subprocess.run", fake_run)

    from coder.sandbox import DockerPythonSandbox

    event = threading.Event()
    event.set()
    sandbox = DockerPythonSandbox(tmp_path)

    with pytest.raises(RunCancelled):
        sandbox._run_limited(
            ["docker", "run"],
            "coder-sandbox-test",
            cancellation_event=event,
        )

    assert proc.killed is True
    assert calls[0][0] == "popen"
    assert any(call[0] == "run" for call in calls)


def test_harness_rejects_python_execution_outside_workspace(tmp_path):
    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("fix")
    with pytest.raises(WorkspaceSecurityError):
        harness.execute(
            "RUN_PYTHON",
            {"script_path": "../outside.py"},
            state,
        )

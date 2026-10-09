from pathlib import Path
from types import SimpleNamespace

import pytest

from coder.agent import CoderAgent
from coder.filesystem import WorkspaceFS, WorkspaceSecurityError
from coder.harness import CoderHarness
from coder.reasoner import CoderReasoner
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


def test_concurrent_workspace_writes_never_leave_partial_file(tmp_path):
    import threading

    fs = WorkspaceFS(tmp_path)
    payloads = ["A" * 100_000, "B" * 100_000]
    barrier = threading.Barrier(len(payloads))
    errors = []

    def write(payload):
        try:
            barrier.wait(timeout=2)
            fs.write_text("concurrent.py", payload)
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=write, args=(payload,)) for payload in payloads]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert fs.read_text("concurrent.py") in payloads


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


def test_workspace_read_file_can_walk_large_files_in_line_chunks(tmp_path):
    fs = WorkspaceFS(tmp_path)
    content = "".join(f"line_{i} = {i}\n" for i in range(1, 1001))
    fs.write_text("large.py", content)
    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("read large file")

    first = harness.execute("READ_FILE", {"path": "large.py"}, state)
    assert first["complete"] is False
    assert first["start_line"] == 1
    assert first["end_line"] == 400
    assert first["total_lines"] == 1000
    assert first["next_start_line"] == 401

    second = harness.execute("READ_FILE", {"path": "large.py", "start_line": 401, "end_line": 800}, state)
    third = harness.execute("READ_FILE", {"path": "large.py", "start_line": 801, "end_line": 1000}, state)
    assert second["content"].startswith("line_401")
    assert third["content"].endswith("line_1000 = 1000\n")
    assert third["full_file_read"] is True
    assert third["read_coverage"] == [[1, 1000]]


def test_harness_rejects_whole_file_write_until_existing_file_was_fully_read(tmp_path):
    fs = WorkspaceFS(tmp_path)
    original = "".join(f"line_{i}\n" for i in range(1, 501))
    fs.write_text("framework.py", original)
    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("update framework")
    harness.execute("READ_FILE", {"path": "framework.py", "start_line": 1, "end_line": 400}, state)
    with pytest.raises(WorkspaceSecurityError, match="尚未完整读取当前文件"):
        harness.execute("WRITE_FILE", {"path": "framework.py", "content": original}, state)


def test_harness_allows_whole_file_write_after_reading_all_chunks(tmp_path):
    fs = WorkspaceFS(tmp_path)
    original = "".join(f"line_{i}\n" for i in range(1, 501))
    fs.write_text("framework.py", original)
    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = __import__("coder.state", fromlist=["CoderState"]).CoderState("update framework")
    harness.execute("READ_FILE", {"path": "framework.py", "start_line": 1, "end_line": 400}, state)
    harness.execute("READ_FILE", {"path": "framework.py", "start_line": 401, "end_line": 500}, state)
    revised = original.replace("line_250", "line_250_changed", 1)
    harness.execute("WRITE_FILE", {"path": "framework.py", "content": revised}, state)
    assert fs.read_text("framework.py") == revised


def test_reasoner_untrusted_tool_output_budget_is_large_enough_for_normal_files():
    rendered = CoderReasoner._untrusted({"path": "a.py", "content": "x" * 20000})
    assert "x" * 19000 in rendered
    assert len(rendered) > 20000


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


def test_workspace_patch_notebook_accepts_unique_source_snippet(tmp_path):
    import json

    fs = WorkspaceFS(tmp_path)
    fs.write_notebook(
        "demo.ipynb",
        json.dumps({
            "cells": [{
                "cell_type": "code",
                "metadata": {},
                "source": ["first = 1\n", "target = 2\n", "last = 3\n"],
            }],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
    )

    fs.patch_notebook("demo.ipynb", 0, "target = 2\r\n", "target = 20\n")

    cell = json.loads(fs.read_text("demo.ipynb"))["cells"][0]
    assert cell["source"] == "first = 1\ntarget = 20\nlast = 3\n"


def test_workspace_patch_notebook_rejects_ambiguous_snippet(tmp_path):
    import json

    fs = WorkspaceFS(tmp_path)
    fs.write_notebook(
        "demo.ipynb",
        json.dumps({
            "cells": [{
                "cell_type": "code",
                "metadata": {},
                "source": ["value = 1\n", "value = 1\n"],
            }],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
    )

    with pytest.raises(WorkspaceSecurityError, match="唯一出现"):
        fs.patch_notebook("demo.ipynb", 0, "value = 1\n", "value = 2\n")


def test_workspace_rejects_invalid_notebook(tmp_path):
    fs = WorkspaceFS(tmp_path)
    with pytest.raises(WorkspaceSecurityError, match="Notebook JSON 无效"):
        fs.write_notebook("demo.ipynb", "{not-json}")


def test_workspace_rejects_invalid_notebook_cell(tmp_path):
    fs = WorkspaceFS(tmp_path)
    bad = '{"cells":[{"cell_type":"unknown","source":[]}],"metadata":{},"nbformat":4}'
    with pytest.raises(WorkspaceSecurityError, match=r"cell\[0\]"):
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



def test_coder_plan_sets_scope_milestones_and_test_policy(tmp_path):
    from coder.agent import CoderAgent
    from coder.state import CoderGoal, CoderState

    state = CoderState(
        request="fill TODO cells in a notebook",
        goal=CoderGoal("fill TODO cells in a notebook", must_create_tests=False, must_pass_tests=False),
    )
    CoderAgent._apply_plan(state, {
        "description": "complete the notebook TODOs",
        "scope_files": ["assignment1.ipynb", "tests/test_assignment1.py"],
        "milestones": ["inspect current notebook", "implement TODOs", "verify results"],
        "success_criteria": ["preserve the existing notebook", "TODOs are implemented", "validation is run"],
        "required_tests": ["tests/test_assignment1.py"],
        "must_create_tests": True,
        "must_pass_tests": False,
    })
    assert state.plan_confirmed is True
    assert state.goal.scope_files == ["assignment1.ipynb", "tests/test_assignment1.py"]
    assert state.goal.must_create_tests is True
    assert state.goal.must_pass_tests is False
    assert state.current_milestone == "inspect current notebook"
    CoderAgent._apply_plan(state, {"scope_files": ["inspect_cell.py"], "milestones": ["inspect current notebook", "implement TODOs"]})
    assert "inspect_cell.py" in state.goal.scope_files
    assert len(state.goal.scope_files) == 3


def test_harness_allows_planned_new_files_and_blocks_unplanned_paths(tmp_path):
    from types import SimpleNamespace
    from coder.harness import CoderHarness
    from coder.state import CoderGoal, CoderState

    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = CoderState(
        request="complete notebook TODOs",
        goal=CoderGoal(
            "complete notebook TODOs",
            scope_files=["assignment1.ipynb", "tests/test_assignment1.py"],
            must_create_tests=False,
            must_pass_tests=False,
        ),
        plan_confirmed=True,
    )
    created = harness.execute(
        "CREATE_TEST",
        {"path": "tests/test_assignment1.py", "content": "def test_smoke(): assert True"},
        state,
    )
    assert created["status"] == "test_created"
    assert "tests/test_assignment1.py" in state.created_tests
    with pytest.raises(PermissionError, match="不在当前计划范围"):
        harness.execute("WRITE_FILE", {"path": "inspect_cell.py", "content": "print(1)"}, state)
    assert not (tmp_path / "inspect_cell.py").exists()


def test_harness_blocks_editing_existing_file_outside_plan_scope(tmp_path):
    from types import SimpleNamespace
    from coder.harness import CoderHarness
    from coder.state import CoderGoal, CoderState

    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = CoderState(
        request="complete notebook TODOs",
        goal=CoderGoal("complete notebook TODOs", scope_files=["assignment1.ipynb"], must_pass_tests=False),
        plan_confirmed=True,
    )
    (tmp_path / "tests").mkdir()
    target = tmp_path / "tests" / "test_assignment1.py"
    target.write_text("def test_original(): assert True\\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="不在当前计划范围"):
        harness.execute("PATCH_FILE", {"path": "tests/test_assignment1.py", "old_text": "test_original", "new_text": "test_rewritten"}, state)
    assert "test_original" in target.read_text(encoding="utf-8")

def test_coder_test_policy_does_not_match_words_containing_test():
    from coder.agent import CoderAgent

    assert CoderAgent._request_requires_tests("complete latest notebook TODO cells") is False
    assert CoderAgent._request_requires_tests("please run pytest") is True
    assert CoderAgent._request_requires_tests("补充单元测试") is True


def test_coder_only_creates_tests_when_user_requests_creation():
    from coder.agent import CoderAgent

    assert CoderAgent._request_requires_test_creation("补充测试用例") is True
    assert CoderAgent._request_requires_test_creation("修复 notebook 中的 TODO") is False


def test_coder_reasoner_accepts_stop_from_json_recovery():
    decision = CoderReasoner._parse('{"action":"STOP","arguments":{},"reasoning_summary":"malformed output"}')
    assert decision["action"] == "STOP"
    assert decision["reasoning_summary"] == "malformed output"


def test_coder_agent_stops_safely_on_stop_action(tmp_path):
    class StopReasoner:
        model = None
        def decide(self, _state, _tools):
            return {"action": "STOP", "arguments": {}, "reasoning_summary": "invalid JSON after repair"}

    class Harness:
        sandbox = SimpleNamespace()
        backup = None
        def tool_specs(self):
            return []
        def execute(self, *_args, **_kwargs):
            raise AssertionError("STOP must not execute a tool")

    agent = CoderAgent(tmp_path, reasoner=StopReasoner(), harness=Harness(), close_model_on_run=False)
    state = agent.run("修改现有 notebook TODO")
    assert state.finished is True
    assert state.goal_verified is False
    assert "invalid JSON after repair" in state.error
    assert state.steps[-1].action == "STOP"


def test_plan_can_authorize_new_helper_files_without_scope_bypass(tmp_path):
    from types import SimpleNamespace
    from coder.harness import CoderHarness
    from coder.state import CoderGoal, CoderState

    harness = CoderHarness(str(tmp_path), sandbox=SimpleNamespace())
    state = CoderState(
        "complete notebook TODOs",
        goal=CoderGoal("complete notebook TODOs", scope_files=["assignment1.ipynb", "inspect_cell.py"], must_pass_tests=False),
        plan_confirmed=True,
    )
    result = harness.execute("WRITE_FILE", {"path": "inspect_cell.py", "content": "print(1)"}, state)
    assert result["status"] == "written"
    assert (tmp_path / "inspect_cell.py").exists()


def test_coder_agent_requires_plan_before_any_tool_execution(tmp_path):
    class SequenceReasoner:
        model = None
        def __init__(self): self.calls = 0
        def decide(self, _state, _tools):
            self.calls += 1
            if self.calls == 1:
                return {"action": "LIST_FILES", "arguments": {}}
            if self.calls == 2:
                return {
                    "action": "PLAN", "arguments": {}, "reasoning_summary": "plan first",
                    "goal": {
                        "description": "complete the requested TODO",
                        "scope_files": ["assignment1.ipynb"],
                        "milestones": ["inspect", "implement", "verify"],
                        "success_criteria": ["complete requested TODO", "validate output"],
                        "must_create_tests": False, "must_pass_tests": False,
                    },
                }
            return {"action": "STOP", "arguments": {}, "reasoning_summary": "end test"}

    class Harness:
        sandbox = SimpleNamespace()
        backup = None
        def tool_specs(self): return []
        def execute(self, action, *_args):
            raise AssertionError(f"tool {action} executed before valid plan")

    reasoner = SequenceReasoner()
    agent = CoderAgent(tmp_path, reasoner=reasoner, harness=Harness(), close_model_on_run=False)
    state = agent.run("complete TODO in assignment1.ipynb")
    assert reasoner.calls == 3
    assert state.steps[0].action == "PLAN"
    assert state.plan_confirmed is True


def test_coder_agent_passes_user_reply_into_following_decisions(tmp_path):
    class SequenceReasoner:
        model = None
        def __init__(self): self.calls = 0; self.seen_responses = []
        def decide(self, state, _tools):
            self.calls += 1
            self.seen_responses.append(list(state.user_responses))
            if self.calls == 1:
                return {"action": "PLAN", "arguments": {}, "goal": {
                    "scope_files": ["main.py"], "milestones": ["inspect", "implement"],
                    "success_criteria": ["preserve behavior"], "must_pass_tests": False,
                }}
            if self.calls == 2:
                return {"action": "ASK_USER", "arguments": {"question": "Should public APIs stay compatible?"}}
            return {"action": "STOP", "arguments": {}, "reasoning_summary": "test complete"}

    class Harness:
        sandbox = SimpleNamespace()
        backup = None
        def tool_specs(self): return []
        def execute(self, *_args): raise AssertionError("no file action expected")

    reasoner = SequenceReasoner()
    agent = CoderAgent(tmp_path, reasoner=reasoner, harness=Harness(), close_model_on_run=False)
    questions = []
    state = agent.run('preserve public API behavior', user_interaction=lambda question: questions.append(question) or 'Yes, preserve compatibility')
    assert questions == ["Should public APIs stay compatible?"]
    assert state.user_responses[-1]["response"] == "Yes, preserve compatibility"
    assert reasoner.seen_responses[-1][-1]["response"] == "Yes, preserve compatibility"

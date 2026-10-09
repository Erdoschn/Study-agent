from __future__ import annotations

import os
from types import SimpleNamespace
from zipfile import ZipFile

from coder.backup import CoderBackupStore
from coder.harness import CoderHarness
from coder.sandbox import SandboxResult
from coder.state import CoderState


def test_backup_snapshot_contains_only_python_code_and_manifest(tmp_path):
    (tmp_path / "main.py").write_bytes(b"VERSION = 1\n")
    (tmp_path / "helper.pyi").write_bytes(b"VERSION: int\n")
    (tmp_path / "notes.txt").write_bytes(b"not code")
    (tmp_path / ".env").write_bytes(b"SECRET=x")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        "def test_main():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "analysis.ipynb").write_text(
        '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}',
        encoding="utf-8",
    )

    backup = CoderBackupStore(tmp_path)
    snapshot = backup.snapshot(3)

    assert snapshot.generation == 3
    assert snapshot.file_count == 4
    assert snapshot.archive == tmp_path / ".coder-backup" / "latest.zip"
    initial = backup.ensure_initial_snapshot(0)
    assert initial.file_count == 4
    assert initial.archive == tmp_path / ".coder-backup" / "initial.zip"
    manifest = backup.read_manifest()
    assert manifest["generation"] == 3
    initial_manifest = backup.read_manifest(initial=True)
    assert initial_manifest["generation"] == 0
    assert {item["path"] for item in manifest["files"]} == {
        "main.py",
        "helper.pyi",
        "tests/test_main.py",
        "analysis.ipynb",
    }
    assert backup.contains_text("main.py", "VERSION = 1\n")
    assert not backup.contains_text(".env", "SECRET=x")


def test_backup_does_not_regress_to_an_older_generation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = CoderBackupStore(workspace)
    (workspace / "main.py").write_text("value = 1\n", encoding="utf-8")
    store.snapshot(6)

    (workspace / "main.py").write_text("value = 2\n", encoding="utf-8")
    result = store.snapshot(5)

    assert result.generation == 6
    assert store.read_manifest()["generation"] == 6
    assert store.contains_text("main.py", "value = 1\n")


def test_backup_is_inside_workspace_but_hidden_from_workspacefs(tmp_path):
    backup = CoderBackupStore(tmp_path)
    backup.snapshot(0)
    fs = CoderHarness(str(tmp_path), sandbox=SimpleNamespace()).fs

    assert backup.root == tmp_path / ".coder-backup"
    assert backup.root.is_relative_to(tmp_path)
    assert all(".coder-backup" not in path for path in fs.list_files())

    from coder.filesystem import WorkspaceSecurityError

    with __import__("pytest").raises(WorkspaceSecurityError):
        fs.read_text(".coder-backup/latest.zip")


def test_harness_updates_backup_only_after_passing_pytest(tmp_path):
    (tmp_path / "main.py").write_bytes(b"VERSION = 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        "def test_main():\n    assert True\n", encoding="utf-8"
    )

    class FakeSandbox:
        def __init__(self):
            self.passes = [True, False, True]

        def run(self, kind, paths):
            passed = self.passes.pop(0)
            return SandboxResult(0 if passed else 1, "1 passed" if passed else "", "")

    sandbox = FakeSandbox()
    harness = CoderHarness(str(tmp_path), sandbox=sandbox)
    state = CoderState("repair")

    # This test exercises backup timing, not the separate full-read write guard.
    harness.execute("READ_FILE", {"path": "main.py"}, state)
    harness.execute(
        "WRITE_FILE",
        {"path": "main.py", "content": "VERSION = 1\n"},
        state,
    )
    first = harness.execute("RUN_PYTEST", {"paths": []}, state)
    assert first["passed"] is True
    assert first["backup_ok"] is True
    assert backup_text(harness, "main.py") == "VERSION = 1\n"
    first_generation = state.backup_generation

    harness.execute(
        "PATCH_FILE",
        {"path": "main.py", "old_text": "VERSION = 1", "new_text": "VERSION = 2"},
        state,
    )
    second = harness.execute("RUN_PYTEST", {"paths": []}, state)
    assert second["passed"] is False
    assert second["backup_ok"] is False
    assert state.backup_generation == first_generation
    assert backup_text(harness, "main.py") == "VERSION = 1\n"

    third = harness.execute("RUN_PYTEST", {"paths": []}, state)
    assert third["passed"] is True
    assert third["backup_ok"] is True
    assert "backup_path" not in third
    assert state.backup_generation == state.modification_generation
    assert backup_text(harness, "main.py") == "VERSION = 2\n"



def test_initial_backup_is_created_before_agent_modifications_and_never_overwritten(tmp_path):
    (tmp_path / "main.py").write_bytes(b"VERSION = 0\n")
    backup = CoderBackupStore(tmp_path)

    first = backup.ensure_initial_snapshot(0)
    (tmp_path / "main.py").write_bytes(b"VERSION = 1\n")
    backup.snapshot(1)
    again = backup.ensure_initial_snapshot(0)

    assert first.archive == again.archive
    assert again.generation == 0
    assert backup.contains_text("main.py", "VERSION = 0\n", initial=True)
    assert not backup.contains_text("main.py", "VERSION = 1\n", initial=True)
    assert backup.contains_text("main.py", "VERSION = 1\n")


def test_backup_archives_are_separate_initial_and_latest(tmp_path):
    (tmp_path / "main.py").write_bytes(b"initial\n")
    backup = CoderBackupStore(tmp_path)
    backup.ensure_initial_snapshot(0)

    (tmp_path / "main.py").write_bytes(b"latest\n")
    backup.snapshot(1)

    assert backup.root / backup.INITIAL_ARCHIVE_NAME != backup.root / backup.ARCHIVE_NAME
    assert backup.contains_text("main.py", "initial\n", initial=True)
    assert backup.contains_text("main.py", "latest\n", initial=False)

def test_goal_verifier_requires_backup_for_latest_verified_generation(tmp_path):
    backup_state = SimpleNamespace(generation=-1)

    class FakeBackup:
        root = tmp_path / "backup"
        ARCHIVE_NAME = "latest.zip"

        def snapshot(self, generation):
            backup_state.generation = generation
            self.root.mkdir(parents=True, exist_ok=True)
            archive = self.root / "latest.zip"
            archive.write_bytes(b"backup")
            return SimpleNamespace(
                generation=generation,
                archive=archive,
                file_count=1,
            )

        def ensure_initial_snapshot(self, generation=0):
            return SimpleNamespace(generation=generation)

        def has_initial_snapshot(self):
            return True

        def has_latest_snapshot(self):
            return (self.root / "latest.zip").is_file()

    class FakeSandbox:
        def run(self, kind, paths):
            return SandboxResult(0, "1 passed", "")

    harness = CoderHarness(
        str(tmp_path),
        sandbox=FakeSandbox(),
        backup=FakeBackup(),
    )
    state = CoderState("repair")
    harness.execute(
        "WRITE_FILE",
        {"path": "a.py", "content": "print(1)\n"},
        state,
    )

    result = harness.execute("RUN_PYTEST", {"paths": []}, state)
    assert result["backup_ok"] is True
    assert state.backup_generation == state.modification_generation
    harness.backup.ensure_initial_snapshot(0)

    state.created_tests.add("tests/test_a.py")
    verified = harness.execute("VERIFY_GOAL", {}, state)
    assert verified["verified"] is True

    os.remove(harness.backup.root / harness.backup.ARCHIVE_NAME)
    state.goal_verified = False
    hidden = harness.execute("VERIFY_GOAL", {}, state)
    assert hidden["verified"] is False


def backup_text(harness: CoderHarness, path: str) -> str:
    with ZipFile(harness.backup.root / harness.backup.ARCHIVE_NAME) as archive:
        return archive.read(path).decode("utf-8")

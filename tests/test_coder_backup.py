from __future__ import annotations

import os
from types import SimpleNamespace
from zipfile import ZipFile

from coder.backup import CoderBackupStore
from coder.harness import CoderHarness
from coder.sandbox import SandboxResult
from coder.state import CoderState


def test_backup_snapshot_contains_only_python_code_and_manifest(tmp_path):
    (tmp_path / "main.py").write_text("VERSION = 1\n", encoding="utf-8")
    (tmp_path / "helper.pyi").write_text("VERSION: int\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not code", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=x", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text(
        "def test_main():\n    assert True\n", encoding="utf-8"
    )

    backup = CoderBackupStore(tmp_path)
    snapshot = backup.snapshot(3)

    assert snapshot.generation == 3
    assert snapshot.file_count == 3
    assert snapshot.archive == tmp_path.parent / f"{tmp_path.name}.coder-backup" / "latest.zip"
    manifest = backup.read_manifest()
    assert manifest["generation"] == 3
    assert {item["path"] for item in manifest["files"]} == {
        "main.py",
        "helper.pyi",
        "tests/test_main.py",
    }
    assert backup.contains_text("main.py", "VERSION = 1\n")
    assert not backup.contains_text(".env", "SECRET=x")


def test_backup_is_outside_model_workspace_and_not_readable_by_workspacefs(tmp_path):
    backup = CoderBackupStore(tmp_path)
    backup.snapshot(0)
    fs = CoderHarness(str(tmp_path), sandbox=SimpleNamespace()).fs

    assert backup.root.parent == tmp_path.parent
    assert backup.root != tmp_path
    assert all("coder-backup" not in path for path in fs.list_files())

    from coder.filesystem import WorkspaceSecurityError

    with __import__("pytest").raises(WorkspaceSecurityError):
        fs.read_text("../" + backup.root.name + "/latest.zip")


def test_harness_updates_backup_only_after_passing_pytest(tmp_path):
    (tmp_path / "main.py").write_text("VERSION = 1\n", encoding="utf-8")
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


def test_goal_verifier_requires_backup_for_latest_verified_generation(tmp_path):
    backup_state = SimpleNamespace(generation=-1)

    class FakeBackup:
        root = tmp_path / "backup"

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

    state.created_tests.add("tests/test_a.py")
    verified = harness.execute("VERIFY_GOAL", {}, state)
    assert verified["verified"] is True

    os.remove(state.backup_path)
    state.goal_verified = False
    hidden = harness.execute("VERIFY_GOAL", {}, state)
    assert hidden["verified"] is False


def backup_text(harness: CoderHarness, path: str) -> str:
    with ZipFile(harness.backup.root / harness.backup.ARCHIVE_NAME) as archive:
        return archive.read(path).decode("utf-8")

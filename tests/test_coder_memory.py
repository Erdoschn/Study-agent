from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from coder.memory import CoderMemoryStore
from core.redaction import contains_suspected_secret, redact_sensitive_text


def test_redaction_removes_credentials_tokens_and_host_paths():
    source = (
        "api_key=super-secret "
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345 "
        "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 "
        "C:\\Users\\Alice\\project\\main.py "
        "file:///home/alice/.env"
    )

    result = redact_sensitive_text(source)

    assert "super-secret" not in result
    assert "abcdefghijklmnopqrstuvwxyz012345" not in result
    assert "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890" not in result
    assert "C:\\Users\\Alice" not in result
    assert "file:///home/alice" not in result
    assert "[REDACTED]" in result or "[REDACTED CREDENTIAL]" in result
    assert "[REDACTED PATH]" in result


def test_redaction_handles_private_key_blocks_and_preserves_ordinary_text():
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        "private-key-material-must-not-be-logged\n"
        "-----END PRIVATE KEY-----"
    )
    result = redact_sensitive_text(f"status=failed\n{private_key}\nretry available")

    assert "private-key-material-must-not-be-logged" not in result
    assert "[REDACTED PRIVATE KEY]" in result
    assert "status=failed" in result
    assert "retry available" in result


def test_secret_detector_catches_common_credentials_and_key_markers():
    assert contains_suspected_secret("api_key = value")
    assert contains_suspected_secret("-----BEGIN PRIVATE KEY-----")
    assert contains_suspected_secret("sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
    assert not contains_suspected_secret("simple harmless status message")


def test_memory_persistence_redacts_sensitive_values(tmp_path):
    store = CoderMemoryStore(tmp_path)
    store._write_unlocked({
        "version": 1,
        "runs": [{
            "id": "run-1",
            "request": "fix this file; api_key=do-not-store-this",
            "summary": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
            "error": "failed at C:\\Users\\Alice\\secret.py",
            "feedback": "",
        }],
        "strategies": [],
        "experiences": [],
        "technologies": [],
    })

    saved = store.path.read_text(encoding="utf-8")
    loaded = store.load()
    serialized = str(loaded)

    assert "do-not-store-this" not in saved
    assert "abcdefghijklmnopqrstuvwxyz012345" not in saved
    assert "C:\\Users\\Alice" not in saved
    assert "do-not-store-this" not in serialized
    assert "abcdefghijklmnopqrstuvwxyz012345" not in serialized


def test_memory_feedback_updates_from_separate_instances_are_not_lost(tmp_path):
    stores = [CoderMemoryStore(tmp_path) for _ in range(8)]
    stores[0]._write_unlocked({
        "version": 1,
        "runs": [
            {"id": f"run-{index}", "project": "audit-demo", "feedback": ""}
            for index in range(8)
        ],
        "strategies": [],
        "experiences": [],
        "technologies": [],
    })

    def save_feedback(index: int) -> None:
        assert stores[index].record_feedback(
            run_id=f"run-{index}",
            feedback=f"review note {index}",
        ) is not None

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save_feedback, range(8)))

    saved = stores[0].load()["runs"]
    assert {run.get("feedback") for run in saved} >= {
        f"review note {index}" for index in range(8)
    }


def test_memory_rejects_symlink_and_corrupt_files(tmp_path):
    target = tmp_path / "external-memory.json"
    target.write_text('{"version":1,"runs":[]}', encoding="utf-8")
    memory_path = tmp_path / ".coder-memory.json"
    try:
        memory_path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable in this environment")

    with pytest.raises(PermissionError, match="符号链接"):
        CoderMemoryStore(tmp_path)


def test_memory_does_not_replace_corrupt_json_with_empty_history(tmp_path):
    store = CoderMemoryStore(tmp_path)
    store.path.write_text("{this is not JSON", encoding="utf-8")

    with pytest.raises(RuntimeError, match="拒绝覆盖"):
        store.load()

    assert store.path.read_text(encoding="utf-8") == "{this is not JSON"


def test_memory_feedback_rejects_oversized_input_before_truncating(tmp_path):
    store = CoderMemoryStore(tmp_path)
    store._write_unlocked({
        "version": 1,
        "runs": [{"id": "run-1", "feedback": ""}],
        "strategies": [],
        "experiences": [],
        "technologies": [],
    })

    with pytest.raises(ValueError, match="16 KiB"):
        store.record_feedback(run_id="run-1", feedback="x" * (16 * 1024 + 1))

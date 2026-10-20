from __future__ import annotations

import pytest

from study_agent.domain.task_dossier import DossierFact, FactOrigin, TaskDossier
from study_agent.infrastructure.dossier_sqlite import SQLiteTaskDossierRepository


def _dossier(user_id: str, version: int, statement: str) -> TaskDossier:
    return TaskDossier(
        user_id=user_id,
        task_id="same-task-id",
        version=version,
        facts=(
            DossierFact(
                id=f"fact-{version}",
                statement=statement,
                origin=FactOrigin.USER_STATED,
            ),
        ),
    )


def test_sqlite_repository_isolates_users_and_preserves_versions(tmp_path):
    repository = SQLiteTaskDossierRepository(tmp_path / "dossiers.sqlite3")
    alice_v1 = _dossier("alice", 1, "Alice's original condition")
    bob_v1 = _dossier("bob", 1, "Bob's independent condition")

    repository.save(alice_v1)
    repository.save(bob_v1)
    repository.save(_dossier("alice", 2, "Alice's amended condition"))

    assert repository.load_latest(user_id="alice", task_id="same-task-id").version == 2
    assert repository.load_latest(user_id="alice", task_id="same-task-id").facts[0].statement == "Alice's amended condition"
    assert repository.load_latest(user_id="bob", task_id="same-task-id") == bob_v1
    assert repository.load_version(user_id="alice", task_id="same-task-id", version=1) == alice_v1
    assert repository.load_latest(user_id="charlie", task_id="same-task-id") is None


def test_sqlite_repository_rejects_overwrite_and_skipped_versions(tmp_path):
    repository = SQLiteTaskDossierRepository(tmp_path / "dossiers.sqlite3")
    repository.save(_dossier("alice", 1, "Original"))

    with pytest.raises(ValueError, match="version must be 2"):
        repository.save(_dossier("alice", 1, "Attempted overwrite"))

    with pytest.raises(ValueError, match="version must be 2"):
        repository.save(_dossier("alice", 3, "Skipped version"))

    assert repository.load_latest(user_id="alice", task_id="same-task-id").facts[0].statement == "Original"


def test_sqlite_repository_validates_scope_and_version(tmp_path):
    repository = SQLiteTaskDossierRepository(tmp_path / "dossiers.sqlite3")

    with pytest.raises(ValueError, match="user_id"):
        repository.load_latest(user_id="", task_id="task")

    with pytest.raises(ValueError, match="version"):
        repository.load_version(user_id="alice", task_id="task", version=0)


def test_sqlite_repository_uses_hardened_storage_path(tmp_path):
    repository = SQLiteTaskDossierRepository(tmp_path / "private" / "dossiers.sqlite3")
    assert repository.database_path.exists()
    assert repository.database_path.is_file()

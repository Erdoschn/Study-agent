from __future__ import annotations

import pytest

from study_agent.domain.answer_revision import (
    AnswerRevision,
    CoverageEntry,
    CoverageLedger,
    CoverageOutcome,
)
from study_agent.infrastructure.answer_revision_sqlite import (
    SQLiteAnswerRevisionRepository,
    SQLiteCoverageLedgerRepository,
)


def _revision(number: int, revision_id: str, *, parent: str | None = None, text: str | None = None):
    return AnswerRevision(
        user_id="student-a",
        task_id="task-a",
        revision_id=revision_id,
        dossier_version=1,
        answer_text=text or f"answer revision {number}",
        parent_revision_id=parent,
        revision_number=number,
        knowledge_point_ids=("electric-field",),
    )


def _ledger(revision_id: str, *, version: int = 1, outcome=CoverageOutcome.SATISFIED):
    return CoverageLedger(
        user_id="student-a",
        task_id="task-a",
        answer_revision_id=revision_id,
        dossier_version=version,
        entries=(CoverageEntry(
            user_id="student-a",
            task_id="task-a",
            answer_revision_id=revision_id,
            dossier_version=version,
            requirement_id="req-units",
            outcome=outcome,
            evidence_refs=("answer-span:0-18",),
        ),),
    )


def test_sqlite_answer_revision_repository_enforces_linear_append_only_versions(tmp_path):
    repo = SQLiteAnswerRevisionRepository(tmp_path / "answers.sqlite3")
    v1 = _revision(1, "answer-1")
    v2 = _revision(2, "answer-2", parent="answer-1", text="revised answer")

    assert repo.load_latest(user_id="student-a", task_id="task-a") is None
    repo.save(v1)
    repo.save(v2)

    assert repo.load_latest(user_id="student-a", task_id="task-a") == v2
    assert repo.load_revision(user_id="student-a", task_id="task-a", revision_id="answer-1") == v1
    assert repo.load_latest(user_id="student-b", task_id="task-a") is None

    with pytest.raises(ValueError, match="parent"):
        repo.save(_revision(3, "answer-3", parent="answer-1"))


def test_sqlite_answer_revision_repository_rejects_skipped_versions_and_invalid_scope(tmp_path):
    repo = SQLiteAnswerRevisionRepository(tmp_path / "answers.sqlite3")
    with pytest.raises(ValueError, match="revision number must be 1"):
        repo.save(_revision(2, "answer-2", parent="answer-1"))
    with pytest.raises(ValueError, match="must not be empty"):
        repo.load_latest(user_id="", task_id="task-a")


def test_sqlite_coverage_repository_requires_matching_persisted_revision_and_is_append_only(tmp_path):
    database = tmp_path / "answers.sqlite3"
    revisions = SQLiteAnswerRevisionRepository(database)
    coverage = SQLiteCoverageLedgerRepository(database)
    revision = _revision(1, "answer-1")
    revisions.save(revision)

    ledger = _ledger("answer-1")
    coverage.save(ledger)
    assert coverage.load(
        user_id="student-a", task_id="task-a", answer_revision_id="answer-1"
    ) == ledger

    with pytest.raises(ValueError, match="already exists"):
        coverage.save(_ledger("answer-1", outcome=CoverageOutcome.OMITTED))

    with pytest.raises(ValueError, match="must be persisted"):
        coverage.save(_ledger("missing-answer"))

    with pytest.raises(ValueError, match="dossier version"):
        coverage.save(_ledger("answer-1", version=2))


def test_sqlite_coverage_repository_does_not_cross_user_or_task_scope(tmp_path):
    database = tmp_path / "answers.sqlite3"
    revisions = SQLiteAnswerRevisionRepository(database)
    coverage = SQLiteCoverageLedgerRepository(database)
    revisions.save(_revision(1, "answer-1"))
    coverage.save(_ledger("answer-1"))

    assert coverage.load(
        user_id="student-b", task_id="task-a", answer_revision_id="answer-1"
    ) is None
    assert coverage.load(
        user_id="student-a", task_id="other-task", answer_revision_id="answer-1"
    ) is None

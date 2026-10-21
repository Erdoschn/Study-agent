"""SQLite repositories for immutable answer revisions and coverage ledgers."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from study_agent.domain.answer_revision import AnswerRevision, CoverageLedger
from study_agent.infrastructure.storage_security import ensure_safe_sqlite_path


class _SQLiteAnswerStore:
    def __init__(self, database_path: str | Path, *, timeout_seconds: float = 5.0) -> None:
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_seconds must be a positive number") from exc
        if timeout <= 0:
            raise ValueError("timeout_seconds must be a positive number")
        self.database_path = ensure_safe_sqlite_path(Path(database_path))
        self.timeout_seconds = timeout
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS answer_revisions (
                    user_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    revision_number INTEGER NOT NULL CHECK (revision_number > 0),
                    revision_id TEXT NOT NULL,
                    dossier_version INTEGER NOT NULL CHECK (dossier_version > 0),
                    content_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (user_id, task_id, revision_number),
                    UNIQUE (user_id, task_id, revision_id)
                );
                CREATE INDEX IF NOT EXISTS idx_answer_revisions_latest
                ON answer_revisions(user_id, task_id, revision_number DESC);
                CREATE TABLE IF NOT EXISTS coverage_ledgers (
                    user_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    answer_revision_id TEXT NOT NULL,
                    dossier_version INTEGER NOT NULL CHECK (dossier_version > 0),
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (user_id, task_id, answer_revision_id)
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _scope(user_id: object, task_id: object) -> tuple[str, str]:
        user = str(user_id or "").strip()
        task = str(task_id or "").strip()
        if not user or not task:
            raise ValueError("user_id and task_id must not be empty")
        if any(len(value) > 256 or any(ord(char) < 32 for char in value) for value in (user, task)):
            raise ValueError("user/task scope must use safe identifiers")
        return user, task


class SQLiteAnswerRevisionRepository(_SQLiteAnswerStore):
    """Persist immutable linear answer revisions under user/task scope."""

    @staticmethod
    def _decode(row: sqlite3.Row, user: str, task: str) -> AnswerRevision:
        revision = AnswerRevision.from_dict(json.loads(row["payload_json"]))
        if (
            revision.user_id != user
            or revision.task_id != task
            or revision.revision_id != row["revision_id"]
            or revision.revision_number != row["revision_number"]
            or revision.dossier_version != row["dossier_version"]
            or revision.content_sha256 != row["content_sha256"]
        ):
            raise ValueError("stored answer revision does not match its database key or hash")
        return revision

    def load_latest(self, *, user_id: str, task_id: str) -> AnswerRevision | None:
        user, task = self._scope(user_id, task_id)
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM answer_revisions
                WHERE user_id = ? AND task_id = ?
                ORDER BY revision_number DESC LIMIT 1
                """,
                (user, task),
            ).fetchone()
        finally:
            connection.close()
        return self._decode(row, user, task) if row is not None else None

    def load_revision(
        self, *, user_id: str, task_id: str, revision_id: str
    ) -> AnswerRevision | None:
        user, task = self._scope(user_id, task_id)
        revision = str(revision_id or "").strip()
        if not revision:
            raise ValueError("revision_id must not be empty")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM answer_revisions
                WHERE user_id = ? AND task_id = ? AND revision_id = ?
                """,
                (user, task, revision),
            ).fetchone()
        finally:
            connection.close()
        return self._decode(row, user, task) if row is not None else None

    def save(self, revision: AnswerRevision) -> None:
        if not isinstance(revision, AnswerRevision):
            raise TypeError("revision must be an AnswerRevision")
        user, task = self._scope(revision.user_id, revision.task_id)
        payload = json.dumps(
            revision.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                """
                SELECT revision_number, revision_id FROM answer_revisions
                WHERE user_id = ? AND task_id = ?
                ORDER BY revision_number DESC LIMIT 1
                """,
                (user, task),
            ).fetchone()
            expected_number = 1 if latest is None else int(latest["revision_number"]) + 1
            if revision.revision_number != expected_number:
                raise ValueError(
                    f"answer revision number must be {expected_number}; received {revision.revision_number}"
                )
            expected_parent = None if latest is None else latest["revision_id"]
            if revision.parent_revision_id != expected_parent:
                raise ValueError("answer revision parent does not match the current latest revision")
            connection.execute(
                """
                INSERT INTO answer_revisions
                    (user_id, task_id, revision_number, revision_id, dossier_version, content_sha256, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user, task, revision.revision_number, revision.revision_id,
                    revision.dossier_version, revision.content_sha256, payload,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class SQLiteCoverageLedgerRepository(_SQLiteAnswerStore):
    """Persist one immutable coverage snapshot per answer revision."""

    def load(
        self, *, user_id: str, task_id: str, answer_revision_id: str
    ) -> CoverageLedger | None:
        user, task = self._scope(user_id, task_id)
        revision_id = str(answer_revision_id or "").strip()
        if not revision_id:
            raise ValueError("answer_revision_id must not be empty")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT payload_json FROM coverage_ledgers
                WHERE user_id = ? AND task_id = ? AND answer_revision_id = ?
                """,
                (user, task, revision_id),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        ledger = CoverageLedger.from_dict(json.loads(row["payload_json"]))
        if (
            ledger.user_id != user
            or ledger.task_id != task
            or ledger.answer_revision_id != revision_id
        ):
            raise ValueError("stored coverage ledger does not match its database key")
        return ledger

    def save(self, ledger: CoverageLedger) -> None:
        if not isinstance(ledger, CoverageLedger):
            raise TypeError("ledger must be a CoverageLedger")
        user, task = self._scope(ledger.user_id, ledger.task_id)
        payload = json.dumps(
            ledger.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO coverage_ledgers
                    (user_id, task_id, answer_revision_id, dossier_version, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user, task, ledger.answer_revision_id,
                    ledger.dossier_version, payload,
                ),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ValueError(
                "coverage ledger already exists for this answer revision; create a new answer revision instead of overwriting"
            ) from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

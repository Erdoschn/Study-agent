"""SQLite adapter for immutable, user/task-scoped Task Dossier snapshots."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from study_agent.infrastructure.storage_security import ensure_safe_sqlite_path
from study_agent.domain.task_dossier import TaskDossier


class SQLiteTaskDossierRepository:
    """Persist append-only Dossier versions in a guarded SQLite database.

    The storage key includes both user and task IDs. Callers must pass the
    trusted authenticated user identity, not an untrusted username from a
    request body. Each task starts at version 1 and advances without gaps.
    """

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
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=self.timeout_seconds,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_dossiers (
                    user_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (user_id, task_id, version)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_dossiers_latest
                ON task_dossiers(user_id, task_id, version DESC)
                """
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _validate_scope(user_id: object, task_id: object) -> tuple[str, str]:
        user = str(user_id or "").strip()
        task = str(task_id or "").strip()
        if not user:
            raise ValueError("user_id must not be empty")
        if not task:
            raise ValueError("task_id must not be empty")
        return user, task

    def load_latest(self, *, user_id: str, task_id: str) -> TaskDossier | None:
        user, task = self._validate_scope(user_id, task_id)
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT payload_json FROM task_dossiers
                WHERE user_id = ? AND task_id = ?
                ORDER BY version DESC LIMIT 1
                """,
                (user, task),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        data = json.loads(row["payload_json"])
        dossier = TaskDossier.from_dict(data)
        if dossier.user_id != user or dossier.task_id != task:
            # Defense in depth against malformed or manually tampered records.
            raise ValueError("stored dossier scope does not match the requested scope")
        return dossier

    def load_version(
        self,
        *,
        user_id: str,
        task_id: str,
        version: int,
    ) -> TaskDossier | None:
        user, task = self._validate_scope(user_id, task_id)
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("version must be a positive integer")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT payload_json FROM task_dossiers
                WHERE user_id = ? AND task_id = ? AND version = ?
                """,
                (user, task, version),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        dossier = TaskDossier.from_dict(json.loads(row["payload_json"]))
        if dossier.user_id != user or dossier.task_id != task or dossier.version != version:
            raise ValueError("stored dossier identity does not match its database key")
        return dossier

    def save(self, dossier: TaskDossier) -> None:
        if not isinstance(dossier, TaskDossier):
            raise TypeError("dossier must be a TaskDossier")
        user, task = self._validate_scope(dossier.user_id, dossier.task_id)
        payload = json.dumps(
            dossier.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS current_version
                FROM task_dossiers WHERE user_id = ? AND task_id = ?
                """,
                (user, task),
            ).fetchone()
            current_version = int(row["current_version"] or 0)
            expected_version = current_version + 1
            if dossier.version != expected_version:
                raise ValueError(
                    f"dossier version must be {expected_version} for this task; "
                    f"received {dossier.version}"
                )
            connection.execute(
                """
                INSERT INTO task_dossiers
                    (user_id, task_id, version, schema_version, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user, task, dossier.version, dossier.schema_version, payload),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

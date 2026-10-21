"""SQLite persistence for immutable, versioned curriculum prerequisite DAGs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from study_agent.domain.prerequisite_graph import PrerequisiteGraph
from study_agent.infrastructure.storage_security import ensure_safe_sqlite_path


class SQLitePrerequisiteGraphRepository:
    """Persist shared curriculum graphs by graph ID and immutable version.

    Per-learner mastery is intentionally not stored here. This repository owns
    the canonical prerequisite DAG only; user-specific learning state belongs
    in a separate user-scoped learner store.
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
        connection = sqlite3.connect(str(self.database_path), timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prerequisite_graph_versions (
                    graph_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (graph_id, version)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_prerequisite_graph_latest
                ON prerequisite_graph_versions(graph_id, version DESC)
                """
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _validate_graph_id(graph_id: object) -> str:
        value = str(graph_id or "").strip()
        if not value:
            raise ValueError("graph_id must not be empty")
        if len(value) > 256 or any(ord(ch) < 32 for ch in value):
            raise ValueError("graph_id must be a safe identifier")
        return value

    def load_latest(self, *, graph_id: str) -> PrerequisiteGraph | None:
        key = self._validate_graph_id(graph_id)
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT payload_json FROM prerequisite_graph_versions
                WHERE graph_id = ? ORDER BY version DESC LIMIT 1
                """,
                (key,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        graph = PrerequisiteGraph.from_dict(json.loads(row["payload_json"]))
        if graph.graph_id != key:
            raise ValueError("stored prerequisite graph scope does not match the requested graph")
        return graph

    def load_version(self, *, graph_id: str, version: int) -> PrerequisiteGraph | None:
        key = self._validate_graph_id(graph_id)
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("version must be a positive integer")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT payload_json FROM prerequisite_graph_versions
                WHERE graph_id = ? AND version = ?
                """,
                (key, version),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        graph = PrerequisiteGraph.from_dict(json.loads(row["payload_json"]))
        if graph.graph_id != key or graph.version != version:
            raise ValueError("stored prerequisite graph identity does not match its database key")
        return graph

    def save(self, graph: PrerequisiteGraph) -> None:
        if not isinstance(graph, PrerequisiteGraph):
            raise TypeError("graph must be a PrerequisiteGraph")
        graph_id = self._validate_graph_id(graph.graph_id)
        payload = json.dumps(
            graph.to_dict(),
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
                FROM prerequisite_graph_versions WHERE graph_id = ?
                """,
                (graph_id,),
            ).fetchone()
            expected_version = int(row["current_version"] or 0) + 1
            if graph.version != expected_version:
                raise ValueError(
                    f"graph version must be {expected_version}; received {graph.version}"
                )
            connection.execute(
                """
                INSERT INTO prerequisite_graph_versions
                    (graph_id, version, schema_version, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (graph_id, graph.version, graph.schema_version, payload),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

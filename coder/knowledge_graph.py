from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class CoderKnowledgeGraph:
    """Persistent engineering knowledge graph kept outside Coder project sandboxes."""

    SCHEMA_VERSION = 1
    MAX_NODES = 1000
    MAX_EDGES = 3000

    def __init__(self, workspace_root: str | Path):
        self.root = Path(workspace_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / ".coder-knowledge.sqlite3"
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    last_seen REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    last_seen REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id, relation)
                );
                INSERT OR REPLACE INTO metadata(key, value)
                    VALUES ('schema_version', '1');
                """
            )

    @staticmethod
    def _id(name: str, node_type: str) -> str:
        raw = f"{node_type}:{str(name or '').strip().casefold()}"
        return re.sub(r"[^a-z0-9_:-]+", "_", raw)[:180]

    def _node(self, db: sqlite3.Connection, name: str, node_type: str) -> str:
        clean = str(name or "").strip()
        if not clean:
            return ""
        node_id = self._id(clean, node_type)
        now = time.time()
        db.execute(
            """
            INSERT INTO nodes(id, name, node_type, count, last_seen, metadata_json)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                count = nodes.count + 1,
                last_seen = excluded.last_seen,
                metadata_json = excluded.metadata_json
            """,
            (node_id, clean[:500], node_type[:80], now, "{}"),
        )
        return node_id

    def _edge(
        self,
        db: sqlite3.Connection,
        source_id: str,
        target_id: str,
        relation: str,
        confidence: float = 0.0,
    ) -> None:
        if not source_id or not target_id or source_id == target_id:
            return
        now = time.time()
        db.execute(
            """
            INSERT INTO edges(source_id, target_id, relation, count, confidence, last_seen)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                count = edges.count + 1,
                confidence = MAX(edges.confidence, excluded.confidence),
                last_seen = excluded.last_seen
            """,
            (source_id, target_id, relation[:80], max(0.0, min(1.0, float(confidence))), now),
        )

    def record_run(self, entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            return
        project = str(entry.get("project", "")).strip()
        if not project:
            return

        with self._lock, self._connect() as db:
            project_id = self._node(db, project, "project")

            for technology in entry.get("technologies", []):
                tech_id = self._node(db, str(technology), "technology")
                self._edge(db, project_id, tech_id, "uses")

            strategy = str(entry.get("strategy", "")).strip()
            if strategy:
                strategy_id = self._node(db, strategy, "strategy")
                self._edge(db, project_id, strategy_id, "used_strategy")

            for experience in entry.get("experience", []):
                experience_id = self._node(db, str(experience), "experience")
                self._edge(db, project_id, experience_id, "learned_experience")

            actions = [
                str(item).strip().upper()
                for item in str(entry.get("strategy", "")).split("→")
                if str(item).strip()
            ]
            for source, target in zip(actions, actions[1:]):
                source_id = self._node(db, source, "action")
                target_id = self._node(db, target, "action")
                self._edge(db, source_id, target_id, "followed_by")

    def record_study_consultation(
        self,
        project: str,
        question: str,
        study_payload: dict[str, Any],
    ) -> None:
        project = str(project or "").strip()
        question = str(question or "").strip()
        if not project or not question:
            return
        with self._lock, self._connect() as db:
            project_id = self._node(db, project, "project")
            concepts = []
            domain = str(study_payload.get("domain", "") or "").strip()
            if domain:
                concepts.append(domain)
            learner_context = study_payload.get("learner_context", {})
            if isinstance(learner_context, dict):
                for key in ("weak_concepts", "learning_concepts", "unassessed_concepts"):
                    items = learner_context.get(key, [])
                    if isinstance(items, list):
                        concepts.extend(str(item).strip() for item in items if str(item).strip())
            for concept in concepts[:12]:
                concept_id = self._node(db, concept, "study_concept")
                self._edge(db, project_id, concept_id, "consulted_study_concept", 0.6)
            request_id = self._node(db, question[:300], "study_question")
            self._edge(db, project_id, request_id, "asked_study_agent", 1.0)

    def snapshot(self, limit: int = 40) -> dict[str, list[dict[str, Any]]]:
        limit = max(1, min(int(limit), 200))
        with self._lock, self._connect() as db:
            nodes = [
                {
                    "name": row[1],
                    "type": row[2],
                    "count": row[3],
                    "last_seen": row[4],
                    "metadata": json.loads(row[5] or "{}"),
                }
                for row in db.execute(
                    "SELECT id, name, node_type, count, last_seen, metadata_json "
                    "FROM nodes ORDER BY last_seen DESC LIMIT ?",
                    (limit,),
                )
            ]
            edges = [
                {
                    "source": row[0],
                    "target": row[1],
                    "relation": row[2],
                    "count": row[3],
                    "confidence": row[4],
                    "last_seen": row[5],
                }
                for row in db.execute(
                    """
                    SELECT source_id, target_id, relation, count, confidence, last_seen
                    FROM edges ORDER BY last_seen DESC LIMIT ?
                    """,
                    (limit * 2,),
                )
            ]
        return {"nodes": nodes, "edges": edges}

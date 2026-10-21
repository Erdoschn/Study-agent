from __future__ import annotations

import pytest

from study_agent.domain.prerequisite_graph import KnowledgePoint, PrerequisiteEdge, PrerequisiteGraph
from study_agent.infrastructure.prerequisite_graph_sqlite import SQLitePrerequisiteGraphRepository


def _graph(version: int, *, description: str = "") -> PrerequisiteGraph:
    return PrerequisiteGraph(
        graph_id="math",
        version=version,
        points=(
            KnowledgePoint("fractions", "Fractions", description=description),
            KnowledgePoint("algebra", "Algebra"),
        ),
        edges=(PrerequisiteEdge("fractions", "algebra"),),
    )


def test_sqlite_prerequisite_repository_preserves_graph_versions(tmp_path):
    repo = SQLitePrerequisiteGraphRepository(tmp_path / "curriculum.sqlite3")
    graph_v1 = _graph(1)
    graph_v2 = _graph(2, description="Revised definition")

    assert repo.load_latest(graph_id="math") is None
    repo.save(graph_v1)
    repo.save(graph_v2)

    assert repo.load_version(graph_id="math", version=1) == graph_v1
    assert repo.load_latest(graph_id="math") == graph_v2


def test_sqlite_prerequisite_repository_rejects_overwrites_and_version_skips(tmp_path):
    repo = SQLitePrerequisiteGraphRepository(tmp_path / "curriculum.sqlite3")
    repo.save(_graph(1))
    with pytest.raises(ValueError, match="graph version must be 2"):
        repo.save(_graph(3))
    with pytest.raises(ValueError, match="graph version must be 2"):
        repo.save(_graph(1))


def test_sqlite_prerequisite_repository_isolates_graph_ids_and_validates_scope(tmp_path):
    repo = SQLitePrerequisiteGraphRepository(tmp_path / "curriculum.sqlite3")
    graph = _graph(1)
    repo.save(graph)
    assert repo.load_latest(graph_id="other-subject") is None
    with pytest.raises(ValueError, match="graph_id must not be empty"):
        repo.load_latest(graph_id="")

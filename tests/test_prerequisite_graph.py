from __future__ import annotations

import pytest

from study_agent.domain.prerequisite_graph import (
    KnowledgePoint,
    PrerequisiteEdge,
    PrerequisiteGraph,
)


def _graph(version: int = 1) -> PrerequisiteGraph:
    points = (
        KnowledgePoint(id="fractions", name="Fractions"),
        KnowledgePoint(id="algebra", name="Algebra"),
        KnowledgePoint(id="functions", name="Functions"),
        KnowledgePoint(id="derivatives", name="Derivatives"),
    )
    edges = (
        PrerequisiteEdge(prerequisite_id="algebra", target_id="functions", evidence_refs=("curriculum:math-1",)),
        PrerequisiteEdge(prerequisite_id="functions", target_id="derivatives"),
        PrerequisiteEdge(prerequisite_id="fractions", target_id="derivatives"),
    )
    return PrerequisiteGraph(graph_id="math-foundations", version=version, points=points, edges=edges)


def test_prerequisite_graph_is_a_dag_and_orders_foundations_before_targets():
    graph = _graph()
    order = graph.topological_order()
    assert order.index("algebra") < order.index("functions") < order.index("derivatives")
    assert order.index("fractions") < order.index("derivatives")
    assert [p.id for p in graph.prerequisites_for("derivatives")] == [
        point_id for point_id in order if point_id in {"algebra", "functions", "fractions"}
    ]
    assert {p.id for p in graph.prerequisites_for("derivatives", transitive=False)} == {
        "functions", "fractions"
    }


def test_prerequisite_graph_rejects_cycles_self_edges_and_missing_endpoints():
    points = (
        KnowledgePoint(id="a", name="A"),
        KnowledgePoint(id="b", name="B"),
    )
    with pytest.raises(ValueError, match="cycle"):
        PrerequisiteGraph(
            graph_id="cycle", version=1, points=points,
            edges=(PrerequisiteEdge("a", "b"), PrerequisiteEdge("b", "a")),
        )
    with pytest.raises(ValueError, match="own prerequisite"):
        PrerequisiteEdge("a", "a")
    with pytest.raises(ValueError, match="unknown point IDs"):
        PrerequisiteGraph(
            graph_id="missing", version=1, points=(points[0],),
            edges=(PrerequisiteEdge("a", "missing"),),
        )


def test_prerequisite_graph_rejects_duplicate_ids_edges_and_bad_confidence():
    with pytest.raises(ValueError, match="duplicate point IDs"):
        PrerequisiteGraph(
            graph_id="dupe", version=1,
            points=(KnowledgePoint("a", "A"), KnowledgePoint("a", "Again")),
        )
    with pytest.raises(ValueError, match="duplicate edges"):
        PrerequisiteGraph(
            graph_id="dupe-edge", version=1,
            points=(KnowledgePoint("a", "A"), KnowledgePoint("b", "B")),
            edges=(PrerequisiteEdge("a", "b"), PrerequisiteEdge("a", "b")),
        )
    with pytest.raises(ValueError, match="confidence"):
        PrerequisiteEdge("a", "b", confidence=float("nan"))


def test_prerequisite_gaps_find_transitive_unmastered_foundations_in_learning_order():
    graph = _graph()
    gaps = graph.prerequisite_gaps(
        "derivatives",
        learner_stages={
            "algebra": "weak",
            "functions": "unknown",
            "fractions": "mastered",
        },
    )
    assert [(gap.prerequisite_point_id, gap.learner_stage) for gap in gaps] == [
        ("algebra", "weak"),
        ("functions", "unknown"),
    ]
    assert all(gap.target_point_id == "derivatives" for gap in gaps)


def test_unknown_learner_topics_remain_unknown_and_do_not_count_as_mastered():
    graph = _graph()
    gaps = graph.prerequisite_gaps("functions", learner_stages={})
    assert [(gap.prerequisite_point_id, gap.learner_stage) for gap in gaps] == [
        ("algebra", "unknown")
    ]


def test_prerequisite_graph_serialization_round_trip_preserves_version_and_evidence():
    graph = _graph()
    restored = PrerequisiteGraph.from_dict(graph.to_dict())
    assert restored == graph
    assert restored.version == 1
    assert restored.edges[0].evidence_refs == ("curriculum:math-1",)

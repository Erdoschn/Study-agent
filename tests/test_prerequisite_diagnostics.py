from __future__ import annotations

import pytest

from study_agent.application.prerequisite_diagnostics import diagnose_prerequisite_gaps
from study_agent.domain.answer_revision import AnswerRevision
from study_agent.domain.prerequisite_graph import KnowledgePoint, PrerequisiteEdge, PrerequisiteGraph


def test_diagnostic_links_gaps_to_the_answer_revision_and_graph_version():
    graph = PrerequisiteGraph(
        graph_id="physics-e-and-v",
        version=3,
        points=(
            KnowledgePoint("potential", "Electric potential"),
            KnowledgePoint("voltage", "Voltage difference"),
            KnowledgePoint("field", "Electric field"),
        ),
        edges=(
            PrerequisiteEdge("potential", "voltage"),
            PrerequisiteEdge("voltage", "field"),
        ),
    )
    revision = AnswerRevision(
        user_id="student-a", task_id="question-9", revision_id="answer-2",
        dossier_version=4, answer_text="The electric field is ...",
        knowledge_point_ids=("field",),
    )
    diagnostic = diagnose_prerequisite_gaps(
        revision, graph, {"potential": "mastered", "voltage": "weak"}
    )
    assert diagnostic.user_id == "student-a"
    assert diagnostic.answer_revision_id == "answer-2"
    assert diagnostic.graph_id == "physics-e-and-v"
    assert diagnostic.graph_version == 3
    assert [(gap.prerequisite_point_id, gap.learner_stage) for gap in diagnostic.gaps] == [
        ("voltage", "weak")
    ]


def test_diagnostic_rejects_answer_annotations_for_unknown_graph_points():
    graph = PrerequisiteGraph(graph_id="math", version=1, points=(KnowledgePoint("a", "A"),))
    revision = AnswerRevision(
        user_id="student-a", task_id="task-a", revision_id="answer-1",
        dossier_version=1, answer_text="Answer", knowledge_point_ids=("not-in-graph",),
    )
    with pytest.raises(ValueError, match="unknown knowledge points"):
        diagnose_prerequisite_gaps(revision, graph, {})

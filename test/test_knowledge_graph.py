from core.knowledge_graph import KnowledgeGraph, POSTGRADUATE_THRESHOLD


def test_easy_correct_cannot_master():
    graph = KnowledgeGraph()
    for _ in range(6):
        graph.record_assessment(["attention"], True, confidence=1.0, difficulty="undergraduate")
    state = graph.nodes["attention"].learner
    assert state.learning_stage != "mastered"
    assert state.max_familiarity <= 0.89


def test_one_postgraduate_correct_cannot_master():
    graph = KnowledgeGraph()
    graph.record_assessment(["attention"], True, confidence=1.0, difficulty="postgraduate_plus")
    state = graph.nodes["attention"].learner
    assert state.learning_stage != "mastered"


def test_repeated_postgraduate_success_can_master():
    graph = KnowledgeGraph()
    for _ in range(5):
        graph.record_assessment(["attention"], True, confidence=0.95, difficulty="postgraduate_plus")
    state = graph.nodes["attention"].learner
    assert state.learning_stage == "mastered"
    assert state.highest_assessment_level >= POSTGRADUATE_THRESHOLD


def test_one_wrong_answer_does_not_immediately_mean_weak():
    graph = KnowledgeGraph()
    for _ in range(4):
        graph.record_assessment(["attention"], True, confidence=0.9, difficulty="postgraduate_plus")
    graph.record_assessment(["attention"], False, confidence=0.6, difficulty="postgraduate_plus")
    state = graph.nodes["attention"].learner
    assert state.learning_stage != "weak"


def test_lower_difficulty_does_not_reduce_previous_evidence_cap():
    graph = KnowledgeGraph()
    for _ in range(5):
        graph.record_assessment(["attention"], True, difficulty="postgraduate_plus")
    state = graph.nodes["attention"].learner
    before = state.max_familiarity
    graph.record_assessment(["attention"], True, difficulty="undergraduate")
    assert state.max_familiarity >= before


def test_student_state_sync_uses_assessed_stage_only():
    from core.state import StudentState

    graph = KnowledgeGraph()
    graph.add_concept("attention")
    graph.add_concept("transformer")
    for _ in range(5):
        graph.record_assessment(["attention"], True, difficulty="postgraduate_plus")
    graph.record_assessment(["transformer"], False, difficulty="graduate")

    student = StudentState()
    student.sync_from_knowledge_graph(graph)

    assert "attention" in student.known_topics
    assert "transformer" in student.weak_topics
    assert "unassessed concept" not in student.known_topics


def test_search_provenance_is_not_a_learner_weak_concept():
    graph = KnowledgeGraph()
    graph.learn_from_search("attention", [{
        "source": "wikipedia",
        "title": "Attention mechanism",
        "identifier": "1",
    }])

    context = graph.context_for("attention")
    assert "Attention mechanism" not in context["learner_context"]["weak_concepts"]



def test_student_state_sync_preserves_unassessed_external_topics():
    from core.state import StudentState

    graph = KnowledgeGraph()
    for _ in range(5):
        graph.record_assessment(["attention"], True, difficulty="postgraduate_plus")

    student = StudentState()
    student.known_topics.add("transformer")
    student.known_topics.add("attention")
    student.weak_topics.add("optimization")

    student.sync_from_knowledge_graph(graph)

    assert "transformer" in student.known_topics
    assert "attention" in student.known_topics
    assert "attention" not in student.weak_topics
    assert "optimization" in student.weak_topics



def test_knowledge_graph_stores_directed_semantic_relation():
    graph = KnowledgeGraph()
    graph.add_relation("attention", "transformer", "part_of", confidence=0.8)

    edge = graph.edges[("attention", "transformer", "part_of")]
    assert edge.source == "attention"
    assert edge.target == "transformer"
    assert edge.relation == "part_of"
    assert edge.confidence == 0.8
    assert graph.related_concepts("attention") == ["transformer"]



def test_invalid_learner_confidence_does_not_poison_state():
    graph = KnowledgeGraph()
    graph.record_assessment(["attention"], True, confidence=float("nan"), difficulty="graduate")
    state = graph.nodes["attention"].learner
    assert state.assessment_history[-1]["confidence"] is None
    assert state.confidence == 0.25



def test_relation_confidence_is_sanitized_at_graph_boundary():
    graph = KnowledgeGraph()
    graph.add_relation("attention", "transformer", "related_to", confidence=float("nan"))
    assert graph.edges[("attention", "transformer", "related_to")].confidence == 0.0

    graph.add_relation("attention", "transformer", "related_to", confidence=2)
    assert graph.edges[("attention", "transformer", "related_to")].confidence == 1.0

from core.knowledge_graph import KnowledgeGraph


def test_knowledge_graph_persists_across_instances(tmp_path):
    db_path = tmp_path / "knowledge_graph.sqlite3"

    first = KnowledgeGraph(storage_path=str(db_path))
    first.add_concept("attention", alias="self-attention")
    first.add_relation("attention", "transformer", "part_of", confidence=0.8)
    for _ in range(5):
        first.record_assessment(
            ["attention"],
            True,
            confidence=0.95,
            difficulty="postgraduate_plus",
        )

    second = KnowledgeGraph(storage_path=str(db_path))
    assert second.persistent is True
    assert "attention" in second.nodes
    assert second.nodes["attention"].aliases == ["self-attention"]
    assert second.nodes["attention"].learner.learning_stage == "mastered"
    assert second.nodes["attention"].learner.exposure_count == 5

    edge = second.edges[("attention", "transformer", "part_of")]
    assert edge.confidence == 0.8
    assert second.related_concepts("attention") == ["transformer"]


def test_knowledge_graph_without_storage_remains_in_memory_only(tmp_path):
    graph = KnowledgeGraph()
    graph.add_concept("attention")
    assert graph.persistent is False
    assert not (tmp_path / "knowledge_graph.sqlite3").exists()

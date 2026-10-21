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



def test_legacy_cyclic_depends_on_rows_are_filtered_on_load(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy-cyclic-graph.sqlite3"
    graph = KnowledgeGraph(storage_path=str(db_path))
    graph.add_concept("advanced")
    graph.add_concept("intermediate")
    graph.add_concept("basic")
    graph.add_relation("advanced", "intermediate", "depends_on")
    graph.add_relation("intermediate", "basic", "depends_on")
    graph.save()

    # Simulate a database created before the prerequisite DAG invariant existed.
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            "INSERT INTO edges(source, target, relation, confidence, evidence_refs_json, learner_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("basic", "advanced", "depends_on", 0.9, "[]",
             '{"familiarity":0.0,"confidence":0.0,"exposure_count":0,"successful_count":0,'
             '"last_seen":null,"learning_stage":"unknown","assessment_history":[],'
             '"max_familiarity":0.2,"highest_assessment_level":0.0}'),
        )
        connection.commit()
    finally:
        connection.close()

    restored = KnowledgeGraph(storage_path=str(db_path))
    assert ("advanced", "intermediate", "depends_on") in restored.edges
    assert ("intermediate", "basic", "depends_on") in restored.edges
    assert ("basic", "advanced", "depends_on") not in restored.edges

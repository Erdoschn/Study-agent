from core.evidence import EvidenceEngine, EvidenceStore


def test_evidence_store_deduplicates_and_preserves_records():
    store = EvidenceStore([{"source": "wiki", "identifier": "1", "title": "A", "abstract": "x", "harness_relevance": "DIRECT"}])
    added = store.add_many([
        {"source": "wiki", "identifier": "1", "title": "A", "abstract": "x"},
        {"source": "arxiv", "identifier": "2", "title": "B", "abstract": "y", "harness_relevance": "PARTIAL"},
    ])
    assert added == 1
    assert len(store.items) == 2
    assert store.prompt_view()[0]["identifier"] == "1"


def test_verify_is_conservative():
    evidence = EvidenceEngine.normalize("transformer attention", [{"source": "wiki", "title": "Transformer attention", "abstract": "attention mechanisms", "identifier": "1"}])
    supported = EvidenceEngine.verify("Transformer attention uses attention mechanisms", evidence)
    assert supported["verification_status"] == "SUPPORTED"
    assert supported["matched_evidence"] == [0]
    insufficient = EvidenceEngine.verify("Transformers were invented in 1900", evidence)
    assert insufficient["verification_status"] == "INSUFFICIENT"

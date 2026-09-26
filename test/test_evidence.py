from core.evidence import EvidenceEngine


def test_evidence_engine_labels_direct_and_irrelevant():
    results = EvidenceEngine.normalize(
        "transformer attention",
        [
            {
                "source": "wikipedia",
                "title": "Transformer attention",
                "abstract": "Transformer attention mechanisms.",
                "identifier": "1",
            },
            {
                "source": "wikipedia",
                "title": "Banana",
                "abstract": "A fruit.",
                "identifier": "2",
            },
        ],
    )
    assert results[0]["harness_relevance"] == "DIRECT"
    assert results[1]["harness_relevance"] == "IRRELEVANT"


def test_evidence_engine_deduplicates():
    item = {
        "source": "arxiv",
        "title": "Attention Is All You Need",
        "abstract": "transformer attention",
        "identifier": "1706.03762",
    }
    results = EvidenceEngine.normalize("transformer attention", [item, dict(item)])
    assert len(results) == 1


def test_evidence_engine_reports_coverage():
    evidence = EvidenceEngine.normalize(
        "transformer attention",
        [{
            "source": "wikipedia",
            "title": "Transformer attention",
            "abstract": "attention mechanisms",
            "identifier": "1",
        }],
    )
    coverage = EvidenceEngine.coverage("transformer attention", evidence)
    assert coverage["status"] in {"COVERED", "PARTIAL"}
    assert coverage["relevant_count"] == 1



def test_evidence_engine_matches_chinese_compound_terms():
    results = EvidenceEngine.normalize(
        "上下文缓存",
        [{
            "source": "wikipedia",
            "title": "上下文缓存机制",
            "abstract": "介绍上下文缓存的工作方式。",
            "identifier": "cn-1",
        }],
    )
    assert results[0]["harness_relevance"] == "DIRECT"



def test_evidence_verify_rejects_bare_topic_as_claim():
    evidence = EvidenceEngine.normalize(
        "attention",
        [{
            "source": "wikipedia",
            "title": "Attention",
            "abstract": "attention mechanisms in machine learning",
            "identifier": "1",
        }],
    )
    verification = EvidenceEngine.verify("attention", evidence)
    assert verification["verification_status"] == "UNCERTAIN"
    assert verification["matched_evidence"] == []



def test_evidence_verify_rejects_opposite_negation_english():
    evidence = EvidenceEngine.normalize(
        "attention uses fixed weights",
        [{
            "source": "wikipedia",
            "title": "Attention",
            "abstract": "attention uses fixed weights for every input",
            "identifier": "1",
        }],
    )
    verification = EvidenceEngine.verify("attention does not use fixed weights", evidence)
    assert verification["verification_status"] == "NOT_MATCHED"
    assert verification["matched_evidence"] == []


def test_evidence_verify_rejects_opposite_negation_chinese():
    evidence = EvidenceEngine.normalize(
        "attention 使用固定权重",
        [{
            "source": "wikipedia",
            "title": "Attention",
            "abstract": "attention 使用固定权重。",
            "identifier": "1",
        }],
    )
    verification = EvidenceEngine.verify("attention 不使用固定权重", evidence)
    assert verification["verification_status"] == "NOT_MATCHED"
    assert verification["matched_evidence"] == []



def test_evidence_verify_allows_unrelated_negation_in_same_evidence():
    evidence = EvidenceEngine.normalize(
        "attention uses query",
        [{
            "source": "wikipedia",
            "title": "Attention",
            "abstract": "attention uses query and values. It does not use fixed weights.",
            "identifier": "1",
        }],
    )
    verification = EvidenceEngine.verify("attention uses query", evidence)
    assert verification["verification_status"] == "MATCHED"
    assert verification["matched_evidence"] == [0]


def test_evidence_verify_preserves_qkv_tokens():
    evidence = EvidenceEngine.normalize(
        "Q K V matrices",
        [{
            "source": "wikipedia",
            "title": "Attention",
            "abstract": "Q, K and V are matrices used in attention.",
            "identifier": "1",
        }],
    )
    verification = EvidenceEngine.verify(
        "Q K V are matrices",
        evidence,
    )
    assert verification["verification_status"] == "MATCHED"


def test_evidence_verify_allows_positive_clause_after_local_negation():
    evidence = EvidenceEngine.normalize(
        "attention uses query",
        [{
            "source": "wikipedia",
            "title": "Attention",
            "abstract": "Attention does not use fixed weights, but it uses query values.",
            "identifier": "1",
        }],
    )
    verification = EvidenceEngine.verify("attention uses query", evidence)
    assert verification["verification_status"] == "MATCHED"

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

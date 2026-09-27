from core.search_strategy import SearchStrategy
from core.state import AgentStep


def test_covered_search_prefers_answer():
    step = AgentStep(
        step_id=1,
        action="SEARCH",
        arguments={"query": "context caching LLM"},
        observation={
            "results": [{"title": "relevant paper"}],
            "coverage": {"status": "COVERED", "relevant_count": 4, "uncovered_terms": []},
        },
        success=True,
    )
    guidance = SearchStrategy.guidance([step])
    assert guidance["stage"] == "evidence_sufficient"
    assert guidance["required_change"] == "prefer_answer"
    assert guidance["prefer_action"] == "ANSWER"
    assert guidance["evidence_status"] == "COVERED"
    assert guidance["relevant_count"] == 4


def test_partial_search_does_not_force_answer():
    step = AgentStep(
        step_id=1,
        action="SEARCH",
        arguments={"query": "context caching LLM"},
        observation={
            "results": [{"title": "weak result"}],
            "coverage": {"status": "PARTIAL", "relevant_count": 1, "uncovered_terms": ["caching"]},
        },
        success=True,
    )
    guidance = SearchStrategy.guidance([step])
    assert guidance["stage"] == "initial"
    assert "prefer_action" not in guidance

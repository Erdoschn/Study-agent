from core.search_strategy import SearchStrategy
from core.state import AgentStep


def _empty(step, query):
    return AgentStep(
        step_id=step,
        action="SEARCH",
        arguments={"query": query},
        success=False,
        error="SEARCH_EMPTY: 搜索请求成功，但没有返回结果。",
    )


def test_long_empty_search_requires_shorter_query():
    steps = [_empty(1, "上下文缓存是什么，这个相关有什么研究，目的是什么，结论是什么？")]
    guidance = SearchStrategy.guidance(steps)
    assert guidance["required_change"] == "shorter_query"
    assert len(guidance["suggested_query"]) < len(steps[0].arguments["query"])


def test_second_chinese_empty_search_requires_english():
    steps = [
        _empty(1, "上下文缓存是什么，这个相关有什么研究，目的是什么，结论是什么？"),
        _empty(2, "上下文缓存 研究"),
    ]
    guidance = SearchStrategy.guidance(steps)
    assert guidance["required_change"] == "english_query"


def test_third_empty_search_requires_one_or_two_english_terms():
    steps = [
        _empty(1, "context caching definition research"),
        _empty(2, "context caching research"),
        _empty(3, "context caching"),
    ]
    guidance = SearchStrategy.guidance(steps)
    assert guidance["required_change"] == "one_or_two_terms"


def test_core_query_removes_question_filler():
    query = "上下文缓存的定义、研究现状、应用目的和研究结论"
    assert SearchStrategy.core_query(query) == "上下文缓存"


def test_ineffective_chinese_rewrite_is_detected():
    history = ["上下文缓存的研究现状", "上下文缓存相关研究", "上下文缓存研究成果"]
    assert SearchStrategy.is_ineffective_rewrite("上下文缓存的研究结论", history)


def test_different_concept_is_not_detected():
    history = ["上下文缓存的研究现状"]
    assert not SearchStrategy.is_ineffective_rewrite("KV cache", history)


def test_timeout_search_requires_source_or_query_change():
    step = AgentStep(
        step_id=1,
        action="SEARCH",
        arguments={"query": "transformer attention"},
        success=False,
        error="SearchTimeoutError: SEARCH_TIMEOUT",
    )
    guidance = SearchStrategy.guidance([step], "SearchTimeoutError")
    assert guidance["required_change"] == "source_or_query"
    assert guidance["suggested_query"] == "transformer attention"



def test_http_search_failure_requires_strategy_change():
    step = AgentStep(
        step_id=1,
        action="SEARCH",
        arguments={"query": "transformer attention", "source": "wikipedia"},
        success=False,
        error="RuntimeError: HTTP 503 ServiceUnavailable",
    )
    guidance = SearchStrategy.guidance([step], "RuntimeError")
    assert guidance["required_change"] == "new_query_or_source"
    assert guidance["suggested_query"] == "transformer attention"

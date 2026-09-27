from types import SimpleNamespace

from core.task_analyzer import TaskAnalyzer
from core.knowledge_graph import KnowledgeGraph
from core.search_strategy import SearchStrategy


def test_context_caching_normalizes_to_llm_scope():
    raw = '{"task_type":"conceptual","domain":"计算机科学-操作系统-缓存机制"}'
    analysis = TaskAnalyzer._parse(raw)
    analysis.domain = TaskAnalyzer._normalize_domain(
        "解释上下文缓存是什么，以及它在 LLM 推理中的作用",
        analysis.domain,
    )
    assert analysis.domain == "LLM推理-上下文缓存"


def test_cpu_cache_stays_in_computer_architecture_scope():
    assert (
        TaskAnalyzer._normalize_domain(
            "解释 CPU cache、L1 cache 和缓存行",
            "缓存机制",
        )
        == "计算机体系结构-CPU缓存"
    )


def test_knowledge_graph_bootstraps_llm_context_cache():
    graph = KnowledgeGraph()
    assert graph.bootstrap_query_context("上下文缓存") == [
        "上下文缓存",
        "大语言模型",
        "LLM推理",
    ]
    assert graph._id("上下文缓存") in graph.nodes
    assert graph._id("大语言模型") in graph.nodes
    assert graph._id("LLM推理") in graph.nodes
    assert len(graph.edges) == 2


def test_cpu_cache_does_not_bootstrap_llm_context_cache():
    graph = KnowledgeGraph()
    assert graph.bootstrap_query_context("CPU cache L1") == []
    assert graph.nodes == {}
    assert graph.edges == {}


def test_search_recovery_requires_shorter_query_after_empty_long_search():
    step = SimpleNamespace(
        action="SEARCH",
        success=False,
        error="SEARCH_EMPTY",
        arguments={"query": "上下文缓存的定义、研究现状、应用目的和研究结论"},
    )
    guidance = SearchStrategy.guidance([step])
    assert guidance["required_change"] == "shorter_query"
    assert guidance["suggested_query"] == "上下文缓存"

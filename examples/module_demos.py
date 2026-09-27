"""Offline module smoke demos for Study Agent.

Run:
    python examples/module_demos.py

These demos intentionally avoid LLM APIs and network access. They exercise
deterministic parsing, graph semantics, learner state, search recovery,
evidence handling, goal matching, Teacher strategy, and the calculator
harness.
"""

from types import SimpleNamespace

from core.task_analyzer import TaskAnalyzer
from core.knowledge_graph import KnowledgeGraph
from core.state import StudentMind
from core.search_strategy import SearchStrategy
from core.evidence import EvidenceEngine, EvidenceStore
from core.goal import GoalMatcher
from core.teacher import Teacher
from core.tool_loop import ToolExecutor


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"{name} FAILED: {detail}")
    print(f"[PASS] {name}")


def demo_task_analyzer():
    raw = '{"task_type":"conceptual","domain":"计算机科学-操作系统-缓存机制","goal":"理解上下文缓存"}'
    analysis = TaskAnalyzer._parse(raw)
    analysis.domain = TaskAnalyzer._normalize_domain("解释上下文缓存是什么，以及它在 LLM 推理中的作用", analysis.domain)
    check("TaskAnalyzer.parse", analysis.task_type == "conceptual")
    check("TaskAnalyzer.cache-domain", analysis.domain == "LLM推理-上下文缓存", analysis.domain)

    cpu = TaskAnalyzer._normalize_domain("解释 CPU cache、L1 cache 和缓存行", "缓存机制")
    check("TaskAnalyzer.cpu-domain", cpu == "计算机体系结构-CPU缓存", cpu)
    print(f"  type={analysis.task_type} domain={analysis.domain}")


def demo_knowledge_graph():
    graph = KnowledgeGraph()
    seeded = graph.bootstrap_query_context("上下文缓存的原理和 LLM 推理中的作用")
    check("KnowledgeGraph.bootstrap", seeded == ["上下文缓存", "大语言模型", "LLM推理"], seeded)
    check("KnowledgeGraph.relation", len(graph.edges) == 2, len(graph.edges))
    context = graph.context_for("上下文缓存")
    check("KnowledgeGraph.context-match", "上下文缓存" in context["matched_concepts"], context)
    check("KnowledgeGraph.no-cpu-pollution", "计算机体系结构-CPU缓存" not in [n.name for n in graph.nodes.values()])
    cpu_graph = KnowledgeGraph()
    check("KnowledgeGraph.cpu-scope", cpu_graph.bootstrap_query_context("CPU cache L1") == [])
    print(f"  nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print(f"  neighbors={graph.neighbors('上下文缓存')}")


def demo_student_mind():
    student = StudentMind()
    student.begin_interaction()
    student.apply_update({"short_term": {"beliefs": ["正在学习 Transformer"]}})
    check("StudentMind.short-term", "正在学习 Transformer" in student.short_term.beliefs)
    check("StudentMind.no-premature-long-term", not student.long_term.beliefs)
    print(f"  short_term={student.short_term.as_dict()}")


def demo_search_strategy():
    core = SearchStrategy.core_query("上下文缓存的定义、研究现状、应用目的和研究结论")
    check("SearchStrategy.core-query", core == "上下文缓存", core)

    failed = SimpleNamespace(
        action="SEARCH", success=False, error="SEARCH_EMPTY",
        arguments={"query": "上下文缓存的定义、研究现状、应用目的和研究结论"},
    )
    guidance = SearchStrategy.guidance([failed])
    check("SearchStrategy.recovery", guidance["required_change"] == "shorter_query", guidance)
    print(f"  guidance={guidance}")


def demo_evidence():
    store = EvidenceStore()
    item = {"source": "arxiv", "identifier": "demo-1", "title": "Context caching for LLM inference",
            "abstract": "Context caching reduces repeated computation in LLM inference.",
            "published": "2026-01-01T00:00:00+00:00"}
    check("EvidenceStore.add", store.add_many([item, item]) == 1)
    normalized = EvidenceEngine.normalize("context caching", [item])
    check("EvidenceEngine.relevance", normalized[0]["harness_relevance"] == "DIRECT", normalized)
    coverage = EvidenceEngine.coverage("context caching", normalized)
    check("EvidenceEngine.coverage", coverage["status"] == "COVERED", coverage)
    verified = EvidenceEngine.verify("context caching reduces repeated computation", normalized)
    check("EvidenceEngine.verify", verified["verification_status"] == "MATCHED", verified)
    print(f"  evidence={store.prompt_view()}")


def demo_goal_matcher():
    goals = [
        "掌握 Transformer attention 的 QKV 计算",
        "学习 Java 文件 IO",
        "理解 LLM 上下文缓存",
    ]
    matched = GoalMatcher.match("解释 LLM 上下文缓存的工作原理", goals)
    check("GoalMatcher.match", "理解 LLM 上下文缓存" in matched, matched)
    print(f"  matched={matched}")


def demo_teacher_strategy():
    state = SimpleNamespace(
        student=SimpleNamespace(
            known_topics={"Transformer"},
            weak_topics={"attention"},
            learning_topics=set(),
            misconceptions=[],
        ),
        task_type="conceptual",
    )
    strategy = Teacher._derive_strategy(state)
    check("Teacher.strategy", strategy["mode"] == "脚手架教学", strategy)
    print(f"  strategy={strategy}")


def demo_tool_executor():
    executor = ToolExecutor()
    result = executor.execute("calculate", {"expression": "8 * (3 + 2)"})
    check("ToolExecutor.calculate", result == 40, result)
    print(f"  calculate=8 * (3 + 2) -> {result}")


def main():
    print("=== Study Agent offline module demos ===")
    demo_task_analyzer()
    demo_knowledge_graph()
    demo_student_mind()
    demo_search_strategy()
    demo_evidence()
    demo_goal_matcher()
    demo_teacher_strategy()
    demo_tool_executor()
    print("\nALL MODULE DEMOS PASSED")


if __name__ == "__main__":
    main()

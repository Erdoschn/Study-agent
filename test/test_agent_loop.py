from types import SimpleNamespace

from core.reasoner import ReasoningDecision
from core.state import AgentState, AgentStep
from core.tool_loop import AgentToolLoop, ToolExecutor
from tools.search import SearchResult, SearchRouter


class StubProvider:
    def __init__(self, name, results):
        self.name = name
        self.results = results
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        return list(self.results)


class StubReasoner:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, state):
        return self.decisions.pop(0)


def _result(source, title):
    return SearchResult(
        source=source,
        source_type="test",
        title=title,
        url=f"https://example.com/{title}",
    )


def test_auto_router_uses_qualitative_source_order():
    router = SearchRouter()
    wiki = StubProvider("wikipedia", [_result("wikipedia", "Definition")])
    arxiv = StubProvider("arxiv", [_result("arxiv", "Paper")])
    router.register(wiki)
    router.register(arxiv)

    from tools.search import SearchQuery

    results = router.search(
        SearchQuery(
            query="agent",
            source="auto",
            source_preferences=["wikipedia", "arxiv"],
        )
    )

    assert [item.source for item in results] == ["wikipedia", "arxiv"]
    assert len(wiki.calls) == 1
    assert len(arxiv.calls) == 1


def test_auto_router_falls_through_empty_source():
    router = SearchRouter()
    wiki = StubProvider("wikipedia", [])
    arxiv = StubProvider("arxiv", [_result("arxiv", "Paper")])
    router.register(wiki)
    router.register(arxiv)

    from tools.search import SearchQuery

    results = router.search(
        SearchQuery(
            query="agent",
            source="auto",
            source_preferences=["wikipedia", "arxiv"],
        )
    )

    assert [item.source for item in results] == ["arxiv"]


def test_empty_search_becomes_explicit_observation_failure():
    router = SearchRouter()
    router.register(StubProvider("arxiv", []))

    executor = ToolExecutor(search_router=router)
    reasoner = StubReasoner(
        [
            ReasoningDecision(
                action="SEARCH",
                reasoning_summary="search",
                tool="search",
                arguments={"query": "agent"},
            ),
            ReasoningDecision(
                action="STOP",
                reasoning_summary="stop after empty search",
            ),
        ]
    )
    state = AgentState(
        question="agent",
        search_sources=["arxiv"],
        max_steps=2,
    )

    state = AgentToolLoop(reasoner, executor).run(state)

    assert state.steps[0].success is False
    assert "SEARCH_EMPTY" in state.steps[0].error


def test_plan_cursor_moves_past_executed_action():
    plan = SimpleNamespace(
        steps=[
            SimpleNamespace(action="ANALYZE"),
            SimpleNamespace(action="SEARCH"),
            SimpleNamespace(action="VERIFY"),
            SimpleNamespace(action="ANSWER"),
        ]
    )
    state = AgentState(question="q", plan=plan, current_plan_step=0)

    assert AgentToolLoop._next_plan_step(state, "SEARCH") == 2

    state.current_plan_step = 2
    assert AgentToolLoop._next_plan_step(state, "VERIFY") == 3


def test_verify_gate_blocks_answer_before_verify():
    plan = SimpleNamespace(
        steps=[
            SimpleNamespace(action="ANALYZE"),
            SimpleNamespace(action="SEARCH"),
            SimpleNamespace(action="VERIFY"),
            SimpleNamespace(action="ANSWER"),
        ]
    )
    state = AgentState(
        question="q",
        plan=plan,
        search_sources=["arxiv"],
        current_plan_step=2,
        evidence=[{"source": "arxiv", "title": "Paper", "url": "https://example.com"}],
    )

    decision = ReasoningDecision(
        action="ANSWER",
        reasoning_summary="answer",
    )
    loop = AgentToolLoop(None, None)
    forced = loop._force_verify(state, decision)

    assert forced.action == "VERIFY"
    assert forced.tool == "verify"
    assert forced.arguments["evidence"] == state.evidence


def test_search_sorting_strategy_reaches_query():
    router = SearchRouter()
    provider = StubProvider("arxiv", [_result("arxiv", "Recent paper")])
    router.register(provider)

    executor = ToolExecutor(search_router=router)
    state = AgentState(
        question="latest research",
        search_sources=["arxiv"],
        search_sort_by="submittedDate",
    )

    results = executor._search(
        {"query": "latest research"},
        state=state,
    )

    assert results[0]["title"] == "Recent paper"
    assert provider.calls[0].sort_by == "submittedDate"

def test_task_analysis_parses_qualitative_search_strategy():
    from core.task_analyzer import TaskAnalyzer

    analysis = TaskAnalyzer._parse(
        '{"task_type":"research","domain":"AI agents",'
        '"required_tools":["search","verify"],'
        '"external_facts_needed":true,'
        '"search_sources":["wikipedia","arxiv"],'
        '"search_sort_by":"submittedDate"}'
    )

    assert analysis.search_sources == ["wikipedia", "arxiv"]
    assert analysis.search_sort_by == "submittedDate"


def test_task_analysis_rejects_unknown_search_values():
    from core.task_analyzer import TaskAnalyzer

    analysis = TaskAnalyzer._parse(
        '{"search_sources":["google","arxiv","arxiv"],'
        '"search_sort_by":"not-a-valid-value"}'
    )

    assert analysis.search_sources == ["arxiv"]
    assert analysis.search_sort_by == "relevance"


def test_reasoner_parses_qualitative_evidence_relevance():
    from core.reasoner import AgentReasoner

    decision = AgentReasoner._parse(
        '{"action":"ANSWER","reasoning_summary":"use direct evidence",'
        '"evidence_relevance":['
        '{"step":1,"index":0,"relevance":"DIRECT","recency":"NEWER","use":true,"reason":"direct support"},'
        '{"step":1,"index":1,"relevance":"irrelevant","recency":"bad","use":false,"reason":"off topic"}'
        ']}'
    )

    assert decision.evidence_relevance[0]["relevance"] == "DIRECT"
    assert decision.evidence_relevance[0]["recency"] == "NEWER"
    assert decision.evidence_relevance[0]["use"] is True
    assert decision.evidence_relevance[1]["relevance"] == "IRRELEVANT"
    assert decision.evidence_relevance[1]["recency"] == "UNKNOWN"


def test_qualitative_relevance_is_fed_into_state():
    from core.tool_loop import AgentToolLoop

    class Reasoner:
        def __init__(self):
            self.n = 0

        def decide(self, state):
            self.n += 1
            if self.n == 1:
                return ReasoningDecision(
                    action="SEARCH",
                    reasoning_summary="search",
                    tool="search",
                    arguments={"query": "attention"},
                    evidence_relevance=[],
                )
            return ReasoningDecision(
                action="ANSWER",
                reasoning_summary="answer",
                evidence_relevance=[
                    {
                        "step": 1,
                        "index": 0,
                        "relevance": "DIRECT",
                        "recency": "NEWER",
                        "use": True,
                        "reason": "核心结果",
                    }
                ],
            )

    class Executor:
        def execute(self, tool, arguments):
            return [{"title": "Attention", "abstract": "attention mechanism"}]

    state = AgentState(question="attention", max_steps=3)
    state = AgentToolLoop(Reasoner(), Executor()).run(state)

    assert state.evidence_relevance[0]["relevance"] == "DIRECT"
    assert state.evidence_relevance[0]["use"] is True


def test_search_uses_larger_default_candidate_pool():
    router = SearchRouter()
    provider = StubProvider("arxiv", [_result("arxiv", "Paper")])
    router.register(provider)

    executor = ToolExecutor(search_router=router)
    executor._search({"query": "attention"})

    assert provider.calls[0].max_results == 10


def test_failed_tool_can_be_retried_with_same_call():
    class FlakyExecutor:
        def __init__(self):
            self.calls = 0
        def execute(self, tool, arguments):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            return [{"title": "recovered"}]

    reasoner = StubReasoner([
        ReasoningDecision(action="SEARCH", reasoning_summary="retry", tool="search", arguments={"query":"q"}),
        ReasoningDecision(action="SEARCH", reasoning_summary="retry same query", tool="search", arguments={"query":"q"}),
        ReasoningDecision(action="ANSWER", reasoning_summary="answer", answer="ok"),
    ])
    state = AgentToolLoop(reasoner, FlakyExecutor()).run(AgentState(question="q"))
    assert state.finished is True
    assert state.final_answer == "ok"
    assert state.recovery_count == 1
    assert state.steps[0].success is False
    assert state.steps[0].observation["status"] == "ERROR"
    assert state.steps[1].success is True

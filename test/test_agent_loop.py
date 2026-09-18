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

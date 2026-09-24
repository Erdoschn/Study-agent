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


def test_failed_exact_tool_call_is_not_retried():
    class FailingExecutor:
        def execute(self, tool, arguments):
            raise RuntimeError("persistent failure")

    reasoner = StubReasoner([
        ReasoningDecision(action="SEARCH", reasoning_summary="search", tool="search", arguments={"query": "q"}),
        ReasoningDecision(action="SEARCH", reasoning_summary="same call", tool="search", arguments={"query": "q"}),
    ])
    state = AgentToolLoop(reasoner, FailingExecutor()).run(AgentState(question="q", max_steps=3))
    assert state.finished is True
    assert "重复工具调用" in state.error

def test_search_observation_preserves_coverage_in_reasoner_prompt():
    from core.reasoner import AgentReasoner
    from core.tool_loop import SearchObservation

    state = AgentState(question="attention")
    state.last_observation = SearchObservation(
        [{"title": "Attention mechanism", "harness_relevance": "PARTIAL"}],
        {"status": "PARTIAL", "relevant_count": 1, "uncovered_terms": ["transformer"]},
    )

    prompt = AgentReasoner._build_prompt(AgentReasoner.__new__(AgentReasoner), state, [])
    import json
    payload = json.loads(prompt)

    assert payload["last_observation"]["coverage"]["status"] == "PARTIAL"
    assert payload["last_observation"]["coverage"]["uncovered_terms"] == ["transformer"]
    assert payload["last_observation"]["results"][0]["harness_relevance"] == "PARTIAL"

def test_adaptive_search_changes_query_after_partial_coverage():
    router = SearchRouter()
    wiki = StubProvider(
        "wikipedia",
        [_result("wikipedia", "Attention mechanism")],
    )
    arxiv = StubProvider(
        "arxiv",
        [_result("arxiv", "Attention mechanism in transformer")],
    )
    router.register(wiki)
    router.register(arxiv)

    executor = ToolExecutor(search_router=router)

    class AdaptiveReasoner:
        def __init__(self):
            self.calls = 0

        def decide(self, state):
            self.calls += 1
            if self.calls == 1:
                assert state.last_observation is None
                return ReasoningDecision(
                    action="SEARCH",
                    reasoning_summary="先搜索核心概念。",
                    tool="search",
                    arguments={"query": "attention transformer", "source": "wikipedia"},
                )

            if self.calls == 2:
                assert state.last_observation["coverage"]["status"] == "PARTIAL"
                assert state.last_observation["coverage"]["uncovered_terms"] == ["transformer"]
                return ReasoningDecision(
                    action="SEARCH",
                    reasoning_summary="首轮证据只覆盖部分关键词，改用另一来源补足。",
                    tool="search",
                    arguments={"query": "attention transformer", "source": "arxiv"},
                )

            assert state.last_observation["coverage"]["status"] == "COVERED"
            return ReasoningDecision(
                action="ANSWER",
                reasoning_summary="证据覆盖充分。",
                answer="ok",
            )

    state = AgentToolLoop(
        AdaptiveReasoner(),
        executor,
    ).run(AgentState(question="attention transformer"))

    assert state.finished is True
    assert state.final_answer == "ok"
    assert [call.query for call in wiki.calls] == ["attention transformer"]
    assert [call.query for call in arxiv.calls] == ["attention transformer"]
    assert state.steps[0].observation["coverage"]["status"] == "PARTIAL"
    assert state.steps[1].observation["coverage"]["status"] == "COVERED"

def test_goal_context_matches_saved_learning_goal_before_reasoning():
    from core.goal import GoalMatcher

    matched = GoalMatcher.context_for(
        "help me derive transformer attention dimensions",
        "解决当前问题",
        ["build a reliable study agent", "understand transformer attention dimensions"],
    )

    assert matched == ["understand transformer attention dimensions"]


def test_goal_context_reaches_reasoner_prompt():
    import json
    from core.reasoner import AgentReasoner

    state = AgentState(question="attention dimensions")
    state.goal_context = ["understand transformer attention dimensions"]
    payload = json.loads(AgentReasoner._build_prompt(AgentReasoner.__new__(AgentReasoner), state, []))

    assert payload["goal_context"] == ["understand transformer attention dimensions"]


def test_goal_context_is_injected_from_student_memory_before_first_decision():
    from core.goal import GoalMatcher

    class Reasoner:
        def __init__(self):
            self.seen = None

        def decide(self, state):
            self.seen = list(state.goal_context)
            return ReasoningDecision(action="ANSWER", reasoning_summary="answer", answer="ok")

    reasoner = Reasoner()
    state = AgentState(question="transformer attention dimensions")
    state.student.mind.long_term.desires.append("understand transformer attention dimensions")

    state = AgentToolLoop(reasoner, ToolExecutor()).run(state)

    assert reasoner.seen == ["understand transformer attention dimensions"]
    assert state.final_answer == "ok"

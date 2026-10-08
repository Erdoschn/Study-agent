import json

from coder.harness import CoderHarness
from coder.reasoner import CoderReasoner
from coder.state import CoderGoal, CoderState
from coder.study_bridge import StudyAgentBridge


def test_coder_reasoner_accepts_ask_study_agent():
    decision = CoderReasoner._parse(
        '{"action":"ASK_STUDY_AGENT","arguments":{"question":"explain tensor shapes"}}'
    )
    assert decision["action"] == "ASK_STUDY_AGENT"
    assert decision["arguments"]["question"] == "explain tensor shapes"


def test_coder_harness_exposes_study_agent_tool_and_passes_context(tmp_path):
    class FakeBridge:
        def __init__(self):
            self.calls = []

        def ask(self, question, *, context=None):
            self.calls.append((question, context))
            return {
                "answer": "Use reshape before transpose.",
                "domain": "Transformer",
                "task_type": "conceptual",
                "learner_context": {
                    "learning_concepts": ["Multi-Head Attention"],
                },
                "evidence": [],
            }

    project = tmp_path / "transformer-demo"
    project.mkdir()
    bridge = FakeBridge()
    harness = CoderHarness(
        project,
        search_router=object(),
        sandbox=object(),
        backup=object(),
        study_bridge=bridge,
    )
    state = CoderState(
        request="implement attention",
        project="transformer-demo",
        goal=CoderGoal("implement attention"),
    )

    specs = harness.tool_specs()
    ask_spec = next(item for item in specs if item["name"] == "ASK_STUDY_AGENT")
    assert ask_spec["parameters"]["required"] == ["question"]

    result = harness.execute(
        "ASK_STUDY_AGENT",
        {"question": "Why do I need reshape before transpose here?"},
        state,
    )

    assert result["status"] == "STUDY_AGENT_ASSISTED"
    assert result["answer"] == "Use reshape before transpose."
    assert bridge.calls[0][1]["request"] == "implement attention"
    assert bridge.calls[0][1]["goal"] == "implement attention"
    assert bridge.calls[0][1]["recent_actions"] == []

    graph = harness.knowledge_graph.snapshot()
    assert any(
        node["name"] == "Transformer" and node["type"] == "study_concept"
        for node in graph["nodes"]
    )
    assert any(
        edge["relation"] == "consulted_study_concept"
        for edge in graph["edges"]
    )


def test_study_agent_bridge_posts_context_and_returns_answer(monkeypatch):
    import coder.study_bridge as module

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, limit):
            assert limit == 1_048_576
            return json.dumps({
                "answer": "Study answer",
                "domain": "Python",
                "task_type": "conceptual",
                "learner_context": {},
                "evidence": [],
            }).encode("utf-8")

    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    bridge = StudyAgentBridge(
        "http://127.0.0.1:8000/internal/study/ask",
        timeout=7,
        api_key="bridge-key",
    )

    result = bridge.ask(
        "Explain dependency injection.",
        context={"request": "build app", "recent_actions": ["READ_FILE"]},
    )

    assert result["answer"] == "Study answer"
    assert seen["url"].endswith("/internal/study/ask")
    assert seen["timeout"] == 7
    assert seen["headers"]["Authorization"] == "Bearer bridge-key"
    assert seen["body"]["requester"] == "coder"
    assert seen["body"]["question"] == "Explain dependency injection."
    assert seen["body"]["context"]["request"] == "build app"

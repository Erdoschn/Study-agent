from types import SimpleNamespace

from core.agent import StudyAgent
from core.reasoner import ReasoningDecision


def test_persistent_learner_state_is_synced_before_task_analysis(monkeypatch):
    seen = {}

    class Analyzer:
        def __init__(self, reasoner):
            pass

        def analyze(self, question, student_state=None):
            seen["known_topics"] = set(student_state.known_topics)
            seen["weak_topics"] = set(student_state.weak_topics)
            return SimpleNamespace(
                task_type="conceptual",
                domain="attention",
                goal="understand attention",
                issues=[],
                knowledge_gaps=[],
                required_tools=[],
                external_facts_needed=False,
                answer_strategy="explain",
            )

    class Reasoner:
        def decide(self, state):
            return ReasoningDecision(
                action="ANSWER",
                reasoning_summary="answer",
                answer="ok",
            )

    class Executor:
        def execute(self, tool, arguments):
            raise AssertionError("no tool call expected")

    monkeypatch.setattr("core.agent.TaskAnalyzer", Analyzer)

    agent = StudyAgent(Reasoner(), tool_executor=Executor())
    for _ in range(5):
        agent.record_assessment(
            ["attention"],
            True,
            difficulty="postgraduate_plus",
        )

    state = agent.run("什么是 attention")

    assert "attention" in seen["known_topics"]
    assert "attention" in state.student.known_topics
    assert state.final_answer == "ok"

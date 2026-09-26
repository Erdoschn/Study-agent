from core.tool_loop import ToolExecutor
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



def test_study_agent_preserves_assessment_pause(monkeypatch):
    from core.agent import StudyAgent
    from core.reasoner import ReasoningDecision

    class Analysis:
        task_type = "conceptual"
        domain = "attention"
        goal = "test attention"
        issues = []
        knowledge_gaps = []
        required_tools = []
        external_facts_needed = False
        answer_strategy = "test"

    class Analyzer:
        def __init__(self, reasoner):
            pass

        def analyze(self, question, student_state=None):
            return Analysis()

    class Reasoner:
        def decide(self, state):
            return ReasoningDecision(
                action="ASSESS",
                reasoning_summary="检查掌握程度",
                tool="assess",
                arguments={
                    "concepts": ["attention"],
                    "difficulty": "postgraduate_plus",
                    "question_type": "open_ended",
                    "question": "What does attention compute?",
                    "expected_answer": "attention maps queries to relevant values",
                    "rubric": ["queries", "relevant", "values"],
                },
            )

    monkeypatch.setattr("core.agent.TaskAnalyzer", Analyzer)
    agent = StudyAgent(Reasoner(), tool_executor=ToolExecutor())

    state = agent.run("检查 attention")
    assert state.final_answer is None
    assert state.error is None
    assert state.pending_assessment is not None
    assert agent.pending_assessment_state is state

    result = agent.submit_assessment_answer(
        "attention maps queries to relevant values"
    )
    assert result["correct"] is True
    assert result["assessment"]["concepts"] == ["attention"]
    assert state.pending_assessment is None
    assert agent.pending_assessment_state is None


def test_pending_assessment_survives_a_later_run(monkeypatch):
    from core.agent import StudyAgent
    from core.reasoner import ReasoningDecision

    class Analysis:
        task_type = "conceptual"
        domain = "attention"
        goal = "test"
        issues = []
        knowledge_gaps = []
        required_tools = []
        external_facts_needed = False
        answer_strategy = "test"

    class Analyzer:
        def __init__(self, reasoner):
            pass

        def analyze(self, question, student_state=None):
            return Analysis()

    class Reasoner:
        def __init__(self):
            self.calls = 0

        def decide(self, state):
            self.calls += 1
            if self.calls == 1:
                return ReasoningDecision(
                    action="ASSESS",
                    reasoning_summary="创建测试",
                    tool="assess",
                    arguments={
                        "concepts": ["attention"],
                        "difficulty": "graduate",
                        "question_type": "open_ended",
                        "question": "What does attention compute?",
                        "expected_answer": "attention maps queries to relevant values",
                    },
                )
            return ReasoningDecision(
                action="ANSWER",
                reasoning_summary="ordinary answer",
                answer="ok",
            )

    monkeypatch.setattr("core.agent.TaskAnalyzer", Analyzer)
    agent = StudyAgent(Reasoner(), tool_executor=ToolExecutor())

    first = agent.run("创建测试")
    assert first.pending_assessment is not None

    second = agent.run("另一个问题")
    assert second.final_answer == "ok"
    assert first.pending_assessment is not None
    assert agent.pending_assessment_state is first

    result = agent.submit_assessment_answer(
        "attention maps queries to relevant values"
    )
    assert result["correct"] is True
    assert first.pending_assessment is None

from core.agent import StudyAgent
from core.reasoner import ReasoningDecision
from core.state import AgentState


class FakeExecutor:
    def execute(self, tool, arguments):
        raise AssertionError(f"unexpected tool call: {tool}")


class AnswerReasoner:
    def decide(self, state):
        return ReasoningDecision(
            action="ANSWER",
            reasoning_summary="当前信息足够回答。",
            answer="reasoner draft",
            model="reasoner-free",
        )


class FakeTeacher:
    def __init__(self):
        self.calls = []

    def generate(self, state, draft_answer=None):
        self.calls.append((state, draft_answer))
        return "teacher final"


def test_modern_answer_is_passed_through_teacher():
    teacher = FakeTeacher()
    agent = StudyAgent(
        reasoner=AnswerReasoner(),
        teacher=teacher,
        tool_executor=FakeExecutor(),
    )

    state = agent.run("解释 attention")

    assert state.final_answer == "teacher final"
    assert len(teacher.calls) == 1
    assert teacher.calls[0][1] == "reasoner draft"
    assert state.steps[-1].action == "ANSWER"
    assert state.finished is True


def test_teacher_failure_keeps_reasoner_draft():
    class FailingTeacher:
        def generate(self, state, draft_answer=None):
            raise RuntimeError("teacher unavailable")

    agent = StudyAgent(
        reasoner=AnswerReasoner(),
        teacher=FailingTeacher(),
        tool_executor=FakeExecutor(),
    )

    state = agent.run("解释 attention")

    assert state.final_answer == "reasoner draft"
    assert state.finished is True
    assert state.error is None


def test_teacher_is_optional():
    agent = StudyAgent(
        reasoner=AnswerReasoner(),
        teacher=None,
        tool_executor=FakeExecutor(),
    )

    state = agent.run("解释 attention")

    assert state.final_answer == "reasoner draft"
    assert state.finished is True

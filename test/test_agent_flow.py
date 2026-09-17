from core.agent import StudyAgent
from core.state import AgentState


class FakeModel:
    name = "fake-free-model"


class FakeRegistry:
    def record_success(self, *args): pass
    def record_failure(self, *args): pass


class FakeRouter:
    registry = FakeRegistry()

    def select_candidates(self, capability, allow_paid=False, exclude=None, **kwargs):
        return [FakeModel()]


class FakeClient:
    def generate(self, system_prompt, user_prompt, json_mode=False):
        if "任务分析器" in system_prompt:
            return '''{
                "task_type": "conceptual",
                "domain": "transformer",
                "goal": "understand attention",
                "issues": [],
                "knowledge_gaps": ["Q/K/V"],
                "required_tools": [],
                "external_facts_needed": false,
                "answer_strategy": "explain the concepts step by step"
            }'''
        raise AssertionError("unexpected model call")


class FakeFactory:
    def create(self, model): return FakeClient()


class FakeReasoner:
    model_router = FakeRouter()
    model_factory = FakeFactory()
    allow_paid = False

    def decide(self, state):
        from core.reasoner import ReasoningDecision
        return ReasoningDecision(action="ANSWER", reasoning_summary="当前信息足够回答。", goal=state.goal, task_type=state.task_type, domain=state.domain, model="fake-free-model")


class FakeTeacher:
    def generate(self, state): return "teaching answer"


def test_agent_runs_analysis_plan_and_reason_loop():
    agent = StudyAgent(reasoner=FakeReasoner(), teacher=FakeTeacher(), tool_executor=object())
    state = agent.run("解释 Transformer attention")
    assert isinstance(state, AgentState)
    assert state.task_analysis is not None
    assert state.plan is not None
    assert state.plan.actions == ["ANALYZE", "ANSWER"]
    assert state.task_type == "conceptual"
    assert state.domain == "transformer"
    assert state.final_answer == "teaching answer"
    assert state.finished is True
    assert state.step_count == 1

from core.model_router import ModelRouter


def test_model_router_has_no_execution_strategy():
    router = ModelRouter(None)
    assert not hasattr(router, "select_execution_strategy")


def test_model_router_only_selects_model_and_effort():
    # The execution path is decided by AgentReasoner inside AgentToolLoop,
    # while ModelRouter only supplies a model/effort for the current role.
    assert hasattr(ModelRouter, "select")
    assert hasattr(ModelRouter, "select_effort")


def test_study_agent_always_enters_agent_loop_for_simple_question(monkeypatch):
    from core.agent import StudyAgent
    from core.task_analyzer import TaskAnalysis
    from core.reasoner import ReasoningDecision

    class Model:
        name = "fake"

    class Registry:
        def record_success(self, *args): pass
        def record_failure(self, *args, **kwargs): pass
        def record_task_outcome(self, *args, **kwargs): pass
        def is_provider_level_failure(self, exc): return False

    class Router:
        registry = Registry()

        def select_candidates(self, capability, **kwargs):
            return [Model()]

    class Factory:
        def create(self, model): return object()

    class Reasoner:
        model_router = Router()
        model_factory = Factory()
        allow_paid = False

        def decide(self, state):
            return ReasoningDecision(
                action="ANSWER",
                reasoning_summary="直接回答是当前最优动作",
                answer="ok",
                model="fake",
            )

    class Analyzer:
        def __init__(self, reasoner): pass

        def analyze(self, question, student_state=None):
            return TaskAnalysis(
                task_type="general",
                domain="general",
                difficulty=1,
            )

    class Executor:
        def execute(self, tool, arguments, state=None):
            raise AssertionError("Reasoner should have answered without a tool")

        def tool_specs(self):
            return []

    monkeypatch.setattr("core.agent.TaskAnalyzer", Analyzer)
    agent = StudyAgent(Reasoner(), teacher=None, tool_executor=Executor())
    state = agent.run("你好")

    assert state.final_answer == "ok"
    assert state.step_count == 1
    assert state.steps[0].action == "ANSWER"
    assert state.metrics["agent_loop"] is True


def test_explicit_assessment_permission_is_carried_into_agent_state():
    from core.state import AgentState

    assert AgentState(
        question="请测试我",
        assessment_requested=True,
    ).assessment_requested is True

    assert AgentState(
        question="什么是 attention？",
        assessment_requested=False,
    ).assessment_requested is False

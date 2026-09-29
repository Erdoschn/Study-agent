import json

from core.assessment_generator import AssessmentGenerator
from core.agent import StudyAgent


class Model:
    name = "assessment-model"
    reasoning_effort_param = None


class Registry:
    def __init__(self):
        self.success = []
        self.failure = []

    def record_success(self, *args):
        self.success.append(args)

    def record_failure(self, *args, **kwargs):
        self.failure.append(args)


class Router:
    def __init__(self, client):
        self.registry = Registry()
        self.client = client

    def select_candidates(self, capability, **kwargs):
        assert capability == "assessment"
        return [Model()]

    def call_model(self, client, system_prompt, user_prompt, json_mode=False,
                   reasoning_effort=None, reasoning_effort_param=None):
        return client.generate(
            system_prompt,
            user_prompt,
            json_mode=json_mode,
        )


class Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate(self, system_prompt, user_prompt, json_mode=False):
        self.calls += 1
        assert json_mode is True
        return json.dumps(self.payload, ensure_ascii=False)


class Factory:
    def __init__(self, client):
        self.client = client

    def create(self, model):
        return self.client


def test_assessment_generator_returns_structured_formal_test():
    payload = {
        "primary_concept": "矩母函数",
        "supporting_concepts": ["原点矩"],
        "question": "定义矩母函数，并说明它如何生成各阶原点矩。",
        "expected_answer": "M_X(t)=E[e^{tX}]；若在0邻域存在，则M_X^{(n)}(0)=E[X^n]。",
        "rubric": [
            "给出 M_X(t)=E[e^{tX}]",
            "说明在适当存在条件下 n 阶导数得到 n 阶原点矩",
        ],
        "question_type": "open_ended",
    }
    router = Router(Client(payload))
    generator = AssessmentGenerator(router, Factory(router.client))

    result = generator.generate("矩母函数", "postgraduate_plus")

    assert result["primary_concept"] == "矩母函数"
    assert result["difficulty_level"] == "postgraduate_plus"
    assert result["difficulty"] == 1.0
    assert len(result["rubric"]) == 2
    assert result["concepts"] == ["矩母函数", "原点矩"]


def test_assessment_generator_rejects_missing_rubric():
    payload = {
        "primary_concept": "attention",
        "question": "Explain attention.",
        "expected_answer": "Attention computes weighted combinations.",
        "rubric": ["mentions attention"],
    }
    router = Router(Client(payload))
    generator = AssessmentGenerator(router, Factory(router.client))

    try:
        generator.generate("attention", "graduate")
        assert False
    except ValueError as exc:
        assert "两个" in str(exc)


def test_study_agent_formal_assessment_updates_learner_after_submission():
    payload = {
        "primary_concept": "attention",
        "supporting_concepts": [],
        "question": "Explain attention as a mechanism for weighting relevant values.",
        "expected_answer": "attention maps a query to relevant values",
        "rubric": [
            "attention",
            "query",
            "relevant",
            "values",
        ],
        "question_type": "open_ended",
    }
    router = Router(Client(payload))
    factory = Factory(router.client)

    class Reasoner:
        model_router = router
        model_factory = factory
        allow_paid = False

    agent = StudyAgent(
        reasoner=Reasoner(),
        teacher=None,
        tool_executor=None,
        knowledge_graph_path=None,
    )
    agent.assessment_generator = AssessmentGenerator(router, factory)

    ready = agent.start_assessment("attention", "postgraduate_plus")
    assert ready["status"] == "ASSESSMENT_READY"
    assert "expected_answer" not in ready

    result = agent.submit_assessment_answer(
        "attention maps a query to relevant values"
    )
    assert result["correct"] is True
    assert result["assessment"]["difficulty_level"] == "postgraduate_plus"
    assert result["learner_state"]["attention"]["learning_stage"] == "new"
    assert agent.learner_state("attention")["learner"]["exposure_count"] == 1
    assert agent.pending_assessment_state is None


def test_formal_assessment_does_not_use_code_or_file_operations():
    payload = {
        "primary_concept": "python safety",
        "supporting_concepts": [],
        "question": "Explain why arbitrary code execution is unsafe for a study agent.",
        "expected_answer": "The agent must not expose arbitrary code execution.",
        "rubric": ["arbitrary code execution", "unsafe", "must not expose"],
        "question_type": "open_ended",
    }
    router = Router(Client(payload))
    generator = AssessmentGenerator(router, Factory(router.client))

    result = generator.generate("python safety", "basic")

    assert "code" in result["question"].lower()
    assert "execute" not in result["question"].lower()

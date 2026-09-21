from core.state import AgentState, StudentState
from core.teacher import Teacher


def test_strategy_prioritizes_misconceptions():
    state = AgentState(question="什么是 attention")
    state.student = StudentState(misconceptions=["把 attention 当成固定权重"])

    strategy = Teacher._derive_strategy(state)

    assert strategy["mode"] == "纠错优先"
    assert "纠正" in strategy["focus"]


def test_strategy_uses_known_topics_as_starting_point():
    state = AgentState(question="什么是 attention")
    state.student.known_topics.add("transformer")

    strategy = Teacher._derive_strategy(state)

    assert strategy["mode"] == "建立在已有知识上"


def test_payload_contains_student_state_evidence_and_draft():
    state = AgentState(question="解释 attention", goal="理解 attention")
    state.task_type = "conceptual"
    state.domain = "transformer"
    state.student.known_topics.add("transformer")
    state.evidence = [{"source": "wikipedia", "title": "Attention", "url": "https://example.com"}]
    state.claims = [{"claim": "attention uses Q and K", "verification_status": "MATCHED"}]

    payload = Teacher._build_payload(state, draft_answer="draft")

    assert payload["draft_answer"] == "draft"
    assert payload["student"]["known_topics"] == ["transformer"]
    assert payload["evidence"][0]["title"] == "Attention"
    assert payload["claims"][0]["verification_status"] == "MATCHED"
    assert payload["teaching_strategy"]["mode"] == "建立在已有知识上"


def test_teacher_generates_with_teaching_capability():
    class Model:
        name = "teacher-free"

    class Registry:
        def __init__(self):
            self.success = []
            self.failure = []

        def record_success(self, *args):
            self.success.append(args)

        def record_failure(self, *args):
            self.failure.append(args)

    class Router:
        def __init__(self):
            self.registry = Registry()

        def select_candidates(self, capability, **kwargs):
            assert capability == "teaching"
            return [Model()]

    class Client:
        def generate(self, system_prompt, user_prompt, json_mode=False):
            assert "teaching_strategy" in user_prompt
            assert json_mode is False
            return "这是教学回答。"

    class Factory:
        def create(self, model):
            return Client()

    router = Router()
    teacher = Teacher(router, Factory())
    state = AgentState(question="解释 attention")

    result = teacher.generate(state, draft_answer="原始草稿")

    assert result == "这是教学回答。"
    assert router.registry.success == [("teacher-free", "teaching")]


def test_teacher_falls_back_across_failed_models():
    class Model:
        def __init__(self, name):
            self.name = name

    class Registry:
        def __init__(self):
            self.events = []

        def record_success(self, *args):
            self.events.append(("success", *args))

        def record_failure(self, *args):
            self.events.append(("failure", *args))

    class Router:
        def __init__(self):
            self.registry = Registry()

        def select_candidates(self, capability, **kwargs):
            assert capability == "teaching"
            return [Model("broken"), Model("working")]

    class Client:
        def __init__(self, model):
            self.model = model

        def generate(self, system_prompt, user_prompt, json_mode=False):
            if self.model.name == "broken":
                raise RuntimeError("temporary failure")
            return "fallback teaching answer"

    class Factory:
        def create(self, model):
            return Client(model)

    router = Router()
    teacher = Teacher(router, Factory())
    state = AgentState(question="解释 attention")

    assert teacher.generate(state) == "fallback teaching answer"
    assert router.registry.events == [
        ("failure", "broken", "teaching"),
        ("success", "working", "teaching"),
    ]


def test_legacy_teacher_adapter_can_be_detected():
    class LegacyTeacher:
        def generate(self, state):
            return "legacy"

    class ModernTeacher:
        def generate(self, state, draft_answer=None):
            return "modern"

    assert Teacher.supports_draft_answer(LegacyTeacher()) is False
    assert Teacher.supports_draft_answer(ModernTeacher()) is True

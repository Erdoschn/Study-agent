from core.task_analyzer import TaskAnalyzer


def test_parse_structured_analysis():
    raw = '''{
        "task_type": "math",
        "domain": "linear algebra",
        "goal": "understand matrix multiplication",
        "issues": ["possible dimension mismatch"],
        "knowledge_gaps": ["matrix dimensions"],
        "required_tools": ["calculate", "search", "invalid"],
        "external_facts_needed": false,
        "answer_strategy": "derive dimensions and verify with a small example"
    }'''
    result = TaskAnalyzer._parse(raw)
    assert result.task_type == "math"
    assert result.domain == "linear algebra"
    assert result.goal == "understand matrix multiplication"
    assert result.issues == ["possible dimension mismatch"]
    assert result.knowledge_gaps == ["matrix dimensions"]
    assert result.required_tools == ["calculate", "search"]
    assert result.external_facts_needed is False


def test_parse_rejects_invalid_json():
    try:
        TaskAnalyzer._parse("not json")
    except RuntimeError as exc:
        assert "JSON 解析失败" in str(exc)
    else:
        raise AssertionError("invalid JSON should fail")


def test_parse_accepts_think_wrapper_and_markdown_json():
    raw = '''<think>internal reasoning</think>\nHere is the analysis:\n```json\n{"task_type":"conceptual","domain":"transformer","goal":"understand attention"}\n```'''
    result = TaskAnalyzer._parse(raw)
    assert result.task_type == "conceptual"
    assert result.domain == "transformer"
    assert result.goal == "understand attention"


def test_parse_normalizes_invalid_task_type_and_non_object_json():
    result = TaskAnalyzer._parse('{"task_type":"unknown_type","domain":""}')
    assert result.task_type == "general"
    assert result.domain == "general"

    try:
        TaskAnalyzer._parse('["math"]')
    except RuntimeError as exc:
        assert "顶层结果必须是对象" in str(exc)
    else:
        raise AssertionError("non-object JSON should fail")


def test_parse_normalizes_string_boolean_and_bounds_lists():
    result = TaskAnalyzer._parse(
        '{"task_type":"conceptual","external_facts_needed":"false",'
        '"issues":["a","a","","b"],'
        '"knowledge_gaps":["gap","gap"]}'
    )
    assert result.external_facts_needed is False
    assert result.issues == ["a", "b"]
    assert result.knowledge_gaps == ["gap"]


def test_parse_has_no_execution_mode_and_defaults_assessment_off():
    result = TaskAnalyzer._parse(
        '{"task_type":"conceptual","assessment_requested":false}'
    )
    assert not hasattr(result, "execution_mode")
    assert result.assessment_requested is False


def test_parse_explicit_assessment_fields():
    result = TaskAnalyzer._parse(
        '{"task_type":"conceptual","assessment_requested":true,'
        '"assessment_concept":"attention","assessment_difficulty":"postgraduate_plus"}'
    )
    assert result.assessment_requested is True
    assert result.assessment_concept == "attention"
    assert result.assessment_difficulty == "postgraduate_plus"



def test_analyze_marks_explicit_assessment_request_without_proactive_testing():
    import json

    class Model:
        name = "fake"
        reasoning_effort_param = None

    class Registry:
        def record_success(self, *args): pass
        def record_failure(self, *args, **kwargs): pass

    class Router:
        registry = Registry()

        def select_candidates(self, capability, **kwargs):
            assert capability == "reasoning"
            return [Model()]

    class Client:
        def generate(self, system_prompt, user_prompt, json_mode=False, **kwargs):
            assert json_mode is True
            return json.dumps({
                "task_type": "conceptual",
                "domain": "transformer",
                "assessment_requested": False,
            }, ensure_ascii=False)

    class Factory:
        def create(self, model): return Client()

    class Reasoner:
        model_router = Router()
        model_factory = Factory()
        allow_paid = False

    analyzer = TaskAnalyzer(Reasoner())
    result = analyzer.analyze("给我一道 self-attention 的题测试一下我")
    assert result.assessment_requested is True
    assert "self-attention" in result.assessment_concept

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

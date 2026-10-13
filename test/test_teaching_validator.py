from core.teaching_validator import TeachingValidator
from core.state import AgentState


def test_normalize_marks_major_error_for_revision():
    result = TeachingValidator._normalize_result({
        "status": "PASS",
        "claims": [{
            "claim": "三阶矩就是偏度",
            "verdict": "ERROR",
            "severity": "major",
            "correction": "三阶中心矩并不等于偏度。",
        }],
        "summary": "发现概念错误。",
    })

    assert result["status"] == "REVISE"
    assert len(TeachingValidator.revision_findings(result)) == 1


def test_minor_confirmed_error_triggers_revision():
    result = TeachingValidator._normalize_result({
        "status": "REVISE",
        "claims": [{
            "claim": "计算结果的单位转换错误",
            "verdict": "ERROR",
            "severity": "minor",
            "correction": "按题目要求使用正确单位。",
        }],
        "summary": "仅风格/轻微问题。",
    })

    assert result["status"] == "REVISE"
    assert TeachingValidator.revision_findings(result) == [{
        "claim": "计算结果的单位转换错误",
        "verdict": "ERROR",
        "severity": "minor",
        "correction": "按题目要求使用正确单位。",
    }]


def test_invalid_validator_output_becomes_uncertain():
    result = TeachingValidator._normalize_result(
        TeachingValidator._extract_json("not json")
    )

    assert result["status"] == "UNCERTAIN"
    assert result["claims"] == []


def test_validator_build_prompt_contains_draft_and_context():
    state = AgentState(question="什么是矩母函数")
    state.task_type = "conceptual"
    state.domain = "probability"
    state.evidence = [{
        "source": "textbook",
        "title": "Probability",
        "abstract": "Moment generating functions encode moments.",
        "harness_relevance": "DIRECT",
    }]

    prompt = TeachingValidator._build_prompt(state, "矩母函数是……")

    assert "矩母函数是……" in prompt
    assert "Moment generating functions" in prompt
    assert "conceptual" in prompt

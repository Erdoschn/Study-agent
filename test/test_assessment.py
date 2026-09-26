from core.assessment import AssessmentEvaluator


def _assessment():
    return {"concepts": ["attention"], "difficulty": "postgraduate_plus", "question": "What is attention?", "expected_answer": "attention maps a query to relevant values", "rubric": ["query", "relevant", "values"]}


def test_exact_answer_is_correct():
    result = AssessmentEvaluator().evaluate(_assessment(), "attention maps a query to relevant values")
    assert result["correct"] is True
    assert result["score"] == 1.0


def test_rubric_partial_answer_is_not_correct():
    result = AssessmentEvaluator().evaluate(_assessment(), "query relevant")
    assert result["correct"] is False
    assert 0 < result["score"] < 0.8


def test_confidence_is_clamped():
    result = AssessmentEvaluator().evaluate(_assessment(), "attention maps a query to relevant values", confidence=2)
    assert result["confidence"] == 1.0


def test_missing_expected_answer_is_rejected():
    assessment = _assessment()
    assessment["expected_answer"] = ""
    try:
        AssessmentEvaluator().evaluate(assessment, "anything")
        assert False
    except ValueError:
        pass



def test_nonfinite_confidence_falls_back_to_score():
    result = AssessmentEvaluator().evaluate(
        _assessment(),
        "attention maps a query to relevant values",
        confidence=float("nan"),
    )
    assert result["confidence"] == result["score"]

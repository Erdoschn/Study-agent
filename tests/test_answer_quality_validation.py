from types import SimpleNamespace

from core.teacher import Teacher
from core.teaching_validator import TeachingValidator


class _Registry:
    def record_task_outcome(self, *args, **kwargs):
        return None


class _Router:
    def __init__(self):
        self.registry = _Registry()

    def select_choice_candidates(self, *args, **kwargs):
        model = SimpleNamespace(name="fake-teacher", reasoning_effort_param=None)
        return [SimpleNamespace(model=model, effort="high")]


def _state():
    analysis = SimpleNamespace(
        difficulty=4,
        task_type="math",
        domain="physics",
        goal="Solve the problem accurately",
    )
    return SimpleNamespace(
        question="Calculate the electric field in N/C.",
        task_type="math",
        domain="physics",
        task_analysis=analysis,
        plan=None,
        metrics={},
    )


def _error(claim="1.23 N/C", correction="1.32 N/C", severity="minor"):
    return {
        "claim": claim,
        "verdict": "ERROR",
        "severity": severity,
        "correction": correction,
    }


def test_validator_minor_errors_trigger_revision():
    result = TeachingValidator._normalize_result({
        "status": "PASS",
        "claims": [_error()],
        "summary": "One confirmed minor arithmetic error.",
    })

    assert result["status"] == "REVISE"
    assert TeachingValidator.revision_findings(result) == [_error()]


def test_teacher_independently_rechecks_answer_after_revision():
    teacher = Teacher(_Router(), model_factory=object(), validate_teaching=True)
    teacher._build_prompt = lambda state, draft_answer=None: "generation prompt"
    responses = iter(["initial answer", "corrected answer"])
    teacher._call_model = lambda model, state, prompt, effort=None: next(responses)

    validations = iter([
        {
            "status": "REVISE",
            "claims": [_error()],
            "summary": "Minor arithmetic error.",
            "model": "fake-checker",
            "effort": "high",
        },
        {
            "status": "PASS",
            "claims": [],
            "summary": "No issues found after revision.",
            "model": "fake-checker",
            "effort": "high",
        },
    ])
    teacher.teaching_validator.validate = lambda state, answer: next(validations)

    state = _state()
    result = teacher.generate(state, draft_answer="reasoner draft")

    assert result == "corrected answer"
    assert state.metrics["teaching_revision"] is True
    assert state.metrics["teaching_revision_verified"] is True
    assert state.metrics["teaching_answer_verified"] is True
    assert state.metrics["teaching_validation_status"] == "PASS"
    assert state.metrics["teaching_revision_errors_remaining"] == 0


def test_teacher_warns_when_revised_answer_still_has_confirmed_errors():
    teacher = Teacher(_Router(), model_factory=object(), validate_teaching=True)
    teacher._build_prompt = lambda state, draft_answer=None: "generation prompt"
    responses = iter(["initial answer", "revised but still wrong answer"])
    teacher._call_model = lambda model, state, prompt, effort=None: next(responses)

    validations = iter([
        {
            "status": "REVISE",
            "claims": [_error()],
            "summary": "Initial answer has an error.",
            "model": "fake-checker",
            "effort": "high",
        },
        {
            "status": "REVISE",
            "claims": [_error("revised value", "recalculate from the given values")],
            "summary": "An error remains.",
            "model": "fake-checker",
            "effort": "high",
        },
    ])
    teacher.teaching_validator.validate = lambda state, answer: next(validations)

    state = _state()
    result = teacher.generate(state, draft_answer="reasoner draft")

    assert "revised but still wrong answer" in result
    assert "答案校验提示" in result
    assert "待核对：revised value" in result
    assert state.metrics["teaching_revision_verified"] is False
    assert state.metrics["teaching_answer_verified"] is False
    assert state.metrics["teaching_revision_errors_remaining"] == 1


def test_teacher_discloses_uncertain_validation_instead_of_claiming_verified():
    teacher = Teacher(_Router(), model_factory=object(), validate_teaching=True)
    teacher._build_prompt = lambda state, draft_answer=None: "generation prompt"
    teacher._call_model = lambda model, state, prompt, effort=None: "candidate answer"
    teacher.teaching_validator.validate = lambda state, answer: {
        "status": "UNCERTAIN",
        "claims": [],
        "summary": "Validator did not have enough evidence.",
    }

    state = _state()
    result = teacher.generate(state)

    assert result.startswith("candidate answer")
    assert "答案校验提示" in result
    assert state.metrics["teaching_answer_verified"] is False

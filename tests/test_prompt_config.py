import json

from core import assessment_generator, reasoner, task_analyzer, teacher, teaching_validator
from coder.reasoner import CoderReasoner
from core.prompt_config import get_prompt, load_prompts


def test_prompt_config_loads_role_specific_prompts():
    data = load_prompts()
    assert data["version"] == 1
    assert get_prompt("study_agent.task_analyzer")
    assert get_prompt("study_agent.reasoner")
    assert get_prompt("study_agent.teacher")
    assert get_prompt("study_agent.teaching_validator")
    assert get_prompt("study_agent.assessment")
    assert get_prompt("study_agent.teacher_revision")
    assert get_prompt("coder.reasoner")
    assert get_prompt("web_model.json_contract")
    assert get_prompt("web_model.json_recovery")
    assert get_prompt("web_model.json_contract")
    assert get_prompt("web_model.json_recovery")


def test_prompt_config_matches_runtime_role_prompts():
    assert task_analyzer.TaskAnalyzer.SYSTEM_PROMPT == get_prompt("study_agent.task_analyzer")
    assert reasoner.AgentReasoner.SYSTEM_PROMPT == get_prompt("study_agent.reasoner")
    assert teacher.Teacher.SYSTEM_PROMPT == get_prompt("study_agent.teacher")
    assert (
        teaching_validator.TeachingValidator.SYSTEM_PROMPT
        == get_prompt("study_agent.teaching_validator")
    )
    assert (
        assessment_generator.AssessmentGenerator.SYSTEM_PROMPT
        == get_prompt("study_agent.assessment")
    )
    assert CoderReasoner.SYSTEM_PROMPT == get_prompt("coder.reasoner")


def test_prompt_config_supports_external_override(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {"version": 1, "study_agent": {"reasoner": "CUSTOM"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert get_prompt("study_agent.reasoner", path) == "CUSTOM"

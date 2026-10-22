import json

from core import assessment_generator, reasoner, task_analyzer, teacher, teaching_validator
from coder.reasoner import CoderReasoner
from core.prompt_config import get_prompt, load_prompts
from core.model_router import ModelRouter


def test_prompt_config_loads_role_specific_prompts():
    data = load_prompts()
    assert data["version"] == 1
    assert get_prompt("study_agent.task_analyzer")
    assert get_prompt("study_agent.reasoner")
    assert get_prompt("study_agent.teacher")
    assert get_prompt("study_agent.teaching_validator")
    assert get_prompt("study_agent.assessment")
    assert get_prompt("study_agent.question_generation")
    assert get_prompt("study_agent.question_regeneration")
    assert get_prompt("study_agent.teacher_revision")
    assert get_prompt("coder.reasoner")
    assert get_prompt("web_model.json_contract")
    assert get_prompt("web_model.json_recovery")
    assert get_prompt("web_model.templates.role_json_contract")
    assert get_prompt("model_reasoning.deep_thinking_system")
    assert get_prompt("model_reasoning.standard_system")


def test_coder_prompt_documents_new_chat_policy():
    prompt = get_prompt("coder.reasoner")
    assert "NEW_CHAT" in prompt
    assert "默认复用当前 DeepSeek Web 对话" in prompt
    assert "不要为了普通的 READ_FILE、PATCH_FILE 或测试循环而主动新建聊天" in prompt


def test_coder_prompt_is_independent_from_study_agent_role():
    prompt = get_prompt("coder.reasoner")
    assert "独立的多语言 Coding Agent 决策器" in prompt
    assert "HTML/CSS/JavaScript、C++ 和 Java" in prompt
    assert "RUN_PROJECT" in prompt
    assert "不负责控制宿主应用、上层工作流或其他 Agent" in prompt
    assert "StudyAgent" not in prompt
    assert "StudyAgent 的 Python Coding Agent" not in prompt


def test_shared_web_model_contract_is_role_neutral():
    prompt = get_prompt("web_model.json_contract")
    assert "You are a browser-backed agent runtime." in prompt
    assert "You are controlling StudyAgent through a browser UI." not in prompt


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



def test_task_analyzer_prompt_requires_explicit_deep_thinking_decision():
    prompt = get_prompt("study_agent.task_analyzer")
    assert '"deep_thinking":false' in prompt
    assert "不要只凭问题长度" in prompt
    assert "错误代价和额外延迟" in prompt


def test_task_analyzer_parses_deep_thinking_as_a_boolean():
    assert task_analyzer.TaskAnalyzer._parse('{"deep_thinking":true}').deep_thinking is True
    assert task_analyzer.TaskAnalyzer._parse('{"deep_thinking":false}').deep_thinking is False
    assert task_analyzer.TaskAnalyzer._parse('{"deep_thinking":"off"}').deep_thinking is False
    # Backward compatibility for legacy model responses without the field.
    assert task_analyzer.TaskAnalyzer._parse("{}").deep_thinking is False


def test_model_router_uses_lowest_effort_when_analyzer_rejects_deep_thinking():
    class Model:
        reasoning_efforts = ["none", "minimal", "low", "high"]
        benchmark_scores = {}

    model = Model()
    assert ModelRouter._select_effort(model, 5, deep_thinking=False) == "none"
    assert ModelRouter._select_effort(model, 5, deep_thinking=True) == "high"


def test_model_router_places_reasoning_guidance_in_api_system_prompt():
    class ApiClient:
        def __init__(self):
            self.system_prompt = ""

        def generate(self, system_prompt, user_prompt, json_mode=False, reasoning_effort=None):
            self.system_prompt = system_prompt
            return "ok"

    client = ApiClient()
    assert ModelRouter.call_model(
        client,
        "ROLE INSTRUCTIONS",
        "task",
        deep_thinking=False,
    ) == "ok"
    assert "ROLE INSTRUCTIONS" in client.system_prompt
    assert get_prompt("model_reasoning.standard_system") in client.system_prompt
    assert get_prompt("model_reasoning.deep_thinking_system") not in client.system_prompt


def test_model_router_passes_thinking_decision_to_browser_model_without_prompt_duplication():
    class BrowserClient:
        def __init__(self):
            self.deep_thinking = None
            self.system_prompt = ""

        def generate(self, system_prompt, user_prompt, json_mode=False, deep_thinking=None):
            self.system_prompt = system_prompt
            self.deep_thinking = deep_thinking
            return "ok"

    client = BrowserClient()
    assert ModelRouter.call_model(
        client,
        "ROLE INSTRUCTIONS",
        "task",
        deep_thinking=True,
    ) == "ok"
    assert client.deep_thinking is True
    assert client.system_prompt == "ROLE INSTRUCTIONS"

def test_standard_task_prompts_do_not_trigger_assessment_generation():
    task_prompt = get_prompt("study_agent.task_analyzer")
    reasoner_prompt = get_prompt("study_agent.reasoner")
    assert "assessment_requested" not in task_prompt
    assert "assessment_concept" not in task_prompt
    assert "ASSESS" not in reasoner_prompt
    assert "assessment_requested" not in reasoner_prompt
    assert get_prompt("study_agent.question_generation") != get_prompt("study_agent.question_regeneration")

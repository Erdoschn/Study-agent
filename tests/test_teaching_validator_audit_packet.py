import json
from types import SimpleNamespace

from core.teacher import Teacher
from core.teaching_validator import TeachingValidator
from core.web_model import BrowserModel


def _state(question: str, *, goal: str = ""):
    return SimpleNamespace(
        question=question,
        task_analysis=SimpleNamespace(
            goal=goal,
            task_type="math",
            domain="physics",
            difficulty=3,
        ),
        goal=goal,
        task_type="math",
        domain="physics",
        evidence=[
            {
                "source": "trusted-reference",
                "title": "Electrostatics reference",
                "identifier": "ref-1",
                "abstract": "A short grounding fact.",
                "harness_relevance": "RELEVANT",
                "harness_recency": "CURRENT",
            }
        ],
        claims=[{"claim": "E=V/d", "verification_status": "MATCHED"}],
    )


def test_audit_packet_extracts_latest_question_and_requirements_not_full_chat():
    question = (
        "以下是当前 StudyAgent 对话上下文。只依据这段上下文回答最后一个用户请求。\n"
        "【用户】上一次无关请求：解释傅里叶变换。\n"
        "【助手】上一次回答的长篇内容不应进入审计。\n"
        "【用户】请解答附件第 4 题，计算电场，并以 N/C 表示。保留三位有效数字，只给最终答案。\n"
        "请回答最后一个【用户】请求。\n\n"
        "【原始文件附件】\n"
        '文件名（JSON）：["hw4 question paper.pdf"]\n'
        "这些原始附件会直接上传到本次 DeepSeek Web 会话。"
    )
    state = _state(
        question,
        goal="解答作业文件第 4 题中的电场问题，并按照要求输出单位和有效数字。",
    )

    payload = json.loads(TeachingValidator._build_prompt(state, "Answer: 1.23 N/C"))
    spec = payload["audit_specification"]

    assert spec["problem_to_solve"] == state.task_analysis.goal
    fragments = spec["extracted_request_questions_and_requirements"]
    assert any("附件第 4 题" in fragment for fragment in fragments)
    assert any("保留三位有效数字" in fragment for fragment in fragments)
    assert any("只给最终答案" in fragment for fragment in fragments)
    assert "上一次无关请求" not in json.dumps(payload, ensure_ascii=False)
    assert "上一次回答的长篇内容" not in json.dumps(payload, ensure_ascii=False)
    assert spec["source_files_attached_to_this_audit_session"] == ["hw4 question paper.pdf"]
    assert payload["candidate_answer_to_audit"] == "Answer: 1.23 N/C"
    assert "hw4 question paper.pdf" in json.dumps(payload, ensure_ascii=False)


def test_audit_packet_reconstructs_problem_from_source_when_user_request_is_vague():
    state = _state(
        "请根据上传文件作答。\n\n"
        "【原始文件附件】\n"
        '文件名（JSON）：["assignment.pdf"]\n'
        "原始文件正文不会注入提示词。",
        goal="",
    )

    payload = json.loads(TeachingValidator._build_prompt(state, "Draft answer"))
    spec = payload["audit_specification"]

    assert spec["problem_to_solve"] == "请根据上传文件作答。"
    assert spec["source_files_attached_to_this_audit_session"] == ["assignment.pdf"]
    assert any("derive the missing audit questions from the source file" in step.lower()
               for step in spec["audit_order"])
    assert any("Before comparing the candidate answer" in item
               for item in spec["review_questions"])
    assert "Read those original files yourself before judging the candidate answer" in spec["source_handling_rule"]
    assert payload["candidate_answer_to_audit"] == "Draft answer"


def test_audit_without_source_file_does_not_claim_that_an_attachment_was_sent():
    state = _state("Why does the electric field have units of N/C?", goal="Explain electric field units.")

    payload = json.loads(TeachingValidator._build_prompt(state, "Because E=F/q."))
    spec = payload["audit_specification"]

    assert spec["source_files_attached_to_this_audit_session"] == []
    assert "No original file is attached" in spec["source_handling_rule"]
    assert not any("attached original source" in item.lower() for item in spec["review_questions"])
    assert payload["candidate_answer_to_audit"] == "Because E=F/q."


def test_real_teacher_and_validator_system_prompts_map_to_isolated_browser_roles():
    assert BrowserModel._infer_session_role(Teacher.SYSTEM_PROMPT) == "answer_generation"
    assert BrowserModel._infer_session_role(TeachingValidator.SYSTEM_PROMPT) == "answer_audit"

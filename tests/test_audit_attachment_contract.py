import json
from types import SimpleNamespace

from core.teaching_validator import TeachingValidator


def _state():
    request = (
        "请根据上传的物理题目计算第 2 问的电场强度，并给出单位。"
        "\n\n【原始文件附件】\n"
        "文件名（JSON）：[\"hw4 question paper.pdf\"]\n"
        "这些原始附件会直接上传到本次 DeepSeek Web 会话。"
        "Study Agent 没有预先解析或提取文件正文。"
    )
    return SimpleNamespace(
        question=request,
        goal="根据物理题第 2 问计算电场强度并给出单位",
        task_type="math",
        domain="电磁学",
        task_analysis=SimpleNamespace(
            goal="根据物理题第 2 问计算电场强度并给出单位",
            task_type="math",
            domain="电磁学",
        ),
        evidence=[{
            "source": "textbook",
            "title": "Electric fields",
            "identifier": "chapter-3",
            "abstract": "The electric field is measured in newtons per coulomb.",
            "harness_relevance": "DIRECT",
            "harness_recency": "UNKNOWN",
        }],
        claims=[{"claim": "The field is 2 N/C.", "evidence_refs": ["chapter-3"]}],
    )


def test_validator_prompt_is_a_focused_audit_spec_not_the_full_transport_prompt():
    state = _state()
    prompt = TeachingValidator._build_prompt(
        state,
        "The electric field is 2 N/C.",
    )
    payload = json.loads(prompt)
    spec = payload["audit_specification"]

    assert spec["problem_to_solve"] == "根据物理题第 2 问计算电场强度并给出单位"
    assert "hw4 question paper.pdf" in spec["source_files_attached_to_this_audit_session"]
    assert any("请根据上传的物理题目计算第 2 问" in item for item in spec["extracted_request_questions_and_requirements"])
    assert any("equations" in question for question in spec["review_questions"])
    assert any("original source file" in question for question in spec["review_questions"])
    assert payload["candidate_answer_to_audit"] == "The electric field is 2 N/C."
    assert payload["evidence"][0]["identifier"] == "chapter-3"
    assert "这些原始附件会直接上传" not in prompt
    assert "Study Agent 没有预先解析" not in prompt
    assert '"question":' not in prompt


def test_validator_prompt_uses_generated_questions_when_request_is_brief():
    state = _state()
    state.question = (
        "Solve the attached question.\n\n【原始文件附件】\n"
        "文件名（JSON）：[\"problem.pdf\"]\n"
        "transport note that should not be repeated"
    )
    state.task_analysis.goal = "Solve the integral in problem 3 and explain the domain"
    prompt = json.loads(TeachingValidator._build_prompt(state, "Integral answer"))
    spec = prompt["audit_specification"]

    assert spec["problem_to_solve"] == "Solve the integral in problem 3 and explain the domain"
    assert any("necessary conditions" in q for q in spec["review_questions"])
    assert any("original source file" in q for q in spec["review_questions"])
    assert "transport note that should not be repeated" not in json.dumps(spec)



def test_validator_extracts_only_the_latest_user_turn_from_chat_history():
    state = _state()
    state.question = (
        "以下是当前 StudyAgent 对话上下文。只依据这段上下文回答最后一个用户请求。\n"
        "【用户】\nExplain transformers.\n"
        "【助手】\nEarlier explanation.\n"
        "【用户】\n请根据附件计算第 2 题的电场。\n"
        "请回答最后一个【用户】请求。\n\n"
        "【原始文件附件】\n文件名（JSON）：[\"physics.pdf\"]\n"
        "transport note"
    )
    state.task_analysis.goal = "根据附件计算第 2 题的电场"
    payload = json.loads(TeachingValidator._build_prompt(state, "E=2 N/C"))
    spec = payload["audit_specification"]

    excerpts = json.dumps(spec["extracted_request_questions_and_requirements"], ensure_ascii=False)
    assert "第 2 题的电场" in excerpts
    assert "Explain transformers" not in excerpts
    assert "Earlier explanation" not in excerpts
    assert spec["problem_to_solve"] == "根据附件计算第 2 题的电场"

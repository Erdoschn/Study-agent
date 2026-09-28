def test_core_exports_learning_apis():
    from core import (
        AssessmentEvaluator,
        GoalMatcher,
        KnowledgeGraph,
        KnowledgeNode,
        LearnerState,
        normalize_difficulty,
    )

    assert AssessmentEvaluator is not None
    assert GoalMatcher is not None
    assert KnowledgeGraph is not None
    assert KnowledgeNode is not None
    assert LearnerState is not None
    assert normalize_difficulty("postgraduate")[0] == "postgraduate"



def test_root_main_exports_cli_entrypoint():
    from main import main

    assert callable(main)


def test_ui_tag_request_does_not_enter_study_agent():
    from examples.study_agent_api import Handler

    question = """### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history, along with 1-3 more specific subtopic tags.

### Output:
JSON format: { "tags": ["tag1", "tag2", "tag3"] }

### Chat History:
<chat_history>
USER: 上下文缓存是什么？我现在想用 coder agent。
</chat_history>"""

    result = Handler._handle_ui_auxiliary_request(
        question,
        [{"role": "user", "content": "上下文缓存是什么？我现在想用 coder agent。"}],
    )

    assert result is not None
    payload = __import__("json").loads(result)
    assert "tags" in payload
    assert "Agent" in payload["tags"] or "LLM" in payload["tags"]


def test_ui_auxiliary_handler_keeps_normal_questions_out():
    from examples.study_agent_api import Handler

    result = Handler._handle_ui_auxiliary_request(
        "什么是 Transformer？",
        [{"role": "user", "content": "什么是 Transformer？"}],
    )

    assert result is None

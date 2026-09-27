"""Tests for the OpenAI-compatible Study Agent API shell."""
import json

from examples.study_agent_api import MODEL_ID, chunk, sse_event


def test_api_model_id_is_stable():
    assert MODEL_ID == "study-agent"


def test_openai_chunk_keeps_status_separate_from_answer():
    payload = chunk(reasoning="🔎 正在搜索...")
    delta = payload["choices"][0]["delta"]
    assert "reasoning_content" in delta
    assert "content" not in delta


def test_openai_chunk_emits_final_answer_only_as_content():
    payload = chunk(content="最终答案")
    delta = payload["choices"][0]["delta"]
    assert delta == {"content": "最终答案"}


def test_sse_event_is_valid_data_frame():
    raw = sse_event(chunk(content="ok")).decode("utf-8")
    assert raw.startswith("data: ")
    assert raw.endswith("\n\n")
    parsed = json.loads(raw[len("data: "):-2])
    assert parsed["choices"][0]["delta"]["content"] == "ok"

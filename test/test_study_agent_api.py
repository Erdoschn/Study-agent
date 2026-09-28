"""Tests for the OpenAI-compatible Study Agent API shell."""
import http.client
import json
import threading
from types import SimpleNamespace

from examples.study_agent_api import Handler, MODEL_ID, AgentHTTPServer, chunk, sse_event


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


def test_openai_chunk_preserves_completion_id_and_role():
    payload = chunk(content="ok", completion_id="chatcmpl-test", role="assistant")
    assert payload["id"] == "chatcmpl-test"
    assert payload["choices"][0]["delta"] == {"role": "assistant", "content": "ok"}


def test_sse_event_is_valid_data_frame():
    raw = sse_event(chunk(content="ok")).decode("utf-8")
    assert raw.startswith("data: ")
    assert raw.endswith("\n\n")
    parsed = json.loads(raw[len("data: "):-2])
    assert parsed["choices"][0]["delta"]["content"] == "ok"


def _start_server(answer="最终答案", delay=0.0):
    server = AgentHTTPServer(("127.0.0.1", 0), Handler)
    server.agent = SimpleNamespace(
        run=lambda question: (
            __import__("time").sleep(delay)
            if delay
            else None
        ) or SimpleNamespace(
            final_answer=answer,
            step_count=1,
            evidence=[],
            claims=[],
        )
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(server, method, path, payload=None):
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    content_type = response.getheader("Content-Type", "")
    conn.close()
    return response.status, content_type, raw


def test_health_endpoint():
    server, _ = _start_server()
    try:
        status, content_type, raw = _request(server, "GET", "/health")
        assert status == 200
        assert "application/json" in content_type
        assert json.loads(raw) == {"status": "ok", "model": MODEL_ID}
    finally:
        server.shutdown()
        server.server_close()


def test_models_endpoint_exposes_stable_model_id():
    server, _ = _start_server()
    try:
        status, _, raw = _request(server, "GET", "/v1/models")
        payload = json.loads(raw)
        assert status == 200
        assert payload["object"] == "list"
        assert payload["data"][0]["id"] == MODEL_ID
    finally:
        server.shutdown()
        server.server_close()


def test_non_stream_chat_completion_is_openai_compatible():
    server, _ = _start_server()
    try:
        status, content_type, raw = _request(
            server,
            "POST",
            "/v1/chat/completions",
            {
                "model": MODEL_ID,
                "stream": False,
                "messages": [{"role": "user", "content": "1+1=?"}],
            },
        )
        payload = json.loads(raw)
        assert status == 200
        assert "application/json" in content_type
        assert payload["object"] == "chat.completion"
        assert payload["model"] == MODEL_ID
        assert payload["choices"][0]["message"] == {
            "role": "assistant",
            "content": "最终答案",
        }
        assert payload["choices"][0]["finish_reason"] == "stop"
    finally:
        server.shutdown()
        server.server_close()


def test_stream_chat_completion_separates_status_and_answer():
    server, _ = _start_server()
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({
                "model": MODEL_ID,
                "stream": True,
                "messages": [{"role": "user", "content": "1+1=?"}],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        conn.close()

        assert response.status == 200
        assert response.getheader("Connection") == "close"
        assert response.getheader("Transfer-Encoding") is None
        assert response.getheader("Content-Length") is None
        assert "text/event-stream" in response.getheader("Content-Type", "")
        frames = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        assert frames[-1] == "[DONE]"
        payloads = [json.loads(frame) for frame in frames[:-1]]
        completion_ids = {payload["id"] for payload in payloads}
        assert len(completion_ids) == 1

        role_delta = payloads[0]["choices"][0]["delta"]
        status_delta = next(
            payload["choices"][0]["delta"]
            for payload in payloads
            if "reasoning_content" in payload["choices"][0]["delta"]
        )
        answer_delta = payloads[-2]["choices"][0]["delta"]
        finish = payloads[-1]["choices"][0]["finish_reason"]

        assert role_delta == {"role": "assistant"}
        assert "reasoning_content" in status_delta
        assert "content" not in status_delta
        assert answer_delta == {"role": "assistant", "content": "最终答案"}
        assert finish == "stop"
    finally:
        server.shutdown()
        server.server_close()


def test_stream_delivers_first_event_before_agent_finishes():
    server, _ = _start_server(delay=0.5)
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=2)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({
                "model": MODEL_ID,
                "stream": True,
                "messages": [{"role": "user", "content": "stream test"}],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()

        # The first SSE frame must arrive while the agent is still sleeping.
        first_line = response.fp.readline().decode("utf-8")
        assert first_line.startswith("data: ")
        first_payload = json.loads(first_line[len("data: "):])
        assert first_payload["choices"][0]["delta"] == {"role": "assistant"}

        response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


def test_stream_empty_answer_is_reported_inside_sse():
    server, _ = _start_server(answer="")
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({
                "model": MODEL_ID,
                "stream": True,
                "messages": [{"role": "user", "content": "test"}],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        conn.close()

        assert response.status == 200
        assert "[DONE]" in raw
        assert "final_answer 为空" in raw
    finally:
        server.shutdown()
        server.server_close()


def test_chat_requires_user_message():
    server, _ = _start_server()
    try:
        status, _, raw = _request(
            server,
            "POST",
            "/v1/chat/completions",
            {"model": MODEL_ID, "stream": False, "messages": []},
        )
        payload = json.loads(raw)
        assert status == 500
        assert payload["error"]["type"] == "server_error"
        assert "messages 中没有可用的 user 内容" in payload["error"]["message"]
    finally:
        server.shutdown()
        server.server_close()



def test_follow_up_prompt_is_detected_as_internal_request():
    prompt = """
    Suggest 3-5 relevant follow-up questions or prompts that the user might
    naturally ask next in this conversation as a user, based on the chat history.
    Response must be a JSON object with a "follow_ups" key.
    """
    assert Handler._is_follow_up_request(prompt) is True


def test_follow_up_generation_is_deterministic_and_does_not_need_agent():
    messages = [
        {"role": "user", "content": "我想自己做一个 agent harness"},
        {"role": "assistant", "content": "可以从工具层开始。"},
        {"role": "user", "content": "继续说说怎么设计。"},
    ]
    first = Handler._build_follow_ups(messages)
    second = Handler._build_follow_ups(messages)
    assert first == second
    assert len(first) == 3

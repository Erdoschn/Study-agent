"""Tests for the OpenAI-compatible Study Agent API shell."""
import http.client
import json
import threading
import time
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


def _start_server(answer="最终答案"):
    server = AgentHTTPServer(("127.0.0.1", 0), Handler)
    server.agent = SimpleNamespace(
        run=lambda question: SimpleNamespace(
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
        assert response.getheader("Connection") == "keep-alive"
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


def test_stream_sends_status_before_agent_finishes():
    server = AgentHTTPServer(("127.0.0.1", 0), Handler)

    def slow_run(question):
        time.sleep(0.5)
        return SimpleNamespace(
            final_answer="最终答案",
            step_count=1,
            evidence=[],
            claims=[],
        )

    server.agent = SimpleNamespace(run=slow_run)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=2)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({
                "model": MODEL_ID,
                "stream": True,
                "messages": [{"role": "user", "content": "slow"}],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        first_line = response.fp.readline().decode("utf-8")
        second_line = response.fp.readline().decode("utf-8")
        assert response.status == 200
        assert first_line.startswith("data: ")
        assert second_line.startswith("data: ")
        first_payload = json.loads(first_line[6:])
        second_payload = json.loads(second_line[6:])
        deltas = [
            first_payload["choices"][0]["delta"],
            second_payload["choices"][0]["delta"],
        ]
        assert any("reasoning_content" in delta for delta in deltas)
        conn.close()
    finally:
        server.shutdown()
        server.server_close()

from core.model_router import ModelRouter
from core.reasoner import OpenAICompatibleClient


def test_openai_compatible_client_sends_reasoning_effort(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request, timeout):
        import json
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = OpenAICompatibleClient(
        "https://example.test/v1",
        "key",
        "gpt-5.6-sol",
    )
    assert client.generate(
        "system",
        "question",
        reasoning_effort="high",
    ) == "ok"

    assert captured["payload"]["reasoning_effort"] == "high"


def test_router_call_model_passes_effort_to_supported_client():
    calls = []

    class Client:
        def generate(self, system_prompt, user_prompt, json_mode=False, reasoning_effort=None):
            calls.append(reasoning_effort)
            return "ok"

    result = ModelRouter.call_model(
        Client(),
        "system",
        "question",
        reasoning_effort="max",
        reasoning_effort_param="reasoning_effort",
    )

    assert result == "ok"
    assert calls == ["max"]


def test_router_call_model_skips_effort_for_legacy_adapter():
    calls = []

    class LegacyClient:
        def generate(self, system_prompt, user_prompt, json_mode=False):
            calls.append(json_mode)
            return "ok"

    result = ModelRouter.call_model(
        LegacyClient(),
        "system",
        "question",
        json_mode=True,
        reasoning_effort="max",
        reasoning_effort_param="reasoning_effort",
    )

    assert result == "ok"
    assert calls == [True]

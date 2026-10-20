import json
from unittest.mock import patch

import pytest

from core.reasoner import OpenAICompatibleClient


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }).encode("utf-8")


@pytest.mark.parametrize(
    ("model", "expected_temperature"),
    [
        ("gpt-5.6-luna", 1),
        ("gpt-5.6-codex", 1),
        ("openai/gpt-5.1", 1),
        ("deepseek-v4-pro", 0.1),
        ("gpt-4.1", 0.1),
    ],
)
def test_openai_compatible_client_uses_supported_temperature(
    model, expected_temperature
):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model=model,
    )

    with patch("core.reasoner.urllib.request.urlopen", side_effect=fake_urlopen):
        assert client.generate("system", "user") == "ok"

    assert captured["payload"]["temperature"] == expected_temperature

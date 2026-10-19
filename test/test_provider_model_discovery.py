from __future__ import annotations

import json

from examples import study_agent_unified_api as unified


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size=-1):
        return self.payload[:size]


def test_provider_model_discovery_adds_namespaced_models_and_keeps_configured_ones(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, request.headers, timeout))
        return _Response({"data": [{"id": "qwen3-32b"}, {"id": "deepseek-r1"}, {"id": "qwen3-32b"}]})

    monkeypatch.setattr(unified.urllib.request, "urlopen", fake_urlopen)
    config = {
        "providers": {
            "HKUST_GEN_AI": {
                "type": "openai_compatible",
                "enabled": True,
                "base_url": "https://hkust.example/v1",
                "headers": {"api-key": "secret-value"},
            }
        },
        "models": {
            "known-qwen": {"provider": "HKUST_GEN_AI", "model": "qwen3-32b", "enabled": True}
        },
    }
    discovered, status = unified.discover_provider_models(config)

    assert "known-qwen" in discovered["models"]
    assert "auto:HKUST_GEN_AI:deepseek-r1" in discovered["models"]
    assert "auto:HKUST_GEN_AI:qwen3-32b" not in discovered["models"]
    assert discovered["models"]["auto:HKUST_GEN_AI:deepseek-r1"]["pricing_unknown"] is True
    assert status["HKUST_GEN_AI"]["ok"] is True
    assert status["HKUST_GEN_AI"]["count"] == 2
    assert seen[0][0] == "https://hkust.example/v1/models"
    assert seen[0][1].get("Api-key") == "secret-value"


def test_provider_discovery_failure_does_not_drop_existing_models(monkeypatch):
    def fail_urlopen(request, timeout):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(unified.urllib.request, "urlopen", fail_urlopen)
    config = {
        "providers": {
            "provider-a": {"type": "openai_compatible", "enabled": True, "base_url": "https://a.example/v1"}
        },
        "models": {
            "configured": {"provider": "provider-a", "model": "configured-model", "enabled": True}
        },
    }
    discovered, status = unified.discover_provider_models(config)
    assert "configured" in discovered["models"]
    assert status["provider-a"]["ok"] is False
    assert "timeout" in status["provider-a"]["error"]

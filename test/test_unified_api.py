"""Tests for the unified StudyAgent API model routing."""
from examples import study_agent_unified_api as unified


def test_api_backend_config_excludes_browser_models():
    config = {
        "providers": {
            "browser": {"type": "browser", "enabled": True},
            "api": {"type": "openai_compatible", "enabled": True},
        },
        "models": {
            "deepseek-web": {"provider": "browser", "enabled": True},
            "remote-model": {"provider": "api", "enabled": True},
            "disabled-model": {"provider": "api", "enabled": False},
        },
    }
    filtered = unified.build_api_backend_config(config)
    assert set(filtered["providers"]) == {"api"}
    assert set(filtered["models"]) == {"remote-model"}


def test_api_backend_config_handles_malformed_sections():
    assert unified.build_api_backend_config({"providers": [], "models": None}) == {
        "providers": {},
        "models": {},
    }


def test_chunk_reports_selected_model():
    payload = unified.chunk("chatcmpl-test", content="ok", model=unified.API_MODEL_ID)
    assert payload["model"] == "study-agent"
    assert payload["choices"][0]["delta"]["content"] == "ok"

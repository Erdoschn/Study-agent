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



def test_single_model_config_prevents_router_fallback():
    config = {
        "providers": {
            "hkust": {"type": "openai_compatible", "enabled": True, "base_url": "https://example.invalid/v1"},
            "other": {"type": "openai_compatible", "enabled": True, "base_url": "https://other.invalid/v1"},
        },
        "models": {
            "hkust-qwen": {"provider": "hkust", "model": "qwen3-32b", "enabled": True},
            "other-model": {"provider": "other", "model": "other-model", "enabled": True},
        },
    }
    filtered = unified.build_api_backend_config(config)
    isolated = unified.build_single_model_config(filtered, "hkust-qwen")
    assert set(isolated["providers"]) == {"hkust"}
    assert set(isolated["models"]) == {"hkust-qwen"}


def test_single_model_config_rejects_unknown_or_disabled_model():
    filtered = unified.build_api_backend_config({
        "providers": {"api": {"type": "openai_compatible", "enabled": True}},
        "models": {"disabled": {"provider": "api", "enabled": False}},
    })
    try:
        unified.build_single_model_config(filtered, "disabled")
    except ValueError as exc:
        assert "未配置或不可用" in str(exc)
    else:
        raise AssertionError("disabled model should be rejected")


def test_prompt_preserves_all_turns_in_order():
    prompt = unified.Handler._prompt([
        {"role": "user", "content": "记住参数 alpha=17"},
        {"role": "assistant", "content": "已记录 alpha=17"},
        {"role": "user", "content": "我刚才的参数是多少？"},
    ])
    assert prompt.index("记住参数 alpha=17") < prompt.index("已记录 alpha=17")
    assert prompt.index("已记录 alpha=17") < prompt.index("我刚才的参数是多少？")


def test_hkust_model_detection_does_not_need_credentials():
    config = {
        "providers": {
            "HKUST_GEN_AI": {
                "type": "openai_compatible",
                "base_url": "https://hkust.azure-api.net/hkust-genai/v1",
                "api_key": "never-print-this",
            }
        },
        "models": {
            "hkust-qwen3-32b": {"provider": "HKUST_GEN_AI", "model": "qwen3-32b"}
        },
    }
    assert unified.is_hkust_model_id("hkust-qwen3-32b", config)
    assert not unified.is_hkust_model_id("some-other-model", config)

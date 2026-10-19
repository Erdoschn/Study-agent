"""Regression tests for model capability cache lifecycle."""
import time

from core.model_capabilities import ModelCapabilityCache


def _provider():
    return {
        "enabled": True,
        "type": "openai_compatible",
        "base_url": "https://provider.invalid/v1",
        "api_key": "test-key",
    }


def test_refresh_skips_disabled_models_and_prunes_inactive_cache(tmp_path, monkeypatch):
    cache = ModelCapabilityCache(tmp_path)
    cache._entries["obsolete-entry"] = {
        "provider": "old-provider",
        "model": "old-model",
        "capabilities": {"supported_levels": ["high"], "default_level": None},
        "source": "provider_metadata",
        "fetched_at": time.time(),
        "expires_at": time.time() + 3600,
        "stale": False,
    }

    def unexpected_request(*args, **kwargs):
        raise AssertionError("disabled models must not trigger provider requests")

    monkeypatch.setattr("core.model_capabilities.urllib.request.urlopen", unexpected_request)
    result = cache.refresh(
        {
            "providers": {"p": _provider()},
            "models": {
                "disabled-model": {
                    "provider": "p",
                    "model": "model-id",
                    "enabled": False,
                }
            },
        }
    )

    assert result["count"] == 0
    assert cache._entries == {}


def test_refresh_preserves_last_good_metadata_when_provider_is_unavailable(tmp_path, monkeypatch):
    cache = ModelCapabilityCache(tmp_path)
    provider = _provider()
    key = cache._key("p", provider, "model-id")
    cache._entries[key] = {
        "provider": "p",
        "model": "model-id",
        "capabilities": {
            "supported_levels": ["low", "high"],
            "default_level": "low",
            "supported": True,
        },
        "source": "provider_metadata",
        "fetched_at": time.time() - 3600,
        "expires_at": time.time() - 1,
        "stale": False,
    }

    def unavailable(*args, **kwargs):
        raise TimeoutError("simulated provider timeout")

    monkeypatch.setattr("core.model_capabilities.urllib.request.urlopen", unavailable)
    result = cache.refresh(
        {
            "providers": {"p": provider},
            "models": {
                "model-id": {
                    "provider": "p",
                    "model": "model-id",
                    "enabled": True,
                }
            },
        },
        force=True,
    )

    assert result["count"] == 1
    item = result["items"][0]
    assert item["capabilities"]["supported_levels"] == ["low", "high"]
    assert item["stale"] is True
    assert item["last_refresh_error"] == "TimeoutError"

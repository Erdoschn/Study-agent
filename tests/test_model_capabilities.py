import json
import time

from core.model_capabilities import ModelCapabilityCache


def test_levels_are_deduplicated_and_unknown_values_are_ignored():
    assert ModelCapabilityCache._levels(["low", "HIGH", "high", "bogus", None]) == ["low", "high"]


def test_extracts_provider_effort_metadata():
    levels, default = ModelCapabilityCache._extract_metadata({
        "id": "example-reasoner",
        "effort": {"supported_levels": ["low", "high", "max"], "default_level": "high"},
    })
    assert levels == ["low", "high", "max"]
    assert default == "high"


def test_cache_clear_removes_only_capability_cache(tmp_path):
    cache = ModelCapabilityCache(tmp_path)
    cache._entries["sample"] = {
        "provider": "test",
        "model": "test-model",
        "capabilities": {"supported_levels": ["low"], "default_level": "low", "supported": True},
        "source": "test",
        "fetched_at": time.time(),
        "expires_at": time.time() + 60,
    }
    cache._save()
    assert json.loads(cache.path.read_text(encoding="utf-8"))["entries"]["sample"]
    result = cache.clear()
    assert result["cleared"] is True
    assert result["bytes"] == 0
    assert not cache.path.exists()
    assert cache.snapshot()["count"] == 0


def test_refresh_uses_cached_record_until_ttl_expires(tmp_path, monkeypatch):
    cache = ModelCapabilityCache(tmp_path, ttl_seconds=3600)
    provider = {"type": "openai_compatible", "enabled": True, "base_url": "https://example.invalid/v1", "api_key": "secret"}
    config = {
        "providers": {"provider-a": provider},
        "models": {"model-a": {"provider": "provider-a", "model": "model-a", "reasoning_efforts": ["low", "high"]}},
    }
    cache.refresh(config)
    assert cache.snapshot()["count"] == 1
    record = cache.snapshot()["items"][0]
    assert record["source"] == "configured_profile"
    assert "secret" not in cache.path.read_text(encoding="utf-8")
    # The valid record should not trigger a second network request while fresh.
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh capability cache should avoid network calls")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    cache.refresh(config)
    assert cache.snapshot()["count"] == 1

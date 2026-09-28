from core.model_registry import ModelRegistry
from core.model_router import ModelRouter


def config():
    return {
        "providers": {
            "test": {
                "type": "openai_compatible",
                "base_url": "http://test",
                "api_key": "test",
                "enabled": True,
            },
            "disabled": {
                "type": "openai_compatible",
                "base_url": "http://test",
                "api_key": "test",
                "enabled": False,
            },
        },
        "models": {
            "model-a": {
                "provider": "test",
                "model": "a",
                "enabled": True,
                "paid": False,
            },
            "model-b": {
                "provider": "test",
                "model": "b",
                "enabled": True,
                "paid": True,
            },
            "model-c": {
                "provider": "disabled",
                "model": "c",
                "enabled": True,
                "paid": False,
            },
        },
    }


def test_registry_filters_disabled_provider():
    registry = ModelRegistry(config())
    names = {model.name for model in registry.available()}
    assert names == {"model-a"}


def test_paid_model_blocked_by_default():
    registry = ModelRegistry(config())
    assert all(not model.paid for model in registry.available(allow_paid=False))


def test_paid_model_can_be_allowed():
    registry = ModelRegistry(config())
    names = {model.name for model in registry.available(allow_paid=True)}
    assert names == {"model-a", "model-b"}


def test_router_selects_available_model():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)
    selection = router.select("reasoning")
    assert selection.model.name == "model-a"


def test_router_learns_capability():
    registry = ModelRegistry(config())
    registry.record_success("model-a", "reasoning")
    score = registry.get("model-a").capability_stats["reasoning"]
    assert score > 0.5
    assert registry.get("model-a").capability_successes["reasoning"] == 1
    assert registry.get("model-a").capability_failures.get("reasoning", 0) == 0


def test_registry_failure_puts_model_on_cooldown(monkeypatch):
    registry = ModelRegistry(config())
    monkeypatch.setattr("time.time", lambda: 100.0)
    registry.record_failure("model-a", "reasoning")
    assert registry.get("model-a").cooldown_until > 100.0
    assert registry.available() == []


def test_registry_success_clears_cooldown(monkeypatch):
    registry = ModelRegistry(config())
    monkeypatch.setattr("time.time", lambda: 100.0)
    registry.record_failure("model-a", "reasoning")
    assert registry.available() == []
    registry.record_success("model-a", "reasoning")
    assert registry.get("model-a").cooldown_until == 0.0
    assert registry.available()[0].name == "model-a"


def test_failure_streak_resets_after_success(monkeypatch):
    registry = ModelRegistry(config())
    monkeypatch.setattr("time.time", lambda: 100.0)
    registry.record_failure("model-a", "reasoning")
    first = registry.get("model-a").cooldown_until
    registry.get("model-a").cooldown_until = 0.0
    registry.record_failure("model-a", "reasoning")
    second = registry.get("model-a").cooldown_until
    assert second - 100.0 > first - 100.0
    registry.get("model-a").cooldown_until = 0.0
    registry.record_success("model-a", "reasoning")
    assert registry.get("model-a").failure_streak == 0
    registry.record_failure("model-a", "reasoning")
    third = registry.get("model-a").cooldown_until
    assert third - 100.0 == 5.0
    assert registry.get("model-a").failures == 3


def test_malformed_model_extra_is_ignored_safely():
    data = config()
    data["models"]["model-a"]["extra"] = ["not", "a", "dict"]
    registry = ModelRegistry(data)
    assert registry.get("model-a").extra == {}


def test_router_uses_capability_specific_reliability():
    registry = ModelRegistry(config())
    model = registry.get("model-a")
    model.capability_successes = {"teaching": 1}
    model.capability_failures = {"reasoning": 1}
    model.capability_stats = {"teaching": 2 / 3, "reasoning": 1 / 3}
    router = ModelRouter(registry)

    reasoning_score = router._score(model, "reasoning")
    teaching_score = router._score(model, "teaching")

    assert reasoning_score < teaching_score


def test_unknown_capability_is_neutral_not_artificially_inflated():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)
    assert router._score(registry.get("model-a"), "reasoning") == 0.5


def test_reliability_score_is_neutral_at_cold_start():
    registry = ModelRegistry(config())
    assert registry.get("model-a").reliability_score == 0.5


def test_repeated_failures_strongly_reduce_reliability(monkeypatch):
    registry = ModelRegistry(config())
    monkeypatch.setattr("time.time", lambda: 100.0)
    model = registry.get("model-a")

    registry.record_failure("model-a", "reasoning")
    model.cooldown_until = 0.0
    first = model.reliability_score

    registry.record_failure("model-a", "reasoning")
    model.cooldown_until = 0.0
    second = model.reliability_score

    assert first < 0.5
    assert second < first
    assert second <= 0.4


def test_success_recovers_reliability_after_failures(monkeypatch):
    registry = ModelRegistry(config())
    monkeypatch.setattr("time.time", lambda: 100.0)
    model = registry.get("model-a")

    registry.record_failure("model-a", "reasoning")
    model.cooldown_until = 0.0
    failed = model.reliability_score

    registry.record_success("model-a", "reasoning")
    assert model.reliability_score > failed
    assert model.failure_streak == 0


def test_router_penalizes_unreliable_model():
    data = config()
    data["models"]["model-b"]["paid"] = False
    registry = ModelRegistry(data)
    router = ModelRouter(registry)

    # Keep model-a and model-b otherwise comparable; only reliability differs.
    for _ in range(4):
        registry.record_success("model-b", "reasoning")
    for _ in range(3):
        registry.get("model-a").cooldown_until = 0.0
        registry.record_failure("model-a", "reasoning")
        registry.get("model-a").cooldown_until = 0.0

    assert registry.get("model-a").reliability_score < registry.get("model-b").reliability_score
    assert router._score(registry.get("model-a"), "reasoning") < router._score(registry.get("model-b"), "reasoning")


def test_runtime_evidence_is_confidence_weighted():
    registry = ModelRegistry(config())
    model = registry.get("model-a")
    for _ in range(1):
        registry.record_success("model-a", "reasoning")
    router = ModelRouter(registry)
    assert router._score(model, "reasoning") == 0.5416666666666666

    for _ in range(4):
        model.cooldown_until = 0.0
        registry.record_success("model-a", "reasoning")
    assert router._score(model, "reasoning") > 0.6


def test_router_prefers_last_successful_model_for_capability():
    registry = ModelRegistry(config())
    data = config()
    data["models"]["model-b"]["paid"] = False
    registry = ModelRegistry(data)
    router = ModelRouter(registry)

    registry.record_success("model-a", "reasoning")
    registry.get("model-a").cooldown_until = 0.0
    registry.get("model-b").cooldown_until = 0.0

    assert router.select_candidates("reasoning")[0].name == "model-a"


def test_provider_level_failure_temporarily_skips_provider():
    registry = ModelRegistry(config())
    registry.record_failure("model-a", "reasoning", provider_level=True)

    assert registry.available() == []
    assert registry.provider_cooldown_until["test"] > 0.0
    assert registry.is_provider_level_failure(TimeoutError("timed out"))
    assert registry.is_provider_level_failure(RuntimeError("LLM HTTP 503: unavailable"))
    assert not registry.is_provider_level_failure(RuntimeError("LLM HTTP 404: model retired"))


def test_efficiency_is_tracked_per_difficulty():
    registry = ModelRegistry(config())
    data = config()
    data["models"]["model-b"]["paid"] = False
    registry = ModelRegistry(data)

    for _ in range(10):
        registry.record_task_outcome("model-a", "reasoning", 4, 2, True)
        registry.record_task_outcome("model-b", "reasoning", 4, 4, True)

    assert registry.get("model-a").efficiency_score("reasoning", 4) > registry.get("model-b").efficiency_score("reasoning", 4)
    assert registry.get("model-a").efficiency_score("reasoning", 3) == 0.5


def test_failed_tasks_do_not_earn_efficiency_reward():
    registry = ModelRegistry(config())
    for _ in range(20):
        registry.record_task_outcome("model-a", "reasoning", 5, 1, False)

    assert registry.get("model-a").efficiency_score("reasoning", 5) == 0.5


def test_router_uses_same_difficulty_efficiency():
    registry = ModelRegistry(config())
    data = config()
    data["models"]["model-b"]["paid"] = False
    registry = ModelRegistry(data)

    for _ in range(10):
        registry.record_task_outcome("model-a", "reasoning", 4, 2, True)
        registry.record_task_outcome("model-b", "reasoning", 4, 5, True)

    router = ModelRouter(registry)
    assert router._score(registry.get("model-a"), "reasoning", type("A", (), {"difficulty": 4})()) > router._score(
        registry.get("model-b"), "reasoning", type("A", (), {"difficulty": 4})()
    )

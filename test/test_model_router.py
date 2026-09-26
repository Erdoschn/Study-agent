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

    models = registry.available()

    names = {model.name for model in models}

    assert names == {"model-a"}


def test_paid_model_blocked_by_default():
    registry = ModelRegistry(config())

    models = registry.available(
        allow_paid=False
    )

    assert all(
        not model.paid
        for model in models
    )


def test_paid_model_can_be_allowed():
    registry = ModelRegistry(config())

    models = registry.available(
        allow_paid=True
    )

    names = {
        model.name
        for model in models
    }

    assert names == {
        "model-a",
        "model-b",
    }


def test_router_selects_available_model():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)

    selection = router.select(
        "reasoning"
    )

    assert selection.model.name == "model-a"


def test_router_learns_capability():
    registry = ModelRegistry(config())

    registry.record_success(
        "model-a",
        "reasoning",
    )

    score = registry.get(
        "model-a"
    ).capability_stats["reasoning"]

    assert score > 0.5

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

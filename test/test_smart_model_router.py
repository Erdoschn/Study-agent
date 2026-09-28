from core.model_registry import ModelRegistry
from core.model_router import ModelRouter
from core.task_analyzer import TaskAnalysis


def config():
    return {
        "providers": {"p": {"enabled": True}},
        "models": {
            "a": {
                "provider": "p",
                "model": "a",
                "enabled": True,
                "paid": False,
            },
            "b": {
                "provider": "p",
                "model": "b",
                "enabled": True,
                "paid": False,
            },
        },
    }


def test_runtime_success_changes_model_order():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)
    before = router.select_candidates("reasoning")[0].name

    for _ in range(3):
        registry.record_success("b", "reasoning")

    after = router.select_candidates("reasoning")[0].name
    assert before != after
    assert after == "b"


def test_runtime_failure_changes_model_order():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)

    registry.record_failure("a", "reasoning")
    registry.get("a").cooldown_until = 0.0
    registry.record_failure("a", "reasoning")
    registry.get("a").cooldown_until = 0.0

    assert router.select_candidates("reasoning")[0].name == "b"


def test_difficulty_is_consumed_by_effort_selection_not_model_score():
    registry = ModelRegistry(config())
    registry.get("a").extra["reasoning_efforts"] = ["low", "high"]
    registry.get("a").extra["reasoning_effort_param"] = "reasoning_effort"
    registry.get("a").extra["benchmark"] = {
        "intelligence_index_by_effort": {"low": 30, "high": 42}
    }
    router = ModelRouter(registry)

    low = TaskAnalysis(difficulty=1)
    hard = TaskAnalysis(difficulty=5)
    assert router.select_effort(registry.get("a"), low) == "low"
    assert router.select_effort(registry.get("a"), hard) == "high"

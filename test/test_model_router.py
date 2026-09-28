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
        },
        "models": {
            "model-a": {
                "provider": "test",
                "model": "a",
                "enabled": True,
                "paid": False,
                "reasoning_efforts": ["low", "high"],
                "reasoning_effort_param": "reasoning_effort",
                "benchmark": {
                    "intelligence_index_by_effort": {"low": 30, "high": 42}
                },
            },
            "model-b": {
                "provider": "test",
                "model": "b",
                "enabled": True,
                "paid": False,
                "reasoning_efforts": ["low", "high"],
                "reasoning_effort_param": "reasoning_effort",
                "benchmark": {
                    "intelligence_index_by_effort": {"low": 20, "high": 40}
                },
            },
            "paid": {
                "provider": "test",
                "model": "paid",
                "enabled": True,
                "paid": True,
            },
        },
    }


class Analysis:
    def __init__(self, difficulty=3):
        self.difficulty = difficulty
        self.task_type = "general"
        self.required_tools = []


def test_registry_filters_paid_by_default():
    registry = ModelRegistry(config())
    assert {m.name for m in registry.available()} == {"model-a", "model-b"}


def test_router_returns_selection_with_effort():
    router = ModelRouter(ModelRegistry(config()))
    selection = router.select("reasoning", task_analysis=Analysis(3))
    assert selection.model.name == "model-a"
    assert selection.effort == "high"


def test_router_only_ranks_by_capability_call_reliability():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)

    # Make model-b historically successful and model-a historically failing.
    for _ in range(4):
        registry.record_success("model-b", "reasoning")
    for _ in range(3):
        registry.get("model-a").cooldown_until = 0.0
        registry.record_failure("model-a", "reasoning")
        registry.get("model-a").cooldown_until = 0.0

    choices = router.select_choice_candidates("reasoning", task_analysis=Analysis(5))
    assert choices[0].model.name == "model-b"
    assert choices[0].model.call_reliability_score("reasoning") > choices[1].model.call_reliability_score("reasoning")


def test_router_difficulty_changes_effort():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)
    model = registry.get("model-a")

    assert router.select_effort(model, Analysis(1)) == "low"
    assert router.select_effort(model, Analysis(3)) == "high"
    assert router.select_effort(model, Analysis(5)) == "high"


def test_router_uses_benchmark_curve_to_avoid_unnecessary_effort():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)
    model = registry.get("model-a")

    # 30/42 = 71.4%, so low effort is sufficient at difficulty 1.
    assert router.select_effort(model, Analysis(1)) == "low"
    # Difficulty 4 asks for a much higher fraction of the model peak.
    assert router.select_effort(model, Analysis(4)) == "high"


def test_router_falls_back_to_difficulty_mapping_with_one_benchmark_point():
    data = config()
    data["models"]["model-a"]["benchmark"]["intelligence_index_by_effort"] = {"high": 42}
    registry = ModelRegistry(data)
    router = ModelRouter(registry)

    assert router.select_effort(registry.get("model-a"), Analysis(1)) == "low"
    assert router.select_effort(registry.get("model-a"), Analysis(5)) == "high"


def test_call_reliability_is_neutral_at_cold_start():
    registry = ModelRegistry(config())
    assert registry.get("model-a").call_reliability_score("reasoning") == 0.5


def test_capability_call_reliability_is_separate_by_role():
    registry = ModelRegistry(config())
    registry.record_success("model-a", "teaching")
    registry.record_failure("model-a", "reasoning")
    assert registry.get("model-a").call_reliability_score("teaching") > 0.5
    assert registry.get("model-a").call_reliability_score("reasoning") < 0.5


def test_efficiency_is_not_used_by_router():
    registry = ModelRegistry(config())
    registry.record_task_outcome("model-a", "reasoning", 5, 1, True)
    registry.record_task_outcome("model-b", "reasoning", 5, 100, True)
    router = ModelRouter(registry)

    # Both are otherwise cold-started, so the router does not use efficiency
    # to override the call-score ordering.
    assert router._score(registry.get("model-a"), "reasoning") == 0.5
    assert router._score(registry.get("model-b"), "reasoning") == 0.5

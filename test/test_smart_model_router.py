from core.model_registry import ModelRegistry
from core.model_router import ModelRouter
from core.task_analyzer import TaskAnalysis


def config():
    return {
        "providers": {"p": {"enabled": True}},
        "models": {
            "math-free": {"provider": "p", "model": "math", "enabled": True, "paid": False, "capabilities": {"math": 0.95, "reasoning": 0.9, "teaching": 0.5}},
            "code-free": {"provider": "p", "model": "code", "enabled": True, "paid": False, "capabilities": {"coding": 0.95, "reasoning": 0.8}},
            "research-free": {"provider": "p", "model": "research", "enabled": True, "paid": False, "capabilities": {"research": 0.95, "reasoning": 0.8}},
            "paid-pro": {"provider": "p", "model": "pro", "enabled": True, "paid": True, "capabilities": {"math": 1.0, "coding": 1.0, "research": 1.0, "reasoning": 1.0, "teaching": 1.0}},
        },
    }


def analysis(task_type, tools=None):
    return TaskAnalysis(task_type, "test", "learn", [], [], tools or [], bool(tools and "search" in tools), "teach")


def test_math_prefers_math_capability():
    router = ModelRouter(ModelRegistry(config()))
    result = router.select_candidates("reasoning", task_analysis=analysis("math"))
    assert result[0].name == "math-free"


def test_coding_prefers_coding_capability():
    router = ModelRouter(ModelRegistry(config()))
    result = router.select_candidates("reasoning", task_analysis=analysis("coding"))
    assert result[0].name == "code-free"


def test_search_requirement_adds_research_weight():
    router = ModelRouter(ModelRegistry(config()))
    result = router.select_candidates("reasoning", task_analysis=analysis("general", ["search"]))
    assert result[0].name == "research-free"


def test_paid_is_not_selected_without_explicit_permission():
    router = ModelRouter(ModelRegistry(config()))
    result = router.select_candidates("reasoning", task_analysis=analysis("math"), allow_paid=False)
    assert all(not model.paid for model in result)


def test_capability_learning_changes_future_routing():
    registry = ModelRegistry(config())
    router = ModelRouter(registry)
    before = router.select_candidates("teaching", task_analysis=analysis("explanation"))[0].name
    registry.record_success("math-free", "teaching")
    registry.record_success("math-free", "teaching")
    after = router.select_candidates("teaching", task_analysis=analysis("explanation"))[0].name
    assert after == "math-free"
    assert before != after or registry.get("math-free").capability_stats["teaching"] > 0.5

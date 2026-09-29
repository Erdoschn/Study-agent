from core.model_router import ModelRouter
from core.task_analyzer import TaskAnalysis


def test_router_direct_for_simple_low_risk_task():
    router = ModelRouter(None)
    choice = router.select_execution_strategy(
        TaskAnalysis(task_type="general", difficulty=1)
    )
    assert choice.strategy == "direct"


def test_router_verifies_moderate_knowledge_task():
    router = ModelRouter(None)
    choice = router.select_execution_strategy(
        TaskAnalysis(task_type="conceptual", difficulty=2)
    )
    assert choice.strategy == "direct_verified"


def test_router_enters_reasoner_for_tool_or_external_fact_tasks():
    router = ModelRouter(None)

    search_task = router.select_execution_strategy(
        TaskAnalysis(
            task_type="research",
            difficulty=2,
            required_tools=["search"],
            external_facts_needed=True,
        )
    )
    assert search_task.strategy == "reasoner"

    difficult_task = router.select_execution_strategy(
        TaskAnalysis(task_type="general", difficulty=4)
    )
    assert difficult_task.strategy == "reasoner"


def test_router_does_not_require_tools_to_select_direct_verification():
    router = ModelRouter(None)
    choice = router.select_execution_strategy(
        TaskAnalysis(task_type="conceptual", difficulty=3),
        tool_available=False,
    )
    assert choice.strategy == "direct_verified"


def test_task_analysis_has_no_fixed_execution_mode():
    analysis = TaskAnalysis(task_type="general", difficulty=1)
    assert not hasattr(analysis, "execution_mode")
    assert analysis.assessment_requested is False


def test_explicit_assessment_signal_is_not_proactive():
    from core.task_analyzer import TaskAnalyzer

    assert TaskAnalyzer._is_explicit_assessment_request("解释一下 self-attention") is False
    assert TaskAnalyzer._is_explicit_assessment_request("给我一道 self-attention 的题测试一下我") is True

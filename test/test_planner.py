from core.planner import TaskPlanner
from core.task_analyzer import TaskAnalysis


def test_planner_creates_dynamic_steps():
    planner = TaskPlanner()
    analysis = TaskAnalysis(
        task_type="math",
        domain="algebra",
        goal="solve and explain",
        required_tools=["calculate", "verify"],
        external_facts_needed=False,
        answer_strategy="derive, calculate, then verify",
    )
    plan = planner.create(analysis)
    assert plan.steps
    assert plan.steps[0].action == "ANALYZE"
    assert "CALCULATE" in plan.actions
    assert "VERIFY" in plan.actions
    assert plan.steps[-1].action == "ANSWER"


def test_planner_uses_search_when_analysis_requires_it():
    planner = TaskPlanner()
    analysis = TaskAnalysis(
        task_type="research",
        domain="machine learning",
        goal="find evidence",
        required_tools=["search"],
        external_facts_needed=True,
        answer_strategy="search authoritative sources and compare evidence",
    )
    plan = planner.create(analysis)
    assert "SEARCH" in plan.actions
    assert plan.actions[-1] == "ANSWER"

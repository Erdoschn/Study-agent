from core.planner import Planner
from core.task_analyzer import TaskAnalysis


def test_planner_creates_dynamic_steps():
    planner = Planner()
    analysis = TaskAnalysis(
        task_type="math",
        domain="algebra",
        goal="solve and explain",
        required_tools=["calculate", "verify"],
        external_facts_needed=False,
        answer_strategy="derive, calculate, then verify",
    )
    plan = planner.create(analysis)
    assert plan
    assert plan[0].action == "ANALYZE"
    assert any(step.action == "CALCULATE" for step in plan)
    assert any(step.action == "VERIFY" for step in plan)
    assert plan[-1].action == "ANSWER"


def test_planner_uses_search_when_analysis_requires_it():
    planner = Planner()
    analysis = TaskAnalysis(
        task_type="research",
        domain="machine learning",
        goal="find evidence",
        required_tools=["search"],
        external_facts_needed=True,
        answer_strategy="search authoritative sources and compare evidence",
    )
    plan = planner.create(analysis)
    actions = [step.action for step in plan]
    assert "SEARCH" in actions
    assert actions[-1] == "ANSWER"

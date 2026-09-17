from dataclasses import dataclass, field

from .task_analyzer import TaskAnalysis


@dataclass
class PlanStep:
    action: str
    purpose: str
    tool: str | None = None


@dataclass
class TaskPlan:
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def actions(self) -> list[str]:
        return [step.action for step in self.steps]


class TaskPlanner:
    """Build a lightweight plan from semantic task analysis; Reasoner remains the executor."""

    def create(self, analysis: TaskAnalysis) -> TaskPlan:
        steps: list[PlanStep] = [PlanStep("ANALYZE", "确认任务目标、前提与缺口")]

        if analysis.external_facts_needed or "search" in analysis.required_tools:
            steps.append(PlanStep("SEARCH", "获取回答所需的外部证据", "search"))

        if "calculate" in analysis.required_tools:
            steps.append(PlanStep("CALCULATE", "执行需要精确计算的部分", "calculate"))

        if "verify" in analysis.required_tools or analysis.issues or analysis.external_facts_needed:
            steps.append(PlanStep("VERIFY", "检查关键结论、前提或证据", "verify"))

        steps.append(PlanStep("ANSWER", analysis.answer_strategy or "基于当前证据回答并解释"))
        return TaskPlan(steps)

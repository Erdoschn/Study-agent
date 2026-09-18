from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentStep:
    step_id: int
    action: str
    model: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning_summary: str = ""
    observation: Any = None
    success: bool = True
    error: str = ""


@dataclass
class StudentState:
    known_topics: set[str] = field(default_factory=set)
    weak_topics: set[str] = field(default_factory=set)
    misconceptions: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    question: str
    student: StudentState = field(default_factory=StudentState)
    goal: str = ""
    task_type: str = ""
    domain: str = ""
    task_analysis: Any = None
    plan: Any = None
    search_sources: list[str] = field(default_factory=list)
    search_sort_by: str = "relevance"
    current_plan_step: int = 0
    steps: list[AgentStep] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    finished: bool = False
    error: str | None = None
    max_steps: int = 8
    action_counts: dict[str, int] = field(default_factory=dict)
    last_action: str = ""
    last_observation: Any = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
        self.last_action = step.action
        self.last_observation = step.observation
        self.action_counts[step.action] = self.action_counts.get(step.action, 0) + 1

    def observations(self) -> list[Any]:
        return [step.observation for step in self.steps if step.observation is not None]

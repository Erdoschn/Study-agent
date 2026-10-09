from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoderGoal:
    description: str
    required_files: list[str] = field(default_factory=list)
    required_tests: list[str] = field(default_factory=list)
    must_modify: bool = True
    must_create_tests: bool = False
    must_pass_tests: bool = True
    # Exact paths or glob patterns approved by the Coder planning action.
    # New files are allowed when the planner adds them to this scope.
    scope_files: list[str] = field(default_factory=list)
    milestones: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)


@dataclass
class CoderStep:
    step_id: int
    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: Any = None
    success: bool = True
    error: str = ""


@dataclass
class CoderState:
    request: str
    goal: CoderGoal | None = None
    steps: list[CoderStep] = field(default_factory=list)
    modified_files: set[str] = field(default_factory=set)
    created_tests: set[str] = field(default_factory=set)
    last_test_result: dict[str, Any] | None = None
    last_observation: Any = None
    goal_verified: bool = False
    finished: bool = False
    error: str | None = None
    summary: str = ""
    modification_generation: int = 0
    test_generation: int = -1
    chat_resets: int = 0
    project: str = ""
    cancelled: bool = False
    backup_generation: int = -1
    initial_backup_generation: int = -1
    metrics: dict[str, Any] = field(default_factory=dict)
    plan_confirmed: bool = False
    current_milestone: str = ""
    completed_milestones: list[str] = field(default_factory=list)
    verified_success_criteria: list[str] = field(default_factory=list)
    asked_user_questions: list[str] = field(default_factory=list)
    user_responses: list[dict[str, str]] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, step: CoderStep) -> None:
        self.steps.append(step)
        self.last_observation = step.observation

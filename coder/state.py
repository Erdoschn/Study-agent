from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoderGoal:
    description: str
    required_files: list[str] = field(default_factory=list)
    required_tests: list[str] = field(default_factory=list)
    must_modify: bool = True
    must_create_tests: bool = True
    must_pass_tests: bool = True


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
    modification_generation: int = 0
    test_generation: int = -1
    chat_resets: int = 0
    backup_generation: int = -1
    initial_backup_generation: int = -1
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, step: CoderStep) -> None:
        self.steps.append(step)
        self.last_observation = step.observation

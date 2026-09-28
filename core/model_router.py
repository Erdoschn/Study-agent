from dataclasses import dataclass
from typing import Any

from .__debug__ import debug
from .model_registry import ModelInfo, ModelRegistry


@dataclass
class ModelSelection:
    model: ModelInfo
    reason: str


class ModelRouter:
    """根据任务需求、可选能力先验、运行证据和预算动态排序。"""

    TASK_WEIGHTS = {
        "math": {"math": 1.0, "reasoning": 0.9},
        "coding": {"coding": 1.0, "reasoning": 0.8},
        "research": {"research": 1.0, "reasoning": 0.8},
        "conceptual": {"reasoning": 0.9, "teaching": 0.7},
        "explanation": {"teaching": 1.0, "reasoning": 0.6},
        "verification": {"reasoning": 1.0, "research": 0.7},
        "general": {"general": 0.8, "reasoning": 0.6},
    }

    STATIC_WEIGHT = 0.65
    RUNTIME_WEIGHT = 0.15
    RELIABILITY_WEIGHT = 0.20
    EFFICIENCY_WEIGHT = 0.15
    RUNTIME_CONFIDENCE_OBSERVATIONS = 5
    UNKNOWN_CAPABILITY_PRIOR = 0.5

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._cursor = 0

    def select_candidates(
        self,
        capability: str,
        allow_paid: bool = False,
        exclude: set[str] | None = None,
        task_analysis: Any = None,
        plan: Any = None,
    ) -> list[ModelInfo]:
        with debug.scope("ModelRouter", f"SELECT CANDIDATES → capability={capability}, allow_paid={allow_paid}"):
            exclude = exclude or set()
            candidates = [
                m for m in self.registry.available(allow_paid=allow_paid)
                if m.name not in exclude
            ]
            if not candidates:
                return []

            scored = [(m, self._score(m, capability, task_analysis, plan)) for m in candidates]
            preferred = self.registry.last_successful_by_capability.get(capability)
            scored.sort(key=lambda item: (
                -item[1],
                0 if preferred and item[0].name == preferred else 1,
                item[0].calls,
                item[0].name,
            ))
            result = [m for m, _ in scored]

            debug.log(
                "ModelRouter",
                "ORDER → " + " → ".join(
                    f"{m.name}(score={s:.3f}, reliability={m.reliability_score:.3f})"
                    for m, s in scored
                ),
            )
            return result

    def select(
        self,
        capability: str,
        allow_paid: bool = False,
        task_analysis: Any = None,
        plan: Any = None,
    ) -> ModelSelection:
        candidates = self.select_candidates(
            capability,
            allow_paid=allow_paid,
            task_analysis=task_analysis,
            plan=plan,
        )
        if not candidates:
            raise RuntimeError("没有可用模型。")
        selected = candidates[0]
        return ModelSelection(
            selected,
            f"按任务需求、模型能力证据、历史可靠性和预算选择 {selected.name}",
        )

    def _requested_capabilities(self, capability: str, analysis: Any = None, plan: Any = None) -> dict[str, float]:
        task_type = str(getattr(analysis, "task_type", "") or "").lower()
        requested = dict(self.TASK_WEIGHTS.get(task_type, {}))
        if not requested:
            requested[capability] = 1.0
        elif capability not in requested:
            requested[capability] = 0.5

        required_tools = set(getattr(analysis, "required_tools", []) or [])
        if plan:
            required_tools.update(
                step.tool for step in getattr(plan, "steps", [])
                if getattr(step, "tool", None)
            )
        if "search" in required_tools:
            requested["research"] = max(requested.get("research", 0), 0.7)
        if "calculate" in required_tools:
            requested["math"] = max(requested.get("math", 0), 0.7)
        return requested

    @classmethod
    def _weighted_fit(cls, values: dict[str, float], requested: dict[str, float]) -> float:
        """Score all requested capabilities; unknown ones remain neutral instead of disappearing."""
        if not requested:
            return cls.UNKNOWN_CAPABILITY_PRIOR
        total = sum(requested.values())
        if total <= 0:
            return cls.UNKNOWN_CAPABILITY_PRIOR
        return sum(
            values.get(key, cls.UNKNOWN_CAPABILITY_PRIOR) * weight
            for key, weight in requested.items()
        ) / total

    @staticmethod
    def _weighted_known(values: dict[str, float], requested: dict[str, float]) -> float | None:
        pairs = [(values[key], weight) for key, weight in requested.items() if key in values]
        if not pairs:
            return None
        total = sum(weight for _, weight in pairs)
        return sum(value * weight for value, weight in pairs) / total

    def _runtime_fit(self, model: ModelInfo, requested: dict[str, float]) -> tuple[float, int]:
        observed = {}
        observations = 0
        for key in requested:
            count = model.capability_observations(key)
            if count:
                observed[key] = model.capability_stats.get(key, self.UNKNOWN_CAPABILITY_PRIOR)
                observations += count
        return self._weighted_fit(observed, requested), observations

    def _score(self, model: ModelInfo, capability: str, analysis: Any = None, plan: Any = None) -> float:
        requested = self._requested_capabilities(capability, analysis, plan)
        static_fit = self._weighted_fit(model.capabilities, requested)
        runtime_fit, observations = self._runtime_fit(model, requested)

        confidence = min(1.0, observations / self.RUNTIME_CONFIDENCE_OBSERVATIONS)
        runtime_adjusted = self.UNKNOWN_CAPABILITY_PRIOR + (
            runtime_fit - self.UNKNOWN_CAPABILITY_PRIOR
        ) * confidence

        capability_score = runtime_adjusted
        if model.capabilities:
            capability_score = self.STATIC_WEIGHT * static_fit + self.RUNTIME_WEIGHT * runtime_adjusted

        # Reliability is independent from task capability: a strong model that
        # frequently fails should not keep winning simply because its static
        # capability metadata is high.
        difficulty = getattr(analysis, "difficulty", 3) or 3
        efficiency_score = model.efficiency_score(capability, difficulty)

        # Efficiency is only a routing signal after task-level history exists.
        # Reliability remains independent and keeps failed models from returning
        # merely because they have a strong capability prior.
        capability_weight = 1.0 - self.RELIABILITY_WEIGHT - self.EFFICIENCY_WEIGHT
        return (
            capability_weight * capability_score
            + self.RELIABILITY_WEIGHT * model.reliability_score
            + self.EFFICIENCY_WEIGHT * efficiency_score
        )

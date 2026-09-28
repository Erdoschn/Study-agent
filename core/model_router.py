from dataclasses import dataclass
from typing import Any

from .__debug__ import debug
from .model_registry import ModelInfo, ModelRegistry


@dataclass
class ModelSelection:
    model: ModelInfo
    reason: str
    effort: str | None = None


class ModelRouter:
    """按任务难度选择 reasoning effort，再按真实调用可靠性排序模型。"""

    EFFORT_ORDER = ("none", "low", "medium", "high", "xhigh", "max")
    DEFAULT_EFFORT_BY_DIFFICULTY = {
        1: "low",
        2: "low",
        3: "high",
        4: "high",
        5: "max",
    }
    # When multiple public benchmark effort points exist, choose the cheapest
    # measured effort that retains this fraction of the model's measured peak.
    BENCHMARK_RETENTION_BY_DIFFICULTY = {
        1: 0.70,
        2: 0.78,
        3: 0.86,
        4: 0.93,
        5: 1.00,
    }

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._cursor = 0

    def select_choice_candidates(
        self,
        capability: str,
        allow_paid: bool = False,
        exclude: set[str] | None = None,
        task_analysis: Any = None,
        plan: Any = None,
    ) -> list[ModelSelection]:
        with debug.scope("ModelRouter", f"SELECT CHOICES → capability={capability}, allow_paid={allow_paid}"):
            exclude = exclude or set()
            candidates = [
                m for m in self.registry.available(allow_paid=allow_paid)
                if m.name not in exclude
            ]
            if not candidates:
                return []

            difficulty = self._difficulty(task_analysis)
            choices = [
                ModelSelection(
                    model=m,
                    effort=self._select_effort(m, difficulty),
                    reason=f"difficulty={difficulty}, call_reliability={m.call_reliability_score(capability):.3f}",
                )
                for m in candidates
            ]
            preferred = self.registry.last_successful_by_capability.get(capability)
            choices.sort(key=lambda choice: (
                -choice.model.call_reliability_score(capability),
                0 if preferred and choice.model.name == preferred else 1,
                choice.model.calls,
                choice.model.name,
            ))
            debug.log(
                "ModelRouter",
                "ORDER → " + " → ".join(
                    f"{choice.model.name}(call_score={choice.model.call_reliability_score(capability):.3f}, effort={choice.effort or 'default'})"
                    for choice in choices
                ),
            )
            return choices

    def select_candidates(
        self,
        capability: str,
        allow_paid: bool = False,
        exclude: set[str] | None = None,
        task_analysis: Any = None,
        plan: Any = None,
    ) -> list[ModelInfo]:
        return [
            choice.model
            for choice in self.select_choice_candidates(
                capability,
                allow_paid=allow_paid,
                exclude=exclude,
                task_analysis=task_analysis,
                plan=plan,
            )
        ]

    def select(
        self,
        capability: str,
        allow_paid: bool = False,
        task_analysis: Any = None,
        plan: Any = None,
    ) -> ModelSelection:
        choices = self.select_choice_candidates(
            capability,
            allow_paid=allow_paid,
            task_analysis=task_analysis,
            plan=plan,
        )
        if not choices:
            raise RuntimeError("没有可用模型。")
        return choices[0]

    @classmethod
    def _difficulty(cls, analysis: Any = None) -> int:
        try:
            return max(1, min(5, int(getattr(analysis, "difficulty", 3) or 3)))
        except (TypeError, ValueError):
            return 3

    @classmethod
    def _select_effort(cls, model: ModelInfo, difficulty: int) -> str | None:
        supported = [effort for effort in model.reasoning_efforts if effort in cls.EFFORT_ORDER]
        if not supported:
            return None

        scores = model.benchmark_scores
        measured = [
            (effort, scores[effort])
            for effort in supported
            if effort in scores
        ]
        # One measured point cannot establish an effort curve. Fall back to a
        # transparent difficulty→effort mapping until there are at least two.
        if len(measured) >= 2:
            peak = max(score for _, score in measured)
            target = peak * cls.BENCHMARK_RETENTION_BY_DIFFICULTY[difficulty]
            for effort in cls.EFFORT_ORDER:
                if effort in supported and effort in scores and scores[effort] >= target:
                    return effort

        desired = cls.DEFAULT_EFFORT_BY_DIFFICULTY[difficulty]
        for effort in cls.EFFORT_ORDER:
            if effort not in supported:
                continue
            if cls.EFFORT_ORDER.index(effort) >= cls.EFFORT_ORDER.index(desired):
                return effort
        return supported[-1]


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

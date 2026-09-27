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

    STATIC_WEIGHT = 0.75
    RUNTIME_WEIGHT = 0.25
    RUNTIME_CONFIDENCE_OBSERVATIONS = 5

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
            # A score is not allowed to fabricate precision. When models tie, prefer
            # the least-used model so cold-start exploration does not distort quality.
            scored.sort(key=lambda item: (-item[1], item[0].calls, item[0].name))
            result = [m for m, _ in scored]

            debug.log(
                "ModelRouter",
                "ORDER → " + " → ".join(f"{m.name}({s:.3f})" for m, s in scored),
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
        # When TaskAnalysis identifies a concrete task, its capability profile
        # is authoritative. The generic requested capability (usually
        # "reasoning") must not override coding/research/math/etc. weights.
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
                observed[key] = model.capability_stats.get(key, 0.5)
                observations += count
        fit = self._weighted_known(observed, requested)
        return (0.5 if fit is None else fit), observations

    def _score(self, model: ModelInfo, capability: str, analysis: Any = None, plan: Any = None) -> float:
        requested = self._requested_capabilities(capability, analysis, plan)

        static_fit = self._weighted_known(model.capabilities, requested)
        runtime_fit, observations = self._runtime_fit(model, requested)

        # Runtime evidence starts at a neutral prior and gains influence gradually.
        confidence = min(1.0, observations / self.RUNTIME_CONFIDENCE_OBSERVATIONS)
        runtime_adjusted = 0.5 + (runtime_fit - 0.5) * confidence

        if static_fit is None:
            # No static capability metadata: do not invent a model-specific capability score.
            # The model is selected by learned evidence; unseen models remain neutral.
            return runtime_adjusted
        return self.STATIC_WEIGHT * static_fit + self.RUNTIME_WEIGHT * runtime_adjusted

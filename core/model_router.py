from dataclasses import dataclass
from typing import Any

from .__debug__ import debug
from .model_registry import ModelInfo, ModelRegistry


@dataclass
class ModelSelection:
    model: ModelInfo
    reason: str
    effort: str | None = None


def get_model_choices(
    router,
    capability: str,
    *,
    allow_paid: bool = False,
    exclude: set[str] | None = None,
    task_analysis: Any = None,
    plan: Any = None,
) -> list[ModelSelection]:
    """Compatibility adapter for lightweight routers that still expose only select_candidates."""
    method = getattr(router, "select_choice_candidates", None)
    if callable(method):
        return method(
            capability,
            allow_paid=allow_paid,
            exclude=exclude,
            task_analysis=task_analysis,
            plan=plan,
        )

    candidates = router.select_candidates(
        capability,
        allow_paid=allow_paid,
        exclude=exclude,
        task_analysis=task_analysis,
        plan=plan,
    )
    return [
        ModelSelection(model=model, reason="legacy router adapter", effort=None)
        for model in candidates
    ]


def call_model_with_effort(
    router,
    client,
    system_prompt: str,
    user_prompt: str,
    *,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
    reasoning_effort_param: str | None = None,
) -> str:
    """Compatibility adapter for lightweight routers/clients used by tests and integrations."""
    method = getattr(router, "call_model", None)
    if callable(method):
        return method(
            client,
            system_prompt,
            user_prompt,
            json_mode=json_mode,
            reasoning_effort=reasoning_effort,
            reasoning_effort_param=reasoning_effort_param,
        )
    if reasoning_effort is None or not reasoning_effort_param:
        return client.generate(system_prompt, user_prompt, json_mode=json_mode)
    return client.generate(
        system_prompt,
        user_prompt,
        json_mode=json_mode,
        **{reasoning_effort_param: reasoning_effort},
    )


class ModelRouter:
    """按任务难度选择 reasoning effort，再按真实调用可靠性排序模型。

    Benchmark 只作为 effort 的先验，不参与模型排序；模型排序唯一使用
    capability-specific 的真实调用可靠性。
    """

    EFFORT_ORDER = ("none", "low", "medium", "high", "xhigh", "max")

    DEFAULT_EFFORT_BY_DIFFICULTY = {
        1: "low",
        2: "low",
        3: "high",
        4: "high",
        5: "max",
    }

    # For models with a measured effort curve, select the lowest measured effort
    # that retains this fraction of the model's own peak benchmark score.
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
        with debug.scope(
            "ModelRouter",
            f"SELECT CHOICES → capability={capability}, allow_paid={allow_paid}",
        ):
            exclude = exclude or set()
            models = [
                model
                for model in self.registry.available(allow_paid=allow_paid)
                if model.name not in exclude
            ]
            if not models:
                return []

            difficulty = self._difficulty(task_analysis)
            choices = [
                ModelSelection(
                    model=model,
                    effort=self._select_effort(model, difficulty),
                    reason=(
                        f"difficulty={difficulty}, "
                        f"call_reliability={model.call_reliability_score(capability):.3f}"
                    ),
                )
                for model in models
            ]

            preferred = self.registry.last_successful_by_capability.get(capability)
            choices.sort(
                key=lambda choice: (
                    -choice.model.call_reliability_score(capability),
                    0 if preferred and choice.model.name == preferred else 1,
                    choice.model.calls,
                    choice.model.name,
                )
            )

            debug.log(
                "ModelRouter",
                "ORDER → "
                + " → ".join(
                    f"{choice.model.name}"
                    f"(call_score={choice.model.call_reliability_score(capability):.3f}, "
                    f"effort={choice.effort or 'default'})"
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

    def select_effort(
        self,
        model: ModelInfo,
        task_analysis: Any = None,
    ) -> str | None:
        return self._select_effort(model, self._difficulty(task_analysis))

    @staticmethod
    def call_model(
        client,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
        reasoning_effort_param: str | None = "reasoning_effort",
    ) -> str:
        """Call clients with effort when supported; keep lightweight test adapters compatible."""
        import inspect

        if reasoning_effort is None or not reasoning_effort_param:
            return client.generate(system_prompt, user_prompt, json_mode=json_mode)

        try:
            params = inspect.signature(client.generate).parameters.values()
            supports_effort = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                or p.name == "reasoning_effort"
                for p in params
            )
        except (TypeError, ValueError):
            supports_effort = True

        if supports_effort:
            return client.generate(
                system_prompt,
                user_prompt,
                json_mode=json_mode,
                **{reasoning_effort_param: reasoning_effort},
            )
        return client.generate(system_prompt, user_prompt, json_mode=json_mode)

    @staticmethod
    def _score(model: ModelInfo, capability: str, analysis: Any = None, plan: Any = None) -> float:
        """Compatibility helper: routing score is now only actual call reliability."""
        return model.call_reliability_score(capability)

    @classmethod
    def _difficulty(cls, analysis: Any = None) -> int:
        # TaskAnalyzer is the pre-reasoner stage, so its initial call should
        # use the cheapest effort before task difficulty is known.
        if analysis is None:
            return 1
        try:
            return max(1, min(5, int(getattr(analysis, "difficulty", 3) or 3)))
        except (TypeError, ValueError):
            return 3

    @classmethod
    def _select_effort(cls, model: ModelInfo, difficulty: int) -> str | None:
        supported = [
            effort
            for effort in model.reasoning_efforts
            if effort in cls.EFFORT_ORDER
        ]
        if not supported:
            return None

        scores = model.benchmark_scores
        measured = [
            (effort, scores[effort])
            for effort in supported
            if effort in scores
        ]

        # A single benchmark point cannot identify an effort-performance curve.
        # Use a transparent difficulty mapping until at least two effort points
        # have been benchmarked for the model.
        if len(measured) >= 2:
            peak = max(score for _, score in measured)
            target = peak * cls.BENCHMARK_RETENTION_BY_DIFFICULTY[difficulty]
            for effort in cls.EFFORT_ORDER:
                if effort in supported and effort in scores and scores[effort] >= target:
                    return effort

        desired = cls.DEFAULT_EFFORT_BY_DIFFICULTY[difficulty]
        desired_index = cls.EFFORT_ORDER.index(desired)
        for effort in supported:
            if cls.EFFORT_ORDER.index(effort) >= desired_index:
                return effort
        return supported[-1]

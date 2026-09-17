from dataclasses import dataclass

from .__debug__ import debug
from .model_registry import (
    ModelInfo,
    ModelRegistry,
)


@dataclass
class ModelSelection:
    model: ModelInfo
    reason: str


class ModelRouter:

    def __init__(
        self,
        registry: ModelRegistry,
    ):
        self.registry = registry
        self._cursor = 0

    def select_candidates(
        self,
        capability: str,
        allow_paid: bool = False,
        exclude: set[str] | None = None,
    ) -> list[ModelInfo]:

        with debug.scope(
            "ModelRouter",
            (
                "SELECT CANDIDATES → "
                f"capability={capability}, "
                f"allow_paid={allow_paid}"
            ),
        ):
            exclude = exclude or set()

            candidates = [
                model
                for model in self.registry.available(
                    allow_paid=allow_paid
                )
                if model.name not in exclude
            ]

            debug.log(
                "ModelRouter",
                "AVAILABLE → "
                + ", ".join(
                    model.name
                    for model in candidates
                ),
            )

            known = [
                model
                for model in candidates
                if capability
                in model.capability_stats
            ]

            unknown = [
                model
                for model in candidates
                if capability
                not in model.capability_stats
            ]

            known.sort(
                key=lambda model: self._score(
                    model,
                    capability,
                ),
                reverse=True,
            )

            if unknown:
                start = (
                    self._cursor
                    % len(unknown)
                )

                unknown = (
                    unknown[start:]
                    + unknown[:start]
                )

                self._cursor += 1

            result = known + unknown

            debug.log(
                "ModelRouter",
                "ORDER → "
                + " → ".join(
                    model.name
                    for model in result
                ),
            )

            return result

    def select(
        self,
        capability: str,
        allow_paid: bool = False,
    ) -> ModelSelection:

        candidates = self.select_candidates(
            capability,
            allow_paid=allow_paid,
        )

        if not candidates:
            raise RuntimeError(
                "没有可用模型。"
            )

        selected = candidates[0]

        debug.log(
            "ModelRouter",
            f"SELECTED → {selected.name}",
        )

        return ModelSelection(
            model=selected,
            reason=(
                f"选择模型 {selected.name}"
            ),
        )

    @staticmethod
    def _score(
        model: ModelInfo,
        capability: str,
    ) -> float:

        capability_score = (
            model.capability_stats.get(
                capability,
                0.5,
            )
        )

        reliability = (
            model.successes
            / max(model.calls, 1)
        )

        return (
            capability_score * 0.75
            + reliability * 0.20
            + min(model.calls, 10) * 0.005
        )
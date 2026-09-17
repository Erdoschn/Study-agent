from dataclasses import dataclass
from typing import Any

from .__debug__ import debug
from .model_registry import ModelInfo, ModelRegistry


@dataclass
class ModelSelection:
    model: ModelInfo
    reason: str


class ModelRouter:
    """根据任务结构、模型能力、历史可靠性和预算动态排序。"""

    # TaskAnalyzer 输出的是结构化 task_type，不依赖问题关键词。
    TASK_WEIGHTS = {
        "math": {"math": 1.0, "reasoning": 0.9},
        "coding": {"coding": 1.0, "reasoning": 0.8},
        "research": {"research": 1.0, "reasoning": 0.8},
        "conceptual": {"reasoning": 0.9, "teaching": 0.7},
        "explanation": {"teaching": 1.0, "reasoning": 0.6},
        "verification": {"reasoning": 1.0, "research": 0.7},
        "general": {"general": 0.8, "reasoning": 0.6},
    }

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
            candidates = [m for m in self.registry.available(allow_paid=allow_paid) if m.name not in exclude]
            if not candidates:
                return []

            scores = [(m, self._score(m, capability, task_analysis, plan)) for m in candidates]
            scores.sort(key=lambda x: x[1], reverse=True)
            result = [m for m, _ in scores]

            debug.log("ModelRouter", "ORDER → " + " → ".join(f"{m.name}({s:.3f})" for m, s in scores))
            return result

    def select(self, capability: str, allow_paid: bool = False, task_analysis: Any = None, plan: Any = None) -> ModelSelection:
        candidates = self.select_candidates(capability, allow_paid=allow_paid, task_analysis=task_analysis, plan=plan)
        if not candidates:
            raise RuntimeError("没有可用模型。")
        selected = candidates[0]
        return ModelSelection(selected, f"按任务结构、{capability}能力、历史可靠性和预算选择 {selected.name}")

    def _score(self, model: ModelInfo, capability: str, analysis: Any = None, plan: Any = None) -> float:
        caps = model.capabilities
        requested = {capability: 1.0}
        task_type = str(getattr(analysis, "task_type", "") or "").lower()
        requested.update(self.TASK_WEIGHTS.get(task_type, {}))

        # 工具需求来自分析/计划，而不是关键词匹配。
        required_tools = set(getattr(analysis, "required_tools", []) or [])
        if plan:
            required_tools.update(s.tool for s in getattr(plan, "steps", []) if getattr(s, "tool", None))
        if "search" in required_tools:
            requested["research"] = max(requested.get("research", 0), 0.7)
        if "calculate" in required_tools:
            requested["math"] = max(requested.get("math", 0), 0.7)

        total_weight = sum(requested.values()) or 1.0
        capability_score = sum(caps.get(k, 0.5) * w for k, w in requested.items()) / total_weight
        reliability = model.successes / max(model.calls, 1)
        exploration = 0.05 if model.calls == 0 else min(model.calls, 10) * 0.005
        return capability_score * 0.70 + reliability * 0.25 + exploration

from dataclasses import dataclass, field
from typing import Any

from .__debug__ import debug


@dataclass
class ModelInfo:
    name: str
    provider: str
    model: str
    enabled: bool = True
    paid: bool = True
    capability_stats: dict[str, float] = field(default_factory=dict)
    capability_successes: dict[str, int] = field(default_factory=dict)
    capability_failures: dict[str, int] = field(default_factory=dict)
    calls: int = 0
    successes: int = 0
    failures: int = 0
    failure_streak: int = 0
    cooldown_until: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    efficiency_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def capabilities(self) -> dict[str, float]:
        """Legacy static capability metadata; no longer used as a routing score."""
        base = self.extra.get("capabilities", {})
        if not isinstance(base, dict):
            base = {}
        result = {}
        for key, value in base.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score != score or score in {float("inf"), float("-inf")}:
                continue
            result[str(key)] = max(0.0, min(1.0, score))
        return result

    @property
    def reasoning_efforts(self) -> list[str]:
        values = self.extra.get("reasoning_efforts", [])
        if not isinstance(values, list):
            return []
        allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
        return [str(value).lower() for value in values if str(value).lower() in allowed]

    @property
    def benchmark(self) -> dict[str, Any]:
        value = self.extra.get("benchmark", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def benchmark_scores(self) -> dict[str, float]:
        benchmark = self.benchmark
        scores = benchmark.get("intelligence_index_by_effort", benchmark.get("scores_by_effort", {}))
        if not isinstance(scores, dict):
            return {}
        result: dict[str, float] = {}
        for effort, value in scores.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score == score and score not in {float("inf"), float("-inf")}:
                result[str(effort).lower()] = score
        return result

    @property
    def reasoning_effort_param(self) -> str | None:
        value = self.extra.get("reasoning_effort_param")
        return str(value).strip() if value else None

    @property
    def reliability_score(self) -> float:
        """Runtime call reliability with a neutral prior and streak penalty.

        The prior prevents one lucky call from dominating. Repeated failures
        reduce the score quickly, while successful calls recover it gradually.
        """
        if self.calls <= 0:
            return 0.5

        # Beta(2, 2) prior: cold-start stays neutral and small samples are
        # deliberately conservative.
        score = (self.successes + 2.0) / (self.calls + 4.0)

        # A consecutive failure streak is more informative than old failures.
        # Cap the penalty so a model can recover after the outage is over.
        streak_penalty = 1.0 - min(0.50, 0.10 * self.failure_streak)
        return max(0.0, min(1.0, score * streak_penalty))

    def call_reliability_score(self, capability: str) -> float:
        """Bayesian-smoothed reliability for actual calls in one role/capability."""
        successes = int(self.capability_successes.get(capability, 0))
        failures = int(self.capability_failures.get(capability, 0))
        observations = successes + failures
        if observations <= 0:
            return self.reliability_score
        return (successes + 2.0) / (observations + 4.0)

    def capability_observations(self, capability: str) -> int:
        return (
            int(self.capability_successes.get(capability, 0))
            + int(self.capability_failures.get(capability, 0))
        )
    
    def efficiency_score(self, capability: str, difficulty: int | str = 3) -> float:
        """Return a confidence-weighted efficiency score for a capability/difficulty."""
        key = str(difficulty)
        stats = self.efficiency_stats.get(capability, {}).get(key)
        if not stats:
            return 0.5
        successes = float(stats.get("successes", 0.0))
        if successes <= 0:
            return 0.5
        avg_steps = max(1.0, float(stats.get("avg_steps", 1.0)))
        raw = 1.0 / (avg_steps ** 0.5)
        confidence = successes / (successes + 5.0)
        return max(0.0, min(1.0, 0.5 + (raw - 0.5) * confidence))


class ModelRegistry:
    """模型注册、可用性与运行统计；不负责最终选择。"""

    BASE_COOLDOWN_SECONDS = 5.0
    MAX_COOLDOWN_SECONDS = 120.0
    PROVIDER_COOLDOWN_SECONDS = 15.0

    def __init__(self, config: dict[str, Any]):
        self.models: dict[str, ModelInfo] = {}
        # Keep a hot model per capability so later turns do not cold-start
        # from the entire pool after a successful call.
        self.last_successful_by_capability: dict[str, str] = {}
        self.provider_cooldown_until: dict[str, float] = {}
        self._load(config)

    def _load(self, config: dict[str, Any]) -> None:
        providers = config.get("providers", {})
        models = config.get("models", {})
        if not isinstance(providers, dict) or not isinstance(models, dict):
            debug.log("ModelRegistry", "CONFIG INVALID → providers/models must be objects")
            return

        for name, item in models.items():
            if not isinstance(item, dict):
                debug.log("ModelRegistry", f"MODEL SKIP → {name}: config is not an object")
                continue
            provider_name = item.get("provider")
            provider = providers.get(provider_name)
            if not isinstance(provider, dict):
                debug.log("ModelRegistry", f"MODEL SKIP → {name}: provider={provider_name!r} missing/invalid")
                continue
            raw_extra = item.get("extra", {})
            extra = dict(raw_extra) if isinstance(raw_extra, dict) else {}
            if "capabilities" in item:
                capabilities = item.get("capabilities")
                if isinstance(capabilities, dict):
                    extra["capabilities"] = dict(capabilities)
                else:
                    extra.pop("capabilities", None)
            elif isinstance(extra.get("capabilities"), dict):
                extra["capabilities"] = dict(extra["capabilities"])
            else:
                extra.pop("capabilities", None)

            for key in ("reasoning_efforts", "benchmark", "reasoning_effort_param"):
                if key in item:
                    extra[key] = item[key]
            self.models[name] = ModelInfo(
                name=name,
                provider=provider_name,
                model=str(item.get("model", "")),
                enabled=bool(item.get("enabled", False)) and bool(provider.get("enabled", False)),
                paid=bool(item.get("paid", True)),
                extra=extra,
            )

    def get(self, name: str) -> ModelInfo:
        try:
            return self.models[name]
        except KeyError as exc:
            raise KeyError(f"找不到模型：{name}") from exc

    def available(self, allow_paid: bool = False) -> list[ModelInfo]:
        import time
        now = time.time()
        result = [
            model for model in self.models.values()
            if model.enabled
            and (allow_paid or not model.paid)
            and model.cooldown_until <= now
            and self.provider_cooldown_until.get(model.provider, 0.0) <= now
        ]
        debug.log(
            "ModelRegistry",
            f"AVAILABLE → allow_paid={allow_paid}, count={len(result)}",
        )
        return result

    def record_success(self, name: str, capability: str | None = None) -> None:
        model = self.get(name)
        model.calls += 1
        model.successes += 1
        model.failure_streak = 0
        model.cooldown_until = 0.0
        if capability:
            self._update_capability(model, capability, True)
            self.last_successful_by_capability[capability] = name
            self.provider_cooldown_until.pop(model.provider, None)
        debug.log(
            "ModelRegistry",
            f"SUCCESS → model={name}, capability={capability or 'none'}, calls={model.calls}, failures={model.failures}, reliability={model.reliability_score:.3f}, cooldown=0",
        )

    def record_failure(self, name: str, capability: str | None = None, provider_level: bool = False) -> None:
        import time
        model = self.get(name)
        model.calls += 1
        model.failures += 1
        model.failure_streak += 1
        streak = max(1, model.failure_streak)
        cooldown = min(self.MAX_COOLDOWN_SECONDS, self.BASE_COOLDOWN_SECONDS * (2 ** min(streak - 1, 5)))
        model.cooldown_until = time.time() + cooldown
        if provider_level:
            self.provider_cooldown_until[model.provider] = time.time() + self.PROVIDER_COOLDOWN_SECONDS
        if capability:
            self._update_capability(model, capability, False)
            if self.last_successful_by_capability.get(capability) == name:
                self.last_successful_by_capability.pop(capability, None)
        debug.log(
            "ModelRegistry",
            f"FAILURE → model={name}, capability={capability or 'none'}, failures={model.failures}, streak={model.failure_streak}, reliability={model.reliability_score:.3f}, cooldown={cooldown:.1f}s",
        )

    def efficiency_score(self, name: str, capability: str, difficulty: int | str = 3) -> float:
        """Return a model's task-efficiency score through the registry API."""
        return self.get(name).efficiency_score(capability, difficulty)

    def record_task_outcome(
        self,
        name: str,
        capability: str,
        difficulty: int | str,
        steps: int,
        success: bool,
    ) -> None:
        """Record task-level efficiency without rewarding failed tasks."""
        model = self.get(name)
        capability_stats = model.efficiency_stats.setdefault(capability, {})
        key = str(difficulty)
        stats = capability_stats.setdefault(
            key,
            {"attempts": 0.0, "successes": 0.0, "avg_steps": 0.0},
        )
        stats["attempts"] += 1.0
        if success:
            stats["successes"] += 1.0
            steps = max(1, int(steps))
            old_successes = stats["successes"] - 1.0
            stats["avg_steps"] = (
                steps if old_successes <= 0
                else ((stats["avg_steps"] * old_successes) + steps) / stats["successes"]
            )
        debug.log(
            "ModelRegistry",
            f"TASK OUTCOME → model={name}, capability={capability}, difficulty={difficulty}, steps={steps}, success={success}, efficiency={model.efficiency_score(capability, difficulty):.3f}",
        )

    @staticmethod
    def is_provider_level_failure(exc: BaseException) -> bool:
        """Classify failures where another model on the same provider is unlikely to help."""
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc or "").lower()
        provider_markers = (
            "timeout", "timed out", "connection", "urlopen",
            "http 401", "http 403", "http 408", "http 429",
            "http 500", "http 502", "http 503", "http 504",
            "temporarily unavailable", "service unavailable",
        )
        return any(marker in message for marker in provider_markers)

    @staticmethod
    def _update_capability(model: ModelInfo, capability: str, success: bool) -> None:
        """Keep a Bayesian-smoothed capability estimate with an explicit observation count."""
        if success:
            model.capability_successes[capability] = model.capability_successes.get(capability, 0) + 1
        else:
            model.capability_failures[capability] = model.capability_failures.get(capability, 0) + 1

        successes = model.capability_successes.get(capability, 0)
        failures = model.capability_failures.get(capability, 0)
        model.capability_stats[capability] = (successes + 1.0) / (successes + failures + 2.0)

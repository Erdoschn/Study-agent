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
    calls: int = 0
    successes: int = 0
    failures: int = 0
    failure_streak: int = 0
    cooldown_until: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def capabilities(self) -> dict[str, float]:
        """静态能力先验 + 运行时学习结果。学习结果优先。"""
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
        result.update(self.capability_stats)
        return result


class ModelRegistry:
    """模型注册、可用性与运行统计；不负责最终选择。"""

    BASE_COOLDOWN_SECONDS = 5.0
    MAX_COOLDOWN_SECONDS = 120.0

    def __init__(self, config: dict[str, Any]):
        self.models: dict[str, ModelInfo] = {}
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
        debug.log(
            "ModelRegistry",
            f"SUCCESS → model={name}, capability={capability or 'none'}, calls={model.calls}, failures={model.failures}, cooldown=0",
        )

    def record_failure(self, name: str, capability: str | None = None) -> None:
        import time
        model = self.get(name)
        model.calls += 1
        model.failures += 1
        model.failure_streak += 1
        streak = max(1, model.failure_streak)
        cooldown = min(self.MAX_COOLDOWN_SECONDS, self.BASE_COOLDOWN_SECONDS * (2 ** min(streak - 1, 5)))
        model.cooldown_until = time.time() + cooldown
        if capability:
            self._update_capability(model, capability, False)
        debug.log(
            "ModelRegistry",
            f"FAILURE → model={name}, capability={capability or 'none'}, failures={model.failures}, streak={model.failure_streak}, cooldown={cooldown:.1f}s",
        )

    @staticmethod
    def _update_capability(model: ModelInfo, capability: str, success: bool) -> None:
        old = model.capability_stats.get(capability, 0.5)
        target = 1.0 if success else 0.0
        model.capability_stats[capability] = old * 0.8 + target * 0.2

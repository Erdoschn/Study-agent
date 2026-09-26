from dataclasses import dataclass, field
from typing import Any


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
    cooldown_until: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def capabilities(self) -> dict[str, float]:
        """静态能力先验 + 运行时学习结果。学习结果优先。"""
        base = self.extra.get("capabilities", {})
        if not isinstance(base, dict):
            base = {}
        result = {str(k): float(v) for k, v in base.items()}
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
        for name, item in models.items():
            provider_name = item.get("provider")
            provider = providers.get(provider_name)
            if not provider:
                continue
            extra = dict(item.get("extra", {}))
            capabilities = item.get("capabilities", extra.get("capabilities", {}))
            if isinstance(capabilities, dict):
                extra["capabilities"] = capabilities
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
        return [
            model for model in self.models.values()
            if model.enabled
            and (allow_paid or not model.paid)
            and model.cooldown_until <= now
        ]

    def record_success(self, name: str, capability: str | None = None) -> None:
        model = self.get(name)
        model.calls += 1
        model.successes += 1
        model.cooldown_until = 0.0
        if capability:
            self._update_capability(model, capability, True)

    def record_failure(self, name: str, capability: str | None = None) -> None:
        import time
        model = self.get(name)
        model.calls += 1
        model.failures += 1
        failures = max(1, model.failures)
        cooldown = min(self.MAX_COOLDOWN_SECONDS, self.BASE_COOLDOWN_SECONDS * (2 ** min(failures - 1, 5)))
        model.cooldown_until = time.time() + cooldown
        if capability:
            self._update_capability(model, capability, False)
    @staticmethod
    def _update_capability(model: ModelInfo, capability: str, success: bool) -> None:
        old = model.capability_stats.get(capability, 0.5)
        target = 1.0 if success else 0.0
        model.capability_stats[capability] = old * 0.8 + target * 0.2

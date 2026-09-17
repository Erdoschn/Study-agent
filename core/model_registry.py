from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    name: str
    provider: str
    model: str

    enabled: bool = True
    paid: bool = True

    # 自动学习得到的能力记录，不要求用户手填。
    capability_stats: dict[str, float] = field(
        default_factory=dict
    )

    # 调用统计。
    calls: int = 0
    successes: int = 0
    failures: int = 0

    # 当前是否暂时不可用。
    cooldown_until: float = 0.0

    extra: dict[str, Any] = field(
        default_factory=dict
    )


class ModelRegistry:
    """
    模型注册表。

    只负责：
    - 读取模型
    - 过滤禁用模型
    - 记录模型运行状态
    - 保存自动学习得到的能力数据

    不负责最终模型选择。
    """

    def __init__(self, config: dict[str, Any]):
        self.models: dict[str, ModelInfo] = {}
        self._load(config)

    def _load(self, config: dict[str, Any]) -> None:
        providers = config.get(
            "providers",
            {},
        )

        models = config.get(
            "models",
            {},
        )

        for name, item in models.items():
            provider_name = item.get(
                "provider"
            )

            provider = providers.get(
                provider_name
            )

            if not provider:
                continue

            self.models[name] = ModelInfo(
                name=name,
                provider=provider_name,
                model=str(
                    item.get(
                        "model",
                        "",
                    )
                ),
                enabled=bool(
                    item.get(
                        "enabled",
                        False,
                    )
                )
                and bool(
                    provider.get(
                        "enabled",
                        False,
                    )
                ),
                paid=bool(
                    item.get(
                        "paid",
                        True,
                    )
                ),
            )

    def get(self, name: str) -> ModelInfo:
        try:
            return self.models[name]
        except KeyError as exc:
            raise KeyError(
                f"找不到模型：{name}"
            ) from exc

    def available(
        self,
        allow_paid: bool = False,
    ) -> list[ModelInfo]:
        import time

        now = time.time()

        result = []

        for model in self.models.values():
            if not model.enabled:
                continue

            if model.paid and not allow_paid:
                continue

            if model.cooldown_until > now:
                continue

            result.append(model)

        return result

    def record_success(
        self,
        name: str,
        capability: str | None = None,
    ) -> None:
        model = self.get(name)

        model.calls += 1
        model.successes += 1

        if capability:
            self._update_capability(
                model,
                capability,
                True,
            )

    def record_failure(
        self,
        name: str,
        capability: str | None = None,
    ) -> None:
        model = self.get(name)

        model.calls += 1
        model.failures += 1

        if capability:
            self._update_capability(
                model,
                capability,
                False,
            )

    @staticmethod
    def _update_capability(
        model: ModelInfo,
        capability: str,
        success: bool,
    ) -> None:
        old = model.capability_stats.get(
            capability,
            0.5,
        )

        target = 1.0 if success else 0.0

        # 简单增量更新，避免单次调用直接决定模型能力。
        model.capability_stats[capability] = (
            old * 0.8 + target * 0.2
        )
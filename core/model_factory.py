from typing import Any

from .__debug__ import debug
from .model_registry import ModelInfo
from .reasoner import OpenAICompatibleClient


class ModelClientFactory:

    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    def create(
        self,
        model: ModelInfo,
    ):
        with debug.scope(
            "ModelClientFactory",
            f"CREATE → {model.name}",
        ):
            providers = self.config.get(
                "providers",
                {},
            )

            provider = providers.get(
                model.provider
            )

            if not provider:
                raise RuntimeError(
                    f"找不到 Provider："
                    f"{model.provider}"
                )

            provider_type = provider.get(
                "type",
                "openai_compatible",
            )

            debug.log(
                "ModelClientFactory",
                f"provider={model.provider}, "
                f"type={provider_type}",
            )

            if provider_type != (
                "openai_compatible"
            ):
                raise RuntimeError(
                    f"暂不支持 Provider 类型："
                    f"{provider_type}"
                )

            return OpenAICompatibleClient(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=model.model,
                timeout=int(provider.get("timeout", 120)),
                headers=provider.get("headers", {}),
            )
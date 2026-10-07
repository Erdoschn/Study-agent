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
        self._browser_clients: dict[str, Any] = {}

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

            if provider_type == "browser":
                return self._create_browser_client(model, provider)

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

    def _create_browser_client(
        self,
        model: ModelInfo,
        provider: dict[str, Any],
    ):
        cached = self._browser_clients.get(model.name)
        if cached is not None:
            return cached

        from .web_model import BrowserModel

        response_selectors = provider.get("response_selectors")
        if isinstance(response_selectors, list):
            response_selectors = tuple(
                str(selector)
                for selector in response_selectors
                if str(selector).strip()
            )
        else:
            response_selectors = None

        client = BrowserModel(
            model=model.model,
            url=provider.get("url"),
            user_data_dir=provider.get("user_data_dir"),
            browser_channel=provider.get("browser_channel"),
            timeout=int(provider.get("timeout", 180)),
            response_selectors=response_selectors,
        )
        self._browser_clients[model.name] = client
        return client

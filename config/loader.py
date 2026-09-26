import json
from pathlib import Path
from typing import Any

from core.__debug__ import debug


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "providers.json"
)


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{CONFIG_PATH}"
        )

    try:
        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"providers.json 格式错误：{exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError("providers.json 顶层必须是 JSON 对象。")
    debug.log(
        "ConfigLoader",
        f"CONFIG → providers={len(config.get('providers', {}) if isinstance(config.get('providers', {}), dict) else {})}, models={len(config.get('models', {}) if isinstance(config.get('models', {}), dict) else {})}, debug={bool(config.get('debug', False))}",
    )
    return config


def setup_debug(
    config: dict[str, Any],
) -> None:
    from core.__debug__ import debug

    debug.set_enabled(
        bool(
            config.get(
                "debug",
                False,
            )
        )
    )


def get_model_config(
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:

    models = config.get(
        "models",
        {},
    )

    if not isinstance(models, dict):
        raise ValueError("models 配置必须是对象。")

    model = models.get(
        model_name
    )

    if not isinstance(model, dict):
        raise KeyError(
            f"找不到模型：{model_name}"
        )

    if not model.get(
        "enabled",
        False,
    ):
        raise RuntimeError(
            f"模型已禁用：{model_name}"
        )

    provider_name = model.get(
        "provider"
    )

    providers = config.get(
        "providers",
        {},
    )

    if not isinstance(providers, dict):
        raise ValueError("providers 配置必须是对象。")

    provider = providers.get(
        provider_name
    )

    if not isinstance(provider, dict):
        raise KeyError(
            f"找不到 Provider："
            f"{provider_name}"
        )

    if not provider.get(
        "enabled",
        False,
    ):
        raise RuntimeError(
            f"Provider 已禁用："
            f"{provider_name}"
        )

    return {
        "provider": provider_name,
        "base_url": provider["base_url"],
        "api_key": provider["api_key"],
        "headers": (
            dict(provider.get("headers", {}))
            if isinstance(provider.get("headers", {}), dict)
            else {}
        ),
        "timeout": int(provider.get("timeout", 120)),
        "model": model["model"],
        "paid": model.get(
            "paid",
            True,
        ),
    }
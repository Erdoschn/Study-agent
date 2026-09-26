import json
from pathlib import Path
from typing import Any


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
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"providers.json 格式错误：{exc}"
        ) from exc


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

    model = models.get(
        model_name
    )

    if not model:
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

    provider = providers.get(
        provider_name
    )

    if not provider:
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
        "headers": dict(provider.get("headers", {})),
        "timeout": int(provider.get("timeout", 120)),
        "model": model["model"],
        "paid": model.get(
            "paid",
            True,
        ),
    }
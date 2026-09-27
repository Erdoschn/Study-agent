import json
import urllib.request
from pathlib import Path
from typing import Any

from core.__debug__ import debug


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "providers.json"
)



def _refresh_provider_models(config: dict[str, Any]) -> None:
    """Refresh model catalogs for every enabled provider that exposes /models.

    providers.json remains the source of provider credentials and local policy.
    The live /models response is used only to discover which model IDs currently
    exist. Existing model metadata (including paid/capabilities) is preserved.
    """
    providers = config.get("providers", {})
    models = config.get("models", {})
    if not isinstance(providers, dict) or not isinstance(models, dict):
        return

    for provider_name, provider in providers.items():
        if not isinstance(provider, dict) or not provider.get("enabled", False):
            continue
        if str(provider.get("type", "openai_compatible")).lower() != "openai_compatible":
            continue

        base_url = str(provider.get("base_url", "")).rstrip("/")
        if not base_url:
            continue

        headers = {"Accept": "application/json"}
        raw_headers = provider.get("headers", {})
        if isinstance(raw_headers, dict):
            headers.update({str(k): str(v) for k, v in raw_headers.items()})
        if "Authorization" not in headers and provider.get("api_key"):
            headers["Authorization"] = f"Bearer {provider['api_key']}"

        request = urllib.request.Request(
            f"{base_url}/models",
            method="GET",
            headers=headers,
        )
        timeout = min(max(int(provider.get("timeout", 30)), 5), 30)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            debug.log(
                "ConfigLoader",
                f"MODEL REFRESH SKIP → provider={provider_name}, "
                f"{type(exc).__name__}: {exc}; keeping local models",
            )
            continue

        data = payload.get("data", []) if isinstance(payload, dict) else []
        live_ids = {
            str(item.get("id", "")).strip()
            for item in data
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        if not live_ids:
            debug.log(
                "ConfigLoader",
                f"MODEL REFRESH SKIP → provider={provider_name}, empty catalog",
            )
            continue

        provider_models = {
            name: item
            for name, item in models.items()
            if isinstance(item, dict) and item.get("provider") == provider_name
        }

        # Disable models no longer advertised by the provider, but never delete
        # them: this preserves local capability/paid metadata and makes fallback
        # possible if discovery is temporarily unavailable.
        for name, item in provider_models.items():
            model_id = str(item.get("model", name)).strip()
            if model_id not in live_ids:
                item["enabled"] = False

        for model_id in live_ids:
            # Prefer an existing exact model-id entry. Otherwise reuse an existing
            # entry pointing at this provider/model before creating a new entry.
            existing = models.get(model_id)
            if not (isinstance(existing, dict) and existing.get("provider") == provider_name):
                existing = next(
                    (
                        item for item in provider_models.values()
                        if str(item.get("model", "")).strip() == model_id
                    ),
                    None,
                )

            if isinstance(existing, dict):
                existing["model"] = model_id
                existing["enabled"] = True
                continue

            # New entries get conservative policy defaults. Paid status is only
            # inferred for newly discovered models; explicit local settings win.
            models[model_id] = {
                "provider": provider_name,
                "model": model_id,
                "enabled": True,
                "paid": not model_id.lower().endswith("-free"),
            }

        debug.log(
            "ConfigLoader",
            f"MODEL REFRESH → provider={provider_name}, "
            f"live={len(live_ids)}, configured={len(provider_models)}",
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
    _refresh_provider_models(config)
    debug.log(
        "ConfigLoader",
        f"CONFIG → providers={len(config.get('providers', {}) if isinstance(config.get('providers', {}), dict) else {})}, models={len(config.get('models', {}) if isinstance(config.get('models', {}), dict) else {})}, debug={bool(config.get('debug', False))}",
    )
    return config


def setup_debug(
    config: dict[str, Any],
) -> None:
    from core.__debug__ import debug

    enabled = bool(config.get("debug", False))
    debug.set_enabled(enabled)
    debug.log("ConfigLoader", f"DEBUG → {'ON' if enabled else 'OFF'}")


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
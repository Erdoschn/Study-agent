import json
import urllib.request
from pathlib import Path
from typing import Any

from core.__debug__ import debug


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "providers.json"
)

MODEL_PROFILES_PATH = (
    Path(__file__).resolve().parent
    / "model_profiles.json"
)



def _refresh_provider_models(config: dict[str, Any]) -> None:
    """Refresh only models explicitly configured in providers.json.

    Provider /models catalogs are used as a live availability check, not as a
    source for expanding the Agent's model pool. New provider models are never
    added automatically.
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

        configured = {
            name: item
            for name, item in models.items()
            if isinstance(item, dict) and item.get("provider") == provider_name
        }
        if not configured:
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
                f"{type(exc).__name__}: {exc}; keeping local model states",
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

        for name, item in configured.items():
            model_id = str(item.get("model", name)).strip()
            item["enabled"] = model_id in live_ids

        debug.log(
            "ConfigLoader",
            f"MODEL REFRESH → provider={provider_name}, "
            f"live={len(live_ids)}, checked={len(configured)}, added=0",
        )


def _merge_model_profiles(config: dict[str, Any]) -> None:
    """Merge tracked benchmark/effort metadata into local provider model entries."""
    if not MODEL_PROFILES_PATH.exists():
        return
    try:
        with MODEL_PROFILES_PATH.open("r", encoding="utf-8") as file:
            profile_data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        debug.log(
            "ConfigLoader",
            f"MODEL PROFILE SKIP → {type(exc).__name__}: {exc}",
        )
        return

    profiles = profile_data.get("models", {}) if isinstance(profile_data, dict) else {}
    models = config.get("models", {})
    if not isinstance(profiles, dict) or not isinstance(models, dict):
        return

    merged = 0
    for item in models.values():
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model", "")).strip()
        profile = profiles.get(model_id)
        if not isinstance(profile, dict):
            continue
        for key in ("reasoning_efforts", "reasoning_effort_param", "benchmark"):
            if key in profile:
                item[key] = profile[key]
        merged += 1

    debug.log(
        "ConfigLoader",
        f"MODEL PROFILES → matched={merged}/{len(models)}",
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
    _merge_model_profiles(config)
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